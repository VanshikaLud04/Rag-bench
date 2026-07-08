from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
import json
import logging
from ...services.rag.retriever import Retriever
from ...services.rag.context_builder import ContextBuilder
from ...services.llm.router import LLMRouter
from ..schemas.requests import QueryRequest
from ...services.evaluation.metrics import compute_faithfulness
from ...core.prompts import PromptManager
from ...services.cache.semantic_cache import SemanticCache
from ...services.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)
router = APIRouter()
retriever = Retriever()
context_builder = ContextBuilder()
llm_router = LLMRouter()
semantic_cache = SemanticCache()
embedder = Embedder()

@router.post("/stream")
async def query_rag_stream(request: QueryRequest = Body(...)):
    top_k = request.top_k or 5
    strategy = request.strategy or "dense"
    model = request.model or "phi3"
    
    async def generator():
        query_vec = embedder.embed_single(request.query)
        cached_answer = semantic_cache.get_cache(query_vec, model)
        
        if cached_answer:
            yield f"data: {json.dumps({'type': 'token', 'content': cached_answer})}\n\n"
            meta = {
                "query": request.query,
                "model": model,
                "cached": True,
                "strategy": strategy,
                "prompt_version": "cache",
                "retrieved_chunks": [],
                "mean_retrieval_score": 1.0
            }
            yield f"data: {json.dumps({'type': 'metadata', 'content': meta})}\n\n"
            return
            
        chunks = retriever.retrieve(query=request.query, top_k=top_k, strategy=strategy)
        context = context_builder.build(request.query, chunks)
        prompt, version = PromptManager.get_prompt("rag_default", query=request.query, context=context.formatted_context)
        
        full_answer = ""
        async for chunk in llm_router.generate_stream(prompt, model=model):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
        semantic_cache.set_cache(request.query, query_vec, full_answer, model)
        
        meta = {
            "query": request.query,
            "model": model,
            "strategy": strategy,
            "prompt_version": version,
            "cached": False,
            "chunks_used": len(chunks),
            "mean_retrieval_score": context.mean_score,
            "retrieved_chunks": [
                {"text": c.text[:200], "score": c.score, "source": c.source}
                for c in chunks
            ]
        }
        yield f"data: {json.dumps({'type': 'metadata', 'content': meta})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

@router.post("/")
async def query_rag_legacy(request: QueryRequest = Body(...)):
    # Standard non-streaming logic left for compatibility if needed
    top_k = request.top_k or 5
    strategy = request.strategy or "dense"
    model = request.model or "phi3"
    
    chunks = retriever.retrieve(query=request.query, top_k=top_k, strategy=strategy)
    context = context_builder.build(request.query, chunks)
    prompt, version = PromptManager.get_prompt("rag_default", query=request.query, context=context.formatted_context)
    
    answer = await llm_router.generate(prompt, model=model)
    return {
        "query": request.query,
        "answer": answer,
        "model": model,
        "strategy": strategy,
        "prompt_version": version,
        "chunks_used": len(chunks),
        "mean_retrieval_score": context.mean_score,
        "retrieved_chunks": [
            {"text": c.text[:200], "score": c.score, "source": c.source}
            for c in chunks
        ]
    }