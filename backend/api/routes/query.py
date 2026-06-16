from fastapi import APIRouter, Body
import logging
from ...services.rag.retriever import Retriever
from ...services.rag.context_builder import ContextBuilder
from ...services.llm.router import LLMRouter
from ..schemas.requests import QueryRequest
from ...services.evaluation.metrics import compute_faithfulness

logger = logging.getLogger(__name__)
router = APIRouter()
retriever = Retriever()
context_builder = ContextBuilder()
llm_router = LLMRouter()


@router.post("/")
async def query_rag(request: QueryRequest = Body(...)):
    top_k = request.top_k or 5
    strategy = request.strategy or "dense"
    model = request.model or "phi3"
    
    max_retries = 2
    attempts = 0
    answer = ""
    chunks = []
    mean_retrieval_score = 0.0
    
    while attempts <= max_retries:
        chunks = retriever.retrieve(
            query=request.query, 
            top_k=top_k, 
            strategy=strategy
        )
        context = context_builder.build(request.query, chunks)
        mean_retrieval_score = context.mean_score
        
        prompt_context = context.formatted_context
        if attempts > 0:
            prompt_context += "\n\nIMPORTANT: Your previous answer was not grounded in the context. You must strictly adhere to the provided context. Do not hallucinate."
            
        answer = await llm_router.generate(prompt_context, model=model)
        
        # Only apply the LLM-as-a-judge retry loop for the RagBench (hybrid) pipeline
        if strategy != "hybrid":
            break
            
        context_texts = [c.text for c in chunks]
        faithfulness = await compute_faithfulness(answer, context_texts)
        
        if faithfulness >= 0.8:
            break
            
        logger.info(f"Faithfulness score {faithfulness} is below threshold. Retrying (Attempt {attempts + 1})...")
        top_k += 2  # fetch more context on retry
        attempts += 1

    return {
        "query": request.query,
        "answer": answer,
        "model": model,
        "strategy": strategy,
        "chunks_used": len(chunks),
        "mean_retrieval_score": mean_retrieval_score,
        "retries": attempts if strategy == "hybrid" else 0,
        "retrieved_chunks": [
            {"text": c.text[:200], "score": c.score, "source": c.source}
            for c in chunks
        ]
    }