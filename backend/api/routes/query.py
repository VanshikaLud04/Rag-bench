from fastapi import APIRouter, Body

from typing import Optional
from ...services.rag.retriever import Retriever
from ...services.rag.context_builder import ContextBuilder
from ...services.llm.router import LLMRouter
from ..schemas.requests import QueryRequest

router = APIRouter()
retriever = Retriever()
context_builder = ContextBuilder()
llm_router = LLMRouter()



@router.post("/")
async def query_rag(request: QueryRequest = Body(...)):
    chunks = retriever.retrieve(
        query=request.query, 
        top_k=request.top_k, 
        strategy=request.strategy
    )
    context = context_builder.build(request.query, chunks)
    answer = await llm_router.generate(context.formatted_context, model=request.model)

    return {
        "query": request.query,
        "answer": answer,
        "model": request.model,
        "strategy": request.strategy,
        "chunks_used": len(chunks),
        "mean_retrieval_score": context.mean_score,
        "retrieved_chunks": [
            {"text": c.text[:200], "score": c.score, "source": c.source}
            for c in chunks
        ]
    }