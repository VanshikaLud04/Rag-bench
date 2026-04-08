from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Optional
from ...services.rag.retriever import Retriever
from ...services.rag.context_builder import ContextBuilder
from ...services.llm.router import LLMRouter

router = APIRouter()
retriever = Retriever()
context_builder = ContextBuilder()
llm_router = LLMRouter()

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = "phi3"
    top_k: Optional[int] = None

@router.post("/")
async def query_rag(request: QueryRequest = Body(...)):
    chunks = retriever.retrieve(request.query, top_k=request.top_k)
    context = context_builder.build(request.query, chunks)
    answer = await llm_router.generate(context.formatted_context, model=request.model)

    return {
        "query": request.query,
        "answer": answer,
        "model": request.model,
        "chunks_used": len(chunks),
        "mean_retrieval_score": context.mean_score,
        "retrieved_chunks": [
            {"text": c.text[:200], "score": c.score, "source": c.source}
            for c in chunks
        ]
    }