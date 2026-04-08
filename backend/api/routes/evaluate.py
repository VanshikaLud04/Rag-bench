import asyncio
from fastapi import APIRouter, Body
from ...services.llm.router import LLMRouter
from ...services.rag.retriever import Retriever
from ...services.rag.context_builder import ContextBuilder
from ...services.evaluation.evaluator import EvaluatorService
from ..schemas.requests import EvaluateRequest
from ..schemas.responses import EvaluateResponse

router = APIRouter()
llm_router = LLMRouter()
retriever = Retriever()
context_builder = ContextBuilder()
evaluator = EvaluatorService()

@router.post("/", response_model=EvaluateResponse)
async def evaluate_rag(request: EvaluateRequest = Body(...)):
    models = request.models or ["phi3", "mistral"]
    
    chunks = retriever.retrieve(request.query, top_k=request.top_k)
    context = context_builder.build(request.query, chunks)

    tasks = [
        llm_router.generate(context.formatted_context, model=m)
        for m in models
    ]
    answers = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for model, answer in zip(models, answers):
        if isinstance(answer, Exception):
            continue
        eval_result = await evaluator.evaluate(
            query=request.query,
            answer=answer,
            retrieved_chunks=chunks,
            ground_truth=request.ground_truth,
            model=model
        )
        results.append({
            "model": eval_result.model,
            "answer": eval_result.answer,
            "context_precision": eval_result.context_precision,
            "context_recall": eval_result.context_recall,
            "faithfulness": eval_result.faithfulness,
            "answer_relevancy": eval_result.answer_relevancy,
            "mean_retrieval_score": eval_result.mean_retrieval_score,
        })

    return {
        "query": request.query,
        "retrieved_chunks_count": len(chunks),
        "results": results
    }