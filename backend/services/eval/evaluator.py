from typing import List, Optional
from dataclasses import dataclass
from ..rag.retriever import RetrievedChunk
from .metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_answer_relevancy
)

@dataclass
class EvaluationResult:
    model: str
    answer: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    retrieved_chunks: List[RetrievedChunk]
    mean_retrieval_score: float

class EvaluatorService:
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        ground_truth: Optional[str] = None,
        model: str = "unknown"
    ) -> EvaluationResult:
        ctx_precision = compute_context_precision(retrieved_chunks, answer)
        ctx_recall = compute_context_recall(retrieved_chunks, ground_truth) if ground_truth else 0.0
        faithfulness = await compute_faithfulness(answer, [c.text for c in retrieved_chunks])
        relevancy = compute_answer_relevancy(query, answer)

        mean_score = (
            sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
            if retrieved_chunks else 0.0
        )

        return EvaluationResult(
            model=model,
            answer=answer,
            context_precision=round(ctx_precision, 4),
            context_recall=round(ctx_recall, 4),
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(relevancy, 4),
            retrieved_chunks=retrieved_chunks,
            mean_retrieval_score=round(mean_score, 4)
        )