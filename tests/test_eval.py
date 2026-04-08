import pytest
from backend.services.evaluation.metrics import (
    compute_context_precision,
    compute_context_recall,
    compute_answer_relevancy
)
from backend.services.rag.retriever import RetrievedChunk

def make_chunk(text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source="test.pdf",
        doc_id="doc1",
        chunk_index=0,
        metadata={}
    )

def test_context_precision():
    chunks = [make_chunk("attention mechanism in transformers")]
    score = compute_context_precision(chunks, "transformers use attention")
    assert 0.0 <= score <= 1.0

def test_answer_relevancy():
    score = compute_answer_relevancy(
        "what is RAG",
        "RAG stands for retrieval augmented generation"
    )
    assert score > 0.5

def test_context_recall_no_ground_truth():
    from backend.services.evaluation.metrics import compute_context_recall
    score = compute_context_recall([], None)
    assert score == 0.0