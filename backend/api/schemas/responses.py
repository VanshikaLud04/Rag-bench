from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChunkResponse(BaseModel):
    text: str
    score: float
    source: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    model: str
    chunks_used: int
    mean_retrieval_score: float
    retrieved_chunks: List[ChunkResponse]

class ModelEvalResult(BaseModel):
    model: str
    answer: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    mean_retrieval_score: float

class EvaluateResponse(BaseModel):
    query: str
    retrieved_chunks_count: int
    results: List[ModelEvalResult]

class IngestResponse(BaseModel):
    message: str
    doc_id: str
    source: str
    chunks_ingested: int