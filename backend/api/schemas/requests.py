from pydantic import BaseModel, Field
from typing import Optional, List

class IngestRequest(BaseModel):
    collection_name: Optional[str] = None

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    model: Optional[str] = "phi3"
    top_k: Optional[int] = Field(default=6, ge=1, le=20)

class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=3)
    ground_truth: Optional[str] = None
    models: Optional[List[str]] = None
    top_k: Optional[int] = Field(default=6, ge=1, le=20)