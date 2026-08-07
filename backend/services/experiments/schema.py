from pydantic import BaseModel
from typing import Optional, List

class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    overlap: int = 50
    strategy: str = "fixed" # "fixed", "markdown", etc.

class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"

class RetrievalConfig(BaseModel):
    strategy: str = "hybrid" # "dense", "sparse", "hybrid"
    top_k: int = 5

class RerankerConfig(BaseModel):
    model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class GenerationConfig(BaseModel):
    model_name: str = "phi3"

class ExperimentConfig(BaseModel):
    dataset_id: str
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    generation: GenerationConfig = GenerationConfig()
    
    def model_dump_json(self, **kwargs):
        return super().model_dump_json(**kwargs)
