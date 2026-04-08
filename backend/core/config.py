from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "RagBench - RAG Evaluation Benchmark for Academic Papers"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")


    chroma_host: str = Field(default="chromadb", env="CHROMA_HOST")
    chroma_port: int = Field(default=8001, env="CHROMA_PORT")
    chroma_collection_name: str = "academic_papers"


    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64

    retrieval_top_k: int = 6
    retrieval_score_threshold: float = 0.3


    ollama_host: str = Field(default="http://ollama:11434", env="OLLAMA_HOST")
    ollama_models: List[str] = ["phi3", "mistral", "llama3.2"]
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = "gemini-1.5-flash"


    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()