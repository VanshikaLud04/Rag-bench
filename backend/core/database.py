import chromadb
from functools import lru_cache
from .config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

@lru_cache()
def get_chroma_client():
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port
    )

def get_or_create_collection(name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./ragbench.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()