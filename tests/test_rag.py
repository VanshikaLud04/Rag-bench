import pytest
from backend.services.ingestion.embedder import Embedder
from backend.services.rag.retriever import Retriever

def test_embedder_single():
    embedder = Embedder()
    vec = embedder.embed_single("what is attention mechanism")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert isinstance(vec[0], float)

def test_embedder_batch():
    embedder = Embedder()
    vecs = embedder.embed_batch(["hello world", "test query"])
    assert len(vecs) == 2

def test_chunk_text():
    from backend.services.ingestion.document_processor import chunk_text
    chunks = chunk_text("word " * 200, chunk_size=50, overlap=10)
    assert len(chunks) > 1