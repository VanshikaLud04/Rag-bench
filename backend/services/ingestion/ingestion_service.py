from ...core.config import settings
from .document_processor import process_document
from .embedder import Embedder
from ..rag.vector_store import VectorStore

class IngestionService:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def ingest(self, file_path: str) -> dict:
        chunks = process_document(
            file_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        self.vector_store.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return {
            "doc_id": metadatas[0]["doc_id"],
            "source": metadatas[0]["source"],
            "chunks_ingested": len(chunks)
        }