import logging
from typing import List, Dict, Any, Optional
from ...core.database import get_or_create_collection
from ...core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.collection = get_or_create_collection(self.collection_name)

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(ids)} vectors to '{self.collection_name}'")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

    def count(self) -> int:
        return self.collection.count()

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})