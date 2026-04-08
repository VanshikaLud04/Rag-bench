from typing import List, Optional, Dict
from dataclasses import dataclass
from ...core.config import settings
from .vector_store import VectorStore
from ..ingestion.embedder import Embedder

@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str
    doc_id: str
    chunk_index: int
    metadata: Dict

class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        threshold = score_threshold or settings.retrieval_score_threshold

        query_vec = self.embedder.embed_single(query)
        results = self.vector_store.query(
            query_embedding=query_vec,
            n_results=k * 2
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)
            if score >= threshold:
                chunks.append(RetrievedChunk(
                    text=doc,
                    score=round(score, 4),
                    source=meta.get("source", "unknown"),
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                ))

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:k]