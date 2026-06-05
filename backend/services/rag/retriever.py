from typing import List, Optional, Dict
from dataclasses import dataclass
import logging
from rank_bm25 import BM25Okapi
from ...core.config import settings
from .vector_store import VectorStore
from ..ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

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

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenization, lowercased
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        strategy: str = "dense",
    ) -> List[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        threshold = score_threshold or settings.retrieval_score_threshold

        if strategy == "sparse":
            return self._retrieve_sparse(query, k)
        elif strategy == "hybrid":
            return self._retrieve_hybrid(query, k, threshold)
        else:
            return self._retrieve_dense(query, k, threshold)

    def _retrieve_dense(self, query: str, k: int, threshold: float) -> List[RetrievedChunk]:
        query_vec = self.embedder.embed_single(query)
        results = self.vector_store.query(
            query_embedding=query_vec,
            n_results=k * 2
        )

        chunks = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return chunks

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

    def _retrieve_sparse(self, query: str, k: int) -> List[RetrievedChunk]:
        all_data = self.vector_store.get_all()
        docs = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])

        if not docs:
            return []

        tokenized_corpus = [self._tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self._tokenize(query)
        
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n = min(k * 2, len(docs))
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]
        
        chunks = []
        for idx in top_indices:
            score = doc_scores[idx]
            if score <= 0:
                continue
            meta = metadatas[idx]
            chunks.append(RetrievedChunk(
                text=docs[idx],
                score=round(float(score), 4),
                source=meta.get("source", "unknown"),
                doc_id=meta.get("doc_id", ""),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            ))
            
        return chunks[:k]

    def _retrieve_hybrid(self, query: str, k: int, threshold: float) -> List[RetrievedChunk]:
        dense_chunks = self._retrieve_dense(query, k * 2, threshold)
        sparse_chunks = self._retrieve_sparse(query, k * 2)
        
        # Apply RRF
        rrf_k = 60
        scores_map = {}
        chunks_map = {}
        
        for rank, chunk in enumerate(dense_chunks, 1):
            doc_identifier = f"{chunk.doc_id}_{chunk.chunk_index}"
            scores_map[doc_identifier] = scores_map.get(doc_identifier, 0) + 1.0 / (rrf_k + rank)
            chunks_map[doc_identifier] = chunk
            
        for rank, chunk in enumerate(sparse_chunks, 1):
            doc_identifier = f"{chunk.doc_id}_{chunk.chunk_index}"
            scores_map[doc_identifier] = scores_map.get(doc_identifier, 0) + 1.0 / (rrf_k + rank)
            chunks_map[doc_identifier] = chunk
            
        # Sort by RRF score
        sorted_identifiers = sorted(scores_map.keys(), key=lambda x: scores_map[x], reverse=True)
        
        hybrid_chunks = []
        for identifier in sorted_identifiers[:k]:
            chunk = chunks_map[identifier]
            chunk.score = round(scores_map[identifier], 4)
            hybrid_chunks.append(chunk)
            
        return hybrid_chunks