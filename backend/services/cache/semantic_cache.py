import logging
import uuid
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self._cache = []  # In-memory list of dicts: {"query": str, "model": str, "embedding": np.array, "answer": str}
        logger.info("Using in-memory semantic cache for local showcase testing.")

    def get_cache(self, query_embedding: list[float], model: str) -> Optional[str]:
        if not self._cache:
            return None
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        
        best_sim = -1.0
        best_answer = None
        
        for item in self._cache:
            if item["model"] != model:
                continue
            c_vec = item["embedding"]
            # Cosine similarity
            sim = np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
            if sim > best_sim:
                best_sim = sim
                best_answer = item["answer"]
                
        if best_sim >= self.threshold:
            logger.info(f"Semantic cache hit (similarity: {best_sim:.4f})")
            return best_answer
            
        return None

    def set_cache(self, query: str, query_embedding: list[float], answer: str, model: str):
        self._cache.append({
            "query": query,
            "model": model,
            "embedding": np.array(query_embedding, dtype=np.float32),
            "answer": answer
        })
