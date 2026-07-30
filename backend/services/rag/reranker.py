from typing import List
from sentence_transformers import CrossEncoder
from .pipeline import Reranker
from .retriever import RetrievedChunk
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return []

        # Optimization: If too few candidates, just return them or optionally still rerank
        if len(chunks) == 1:
            return chunks

        # Prepare pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]
        
        try:
            scores = self.model.predict(pairs)
            
            # Update chunks with new scores
            for i, chunk in enumerate(chunks):
                chunk.score = float(scores[i])
                
            # Sort by new cross-encoder score
            reranked_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
            return reranked_chunks
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original order
            return chunks
