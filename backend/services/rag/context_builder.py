from dataclasses import dataclass
from typing import List
from .retriever import RetrievedChunk

@dataclass
class BuiltContext:
    formatted_context: str
    chunks: List[RetrievedChunk]
    mean_score: float

class ContextBuilder:
    def build(self, query: str, chunks: List[RetrievedChunk]) -> BuiltContext:
        if not chunks:
            return BuiltContext(
                formatted_context="No relevant context found.",
                chunks=[],
                mean_score=0.0
            )

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Source: {chunk.source}\n{chunk.text}"
            )

        formatted = (
            f"Answer the following question using only the context below.\n\n"
            f"Question: {query}\n\nContext:\n" + "\n\n".join(parts)
        )

        mean_score = sum(c.score for c in chunks) / len(chunks)
        return BuiltContext(
            formatted_context=formatted,
            chunks=chunks,
            mean_score=round(mean_score, 4)
        )