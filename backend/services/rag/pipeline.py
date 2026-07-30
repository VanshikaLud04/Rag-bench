from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from .retriever import RetrievedChunk

@dataclass
class Citation:
    answer_sentence: str
    chunk: RetrievedChunk
    similarity_score: float

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        pass

class ContextCompressor(ABC):
    @abstractmethod
    def compress(self, query: str, chunks: List[RetrievedChunk], budget: int) -> List[RetrievedChunk]:
        pass

class CitationEngine(ABC):
    @abstractmethod
    def ground(self, answer_sentences: List[str], chunks: List[RetrievedChunk]) -> List[Citation]:
        pass

class RetrievalPipeline(ABC):
    @abstractmethod
    def run(self, query: str) -> List[RetrievedChunk]:
        pass
