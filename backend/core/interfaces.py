from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int, **kwargs) -> List[Any]:
        """Retrieve chunks based on query."""
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

class BaseGenerator(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

class BaseEvaluator(ABC):
    @abstractmethod
    async def compute(self, query: str, contexts: List[str], answer: str, **kwargs) -> float:
        """Compute an evaluation metric score."""
        pass

class BaseOptimizer(ABC):
    @abstractmethod
    def generate_candidates(self, base_config: Dict[str, Any], n: int, constraints: Any) -> List[Dict[str, Any]]:
        """Generate candidate configurations based on the search space and constraints."""
        pass

class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, candidates: List[Dict[str, Any]], **kwargs) -> List[Any]:
        """Execute the candidate configurations and return results."""
        pass
