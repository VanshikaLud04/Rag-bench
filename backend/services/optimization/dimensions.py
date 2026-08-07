from typing import List, Any, Dict
import random

class SearchDimension:
    def __init__(self, name: str, choices: List[Any], default: Any):
        self.name = name
        self.choices = choices
        self.default = default
        
    def sample(self) -> Any:
        return random.choice(self.choices)

class ChunkDimension(SearchDimension):
    def __init__(self):
        super().__init__(
            name="chunking",
            choices=[
                {"chunk_size": 256, "overlap": 32, "strategy": "fixed"},
                {"chunk_size": 512, "overlap": 64, "strategy": "fixed"},
                {"chunk_size": 1024, "overlap": 128, "strategy": "fixed"}
            ],
            default={"chunk_size": 512, "overlap": 64, "strategy": "fixed"}
        )

class RetrievalDimension(SearchDimension):
    def __init__(self):
        super().__init__(
            name="retrieval",
            choices=[
                {"strategy": "dense", "top_k": 3},
                {"strategy": "dense", "top_k": 5},
                {"strategy": "dense", "top_k": 10},
                {"strategy": "hybrid", "top_k": 3},
                {"strategy": "hybrid", "top_k": 5}
            ],
            default={"strategy": "hybrid", "top_k": 5}
        )

class EmbeddingDimension(SearchDimension):
    def __init__(self):
        super().__init__(
            name="embedding",
            choices=[
                {"model_name": "all-MiniLM-L6-v2"},
                {"model_name": "BAAI/bge-large-en-v1.5"}
            ],
            default={"model_name": "all-MiniLM-L6-v2"}
        )

class RerankerDimension(SearchDimension):
    def __init__(self):
        super().__init__(
            name="reranker",
            choices=[
                {"model_name": None},
                {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
            ],
            default={"model_name": None}
        )

class GeneratorDimension(SearchDimension):
    def __init__(self):
        super().__init__(
            name="generation",
            choices=[
                {"model_name": "phi3"},
                {"model_name": "mistral"},
                {"model_name": "gpt-4o"},
                {"model_name": "gemini-1.5-flash"}
            ],
            default={"model_name": "phi3"}
        )

def get_default_dimensions() -> Dict[str, SearchDimension]:
    return {
        "chunking": ChunkDimension(),
        "retrieval": RetrievalDimension(),
        "embedding": EmbeddingDimension(),
        "reranker": RerankerDimension(),
        "generation": GeneratorDimension()
    }
