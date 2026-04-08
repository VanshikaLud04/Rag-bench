from functools import lru_cache
from sentence_transformers import SentenceTransformer
from ...core.config import settings

@lru_cache()
def get_model():
    return SentenceTransformer(settings.embedding_model)

class Embedder:
    def __init__(self):
        self.model = get_model()

    def embed_single(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        ).tolist()