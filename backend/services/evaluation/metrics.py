import asyncio
import httpx
import logging
from typing import List
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

@lru_cache()
def get_model() -> SentenceTransformer:
    from ...core.config import settings
    return SentenceTransformer(settings.embedding_model)


async def _encode(text: str) -> list:
    """Offloads CPU-heavy encode() to a thread so the event loop stays free."""
    model = get_model()
    return await asyncio.to_thread(model.encode, text, normalize_embeddings=True)


async def compute_context_precision(chunks, answer: str) -> float:
    """Fraction of retrieved chunks semantically relevant to the answer."""
    if not chunks:
        return 0.0
    answer_emb = await _encode(answer)
    scores = []
    for c in chunks:
        chunk_emb = await _encode(c.text)
        scores.append(float(util.cos_sim(chunk_emb, answer_emb)))
    relevant = sum(1 for s in scores if s > 0.5)
    return relevant / len(chunks)


async def compute_context_recall(chunks, ground_truth: str) -> float:
    """Max similarity between any chunk and the ground-truth answer."""
    if not chunks or not ground_truth:
        return 0.0
    gt_emb = await _encode(ground_truth)
    scores = []
    for c in chunks:
        chunk_emb = await _encode(c.text)
        scores.append(float(util.cos_sim(chunk_emb, gt_emb)))
    return max(scores)


async def compute_faithfulness(answer: str, context_texts: List[str]) -> float:
    """
    LLM-as-judge: asks the local Ollama model whether the answer is
    grounded in the retrieved context.  Falls back to embedding overlap
    if the LLM call fails (network / model not pulled yet).
    """
    if not context_texts:
        return 0.0

    from ...core.config import settings

    context_blob = "\n\n".join(context_texts[:3])
    prompt = (
        f"Context:\n{context_blob}\n\n"
        f"Answer: {answer}\n\n"
        "Is this answer completely derived from the context above? "
        "Reply with YES or NO only, nothing else."
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": settings.ollama_models[0],
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            verdict = resp.json()["response"].strip().upper()
            logger.info("Faithfulness LLM verdict: %s", verdict)
            return 1.0 if verdict.startswith("YES") else 0.0

    except Exception as exc:

        logger.warning("LLM-as-judge failed (%s), falling back to embeddings.", exc)
        answer_emb = await _encode(answer)
        scores = []
        for ctx in context_texts:
            ctx_emb = await _encode(ctx)
            scores.append(float(util.cos_sim(ctx_emb, answer_emb)))
        return max(scores)


async def compute_answer_relevancy(query: str, answer: str) -> float:

    q_emb = await _encode(query)
    a_emb = await _encode(answer)
    return float(util.cos_sim(q_emb, a_emb))