import logging
import time
from functools import wraps

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger(name)

def sanitize_text(text: str, max_len: int = 10000) -> str:
    return text.strip()[:max_len]

def timed(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        get_logger(fn.__module__).info(f"{fn.__name__} took {ms:.1f}ms")
        return result
    return wrapper