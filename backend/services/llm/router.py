from ...core.config import settings
from .ollama_client import generate_ollama, generate_ollama_stream
from .gemini_client import generate_gemini, generate_gemini_stream

class LLMRouter:
    async def generate(self, prompt: str, model: str) -> str:
        if model.startswith("gemini"):
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key not set")
            return await generate_gemini(prompt)
        else:
            return await generate_ollama(prompt, model=model)

    async def generate_stream(self, prompt: str, model: str):
        if model.startswith("gemini"):
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key not set")
            async for chunk in generate_gemini_stream(prompt):
                yield chunk
        else:
            async for chunk in generate_ollama_stream(prompt, model=model):
                yield chunk