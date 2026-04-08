from ...core.config import settings
from .ollama_client import generate_ollama
from .gemini_client import generate_gemini

class LLMRouter:
    async def generate(self, prompt: str, model: str) -> str:
        if model.startswith("gemini"):
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key not set")
            return await generate_gemini(prompt)
        else:
            return await generate_ollama(prompt, model=model)