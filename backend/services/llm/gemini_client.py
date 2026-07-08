import google.generativeai as genai
from ...core.config import settings

def get_gemini_model():
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(settings.gemini_model)

async def generate_gemini(prompt: str) -> str:
    model = get_gemini_model()
    response = model.generate_content(prompt)
    return response.text

async def generate_gemini_stream(prompt: str):
    model = get_gemini_model()
    response = await model.generate_content_async(prompt, stream=True)
    async for chunk in response:
        yield chunk.text