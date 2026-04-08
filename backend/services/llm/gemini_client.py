import google.generativeai as genai
from ...core.config import settings

def get_gemini_model():
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(settings.gemini_model)

async def generate_gemini(prompt: str) -> str:
    model = get_gemini_model()
    response = await model.generate_content_async(prompt)
    return response.text