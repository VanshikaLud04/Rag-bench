import logging
from typing import List, Optional
import openai

from ...core.config import settings
from ...core.prompts import PromptManager

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self):
        if settings.openai_api_key:
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None

    async def evaluate_faithfulness(self, answer: str, context_texts: List[str]) -> Optional[float]:
        if not self.client or not context_texts:
            return None

        context_blob = "\n\n".join(context_texts[:3])
        prompt, _ = PromptManager.get_prompt("judge_faithfulness", context=context_blob, answer=answer)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an impartial judge evaluating the faithfulness of an answer to a context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0,
            )
            verdict = response.choices[0].message.content.strip().upper()
            logger.info(f"GPT-4o Faithfulness verdict: {verdict}")
            return 1.0 if "YES" in verdict else 0.0
        except Exception as e:
            logger.error(f"GPT-4o judge failed: {e}")
            return None
