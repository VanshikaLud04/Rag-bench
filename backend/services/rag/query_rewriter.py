import json
from typing import List
from ..llm.router import LLMRouter

class QueryRewriter:
    def __init__(self, llm_router: LLMRouter = None, model: str = "phi3"):
        self.llm = llm_router or LLMRouter()
        self.model = model

    async def rewrite(self, query: str) -> List[str]:
        # Skip rewriting for very short queries
        if len(query.split()) < 3:
            return [query]
            
        prompt = f"""
You are an expert search query assistant. The user wants to search a knowledge base.
Rewrite the following user query into up to 3 distinct search-optimized variants. 
Expand acronyms, fix ambiguity, and split compound questions if necessary.
Return the result strictly as a JSON array of strings. Do not include any other text.

User query: {query}
"""
        try:
            response_text = await self.llm.generate(prompt, self.model)
            # Try to parse JSON from response (in case of markdown formatting)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
                
            variants = json.loads(response_text)
            if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                # Ensure the original query is always part of the variants
                if query not in variants:
                    variants.insert(0, query)
                return variants[:4] # Original + 3 variants max
            return [query]
        except Exception as e:
            # Fallback to original query on failure
            return [query]
