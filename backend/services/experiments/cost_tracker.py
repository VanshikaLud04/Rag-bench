try:
    from litellm import cost_per_token
except ImportError:
    cost_per_token = None

import logging

logger = logging.getLogger(__name__)

# Standard fallback pricing per 1M tokens (input, output)
MODEL_PRICING_PER_1M = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}

class CostTracker:
    @staticmethod
    def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of an LLM call using litellm with fallback pricing table.
        For local models (e.g. phi3, mistral, llama3), returns 0.0.
        """
        local_models = ['phi3', 'mistral', 'llama3', 'llama3.2', 'local']
        if any(model_name.lower().startswith(m) for m in local_models):
            return 0.0
            
        if cost_per_token is not None:
            try:
                mapped_model = model_name
                if "gemini" in model_name and not mapped_model.startswith("gemini/"):
                    mapped_model = f"gemini/{model_name}"
                cost, _ = cost_per_token(
                    model=mapped_model, 
                    prompt_tokens=prompt_tokens, 
                    completion_tokens=completion_tokens
                )
                return float(cost)
            except Exception as e:
                logger.warning(f"litellm cost calculation failed for {model_name}: {e}. Using fallback rates.")

        # Fallback pricing lookup
        base_name = model_name.replace("gemini/", "")
        if base_name in MODEL_PRICING_PER_1M:
            in_rate, out_rate = MODEL_PRICING_PER_1M[base_name]
            return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000, 6)
            
        return 0.0

