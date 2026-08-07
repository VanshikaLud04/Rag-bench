from litellm import cost_per_token
import logging

logger = logging.getLogger(__name__)

class CostTracker:
    @staticmethod
    def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of an LLM call using litellm.
        For local models (e.g. phi3, mistral, llama3), returns 0.0.
        """
        # Define local models that have zero API cost
        local_models = ['phi3', 'mistral', 'llama3', 'llama3.2']
        if any(model_name.lower().startswith(m) for m in local_models):
            return 0.0
            
        try:
            # Map common internal names to litellm compatible ones if needed
            mapped_model = model_name
            if model_name == "gpt-4o":
                mapped_model = "gpt-4o"
            elif "gemini" in model_name:
                # litellm expects format like gemini/gemini-1.5-flash
                if not mapped_model.startswith("gemini/"):
                    mapped_model = f"gemini/{model_name}"
                    
            cost, _ = cost_per_token(
                model=mapped_model, 
                prompt_tokens=prompt_tokens, 
                completion_tokens=completion_tokens
            )
            return cost
        except Exception as e:
            logger.warning(f"Could not calculate cost for model {model_name}: {e}. Defaulting to 0.0")
            return 0.0
