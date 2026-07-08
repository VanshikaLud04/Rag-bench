from typing import Tuple
import hashlib

class PromptManager:
    """
    Manages system prompts and their versions for A/B testing and evaluation logging.
    """
    
    _prompts = {
        "rag_default": "Answer the following question using only the context below.\n\nQuestion: {query}\n\nContext:\n{context}",
        "rag_strict": "You are a strict assistant. Answer the question relying SOLELY on the provided context. Do not hallucinate.\n\nQuestion: {query}\n\nContext:\n{context}",
        "judge_faithfulness": "Context:\n{context}\n\nAnswer: {answer}\n\nIs this answer completely derived from the context above? Reply with YES or NO only, nothing else."
    }

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> Tuple[str, str]:
        """
        Returns the formatted prompt and its version hash.
        """
        template = cls._prompts.get(prompt_name)
        if not template:
            raise ValueError(f"Prompt {prompt_name} not found.")
            
        version = f"{prompt_name}_" + hashlib.md5(template.encode()).hexdigest()[:8]
        
        try:
            formatted = template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required format variable {e} for prompt {prompt_name}")
            
        return formatted, version
