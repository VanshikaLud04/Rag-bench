from typing import Dict, Any, List
from ..experiments.schema import ExperimentConfig

class ConstraintSolver:
    def __init__(self, constraints: List[str]):
        """
        constraints: e.g. ["latency <= 500", "local_only = true", "cost <= 0.01"]
        """
        self.constraints = constraints
        self.local_only = any("local_only" in c.lower() and "true" in c.lower() for c in constraints)
        self.latency_limit = self._extract_limit("latency", constraints)
        self.cost_limit = self._extract_limit("cost", constraints)

    def _extract_limit(self, metric: str, constraints: List[str]) -> float:
        for c in constraints:
            if metric in c.lower() and "<=" in c:
                try:
                    val_str = c.split("<=")[1].strip().replace("ms", "").replace("$", "")
                    return float(val_str)
                except:
                    pass
        return float('inf')

    def repair(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to repair a candidate config to satisfy constraints.
        Returns the mutated config dictionary.
        """
        # 1. Local Only constraint repair
        if self.local_only:
            gen_model = config_dict.get("generation", {}).get("model_name", "")
            cloud_models = ["gpt-4o", "gemini-1.5-flash", "gpt-3.5", "claude"]
            if any(c in gen_model.lower() for c in cloud_models):
                # Swap to a local model
                config_dict["generation"]["model_name"] = "phi3"
                
        # 2. Latency constraint repair
        # If latency target is extremely tight (< 200ms), forcefully disable reranker 
        # and shrink top_k to minimize processing time.
        if self.latency_limit < 200:
            if config_dict.get("reranker", {}).get("model_name") is not None:
                config_dict["reranker"]["model_name"] = None
            if config_dict.get("retrieval", {}).get("top_k", 5) > 3:
                config_dict["retrieval"]["top_k"] = 3
                
        # 3. Cost constraint repair
        # If budget is strictly 0, force local models
        if self.cost_limit <= 0.0:
             config_dict["generation"]["model_name"] = "phi3"
             
        return config_dict
