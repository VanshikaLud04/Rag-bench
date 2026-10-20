from typing import Dict, Any, List

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
                
        # 2. Latency constraints cannot be repaired reliably from configuration
        # alone.  Actual retrieval latency is measured by the runner, which then
        # filters candidates against ``latency_limit``.

        # 3. Cost constraint repair
        # If budget is strictly 0, force local models
        if self.cost_limit <= 0.0:
             config_dict["generation"]["model_name"] = "phi3"
             
        return config_dict
