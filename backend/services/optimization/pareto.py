from typing import List, Dict, Any, Tuple

class ParetoCalculator:
    @staticmethod
    def _is_strictly_better(run_a: Dict[str, float], run_b: Dict[str, float], objectives: Dict[str, str]) -> bool:
        """
        Returns True if run_a dominates run_b across all provided objectives.
        objectives is a dict mapping metric names to 'maximize' or 'minimize'.
        e.g. {"faithfulness": "maximize", "latency": "minimize"}
        """
        at_least_one_better = False
        
        for metric, direction in objectives.items():
            # Missing metrics must fail loudly. Defaulting them to 0.0 silently
            # distorts dominance: a missing 'minimize' metric looks perfect and
            # a missing 'maximize' metric looks dominated.
            if metric not in run_a or metric not in run_b:
                raise ValueError(
                    f"Pareto comparison requires metric {metric!r} on every run; "
                    "missing metrics are not defaulted."
                )
            val_a = run_a[metric]
            val_b = run_b[metric]
            
            if direction == "maximize":
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    at_least_one_better = True
            elif direction == "minimize":
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    at_least_one_better = True
                    
        return at_least_one_better

    @staticmethod
    def get_pareto_frontier(runs: List[Dict[str, Any]], objectives: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Returns the non-dominated set (Pareto Frontier) from the list of runs.
        Each run dict must contain a 'metrics' sub-dictionary matching the objective keys.
        """
        if not runs:
            return []
            
        for run in runs:
            metrics = run.get("metrics", {})
            missing = [m for m in objectives if m not in metrics]
            if missing:
                raise ValueError(
                    f"Run {run.get('label', '<unlabeled>')!r} is missing objective "
                    f"metrics: {missing}"
                )
            
        frontier = []
        
        for i, run_a in enumerate(runs):
            metrics_a = run_a.get("metrics", {})
            is_dominated = False
            
            for j, run_b in enumerate(runs):
                if i == j:
                    continue
                metrics_b = run_b.get("metrics", {})
                
                if ParetoCalculator._is_strictly_better(metrics_b, metrics_a, objectives):
                    is_dominated = True
                    break
                    
            if not is_dominated:
                frontier.append(run_a)
                
        return frontier
