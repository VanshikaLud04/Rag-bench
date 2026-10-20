import copy
import json
from typing import List, Dict, Any
from ...core.interfaces import BaseOptimizer
from .dimensions import get_default_dimensions
from .constraints import ConstraintSolver

class RandomOptimizer(BaseOptimizer):
    def __init__(self):
        self.dimensions = get_default_dimensions()
        
    def generate_candidates(self, base_config: Dict[str, Any], n: int, constraints: List[str]) -> List[Dict[str, Any]]:
        solver = ConstraintSolver(constraints)
        candidates = []
        
        seen = set()
        # The configured dimensions have a finite search space.  Bound retries
        # so callers get unique configurations without risking an infinite loop.
        max_attempts = max(n * 20, 100)
        attempts = 0
        while len(candidates) < n and attempts < max_attempts:
            attempts += 1
            # Sample dimensions
            candidate = copy.deepcopy(base_config)
            
            # This is a naive translation from the search dimensions into the ExperimentConfig schema
            # In a real system, the dimensions would map directly to the Pydantic schema structure
            candidate["chunking"] = self.dimensions["chunking"].sample()
            candidate["retrieval"] = self.dimensions["retrieval"].sample()
            candidate["embedding"] = self.dimensions["embedding"].sample()
            candidate["reranker"] = self.dimensions["reranker"].sample()
            candidate["generation"] = self.dimensions["generation"].sample()
            
            # Repair using constraint solver
            candidate = solver.repair(candidate)
            key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            
        return candidates
