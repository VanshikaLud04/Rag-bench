import asyncio
import logging
from typing import List, Dict, Any
import uuid

from ...core.interfaces import BaseExecutor
from ...core.database import SessionLocal
from ...core.models import Experiment, ExperimentRun
from ..experiments.runner import ExperimentRunner

logger = logging.getLogger(__name__)

class ParallelExecutor(BaseExecutor):
    async def execute(
        self, 
        candidates: List[Dict[str, Any]], 
        parent_experiment_id: str,
        search_strategy: str = "random",
        seed: str = ""
    ) -> List[str]:
        """
        Executes candidate configurations concurrently.
        Returns a list of run_ids.
        """
        run_ids = []
        db = SessionLocal()
        
        try:
            parent_exp = db.query(Experiment).filter_by(id=parent_experiment_id).first()
            if not parent_exp:
                logger.error(f"Parent experiment {parent_experiment_id} not found.")
                return []

            # Create an Experiment and ExperimentRun for each candidate
            for i, config_dict in enumerate(candidates):
                exp_id = str(uuid.uuid4())
                run_id = str(uuid.uuid4())
                
                exp = Experiment(
                    id=exp_id,
                    name=f"{parent_exp.name} - Opt {search_strategy} - Trial {i+1}",
                    description=f"Auto-generated trial from optimization on {parent_experiment_id}",
                    config=config_dict
                )
                
                run = ExperimentRun(
                    id=run_id,
                    experiment_id=exp_id,
                    status="queued",
                    parent_experiment_id=parent_experiment_id,
                    trial_number=i+1,
                    search_strategy=search_strategy,
                    seed=seed,
                    config_delta={} # Simplified: ideally compute diff vs parent config
                )
                
                db.add(exp)
                db.add(run)
                run_ids.append(run_id)
                
            db.commit()
            
            # Dispatch all runners concurrently
            tasks = []
            for run_id in run_ids:
                runner = ExperimentRunner(run_id=run_id)
                tasks.append(runner.run())
                
            # Fire and forget (in a real production app, use Celery/Ray here)
            asyncio.create_task(self._run_all(tasks))
            
            return run_ids
            
        except Exception as e:
            logger.error(f"Failed to execute optimization candidates: {e}")
            db.rollback()
            return []
        finally:
            db.close()
            
    async def _run_all(self, tasks: List[asyncio.Task]):
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Optimization task failed: {res}")
