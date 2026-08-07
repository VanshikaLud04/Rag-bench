from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid
import logging
from ...core.database import SessionLocal
from ...core.models import Experiment, ExperimentRun, ExperimentMetrics
from ...services.experiments.schema import ExperimentConfig
from ...services.experiments.runner import ExperimentRunner
from ...services.optimization.optimizer import RandomOptimizer
from ...services.optimization.executor import ParallelExecutor
from ...services.optimization.pareto import ParetoCalculator

logger = logging.getLogger(__name__)
router = APIRouter()

class CreateExperimentRequest(BaseModel):
    name: str
    description: str = ""
    config: ExperimentConfig

@router.post("/")
async def create_experiment(request: CreateExperimentRequest):
    db = SessionLocal()
    try:
        exp_id = str(uuid.uuid4())
        experiment = Experiment(
            id=exp_id,
            name=request.name,
            description=request.description,
            config=request.config.model_dump()
        )
        db.add(experiment)
        db.commit()
        return {"status": "success", "experiment_id": exp_id}
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create experiment")
    finally:
        db.close()

@router.post("/{experiment_id}/run")
async def run_experiment(experiment_id: str, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        experiment = db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        run_id = str(uuid.uuid4())
        run = ExperimentRun(
            id=run_id,
            experiment_id=experiment_id,
            status="queued"
        )
        db.add(run)
        db.commit()
        
        # Dispatch to runner
        runner = ExperimentRunner(run_id=run_id)
        # BackgroundTasks in FastAPI cannot await async methods directly if they are heavy I/O in the same event loop, 
        # but since runner.run() is async, we can just pass it directly.
        background_tasks.add_task(runner.run)
        
        return {"status": "success", "run_id": run_id}
    except Exception as e:
        logger.error(f"Error starting experiment run: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/leaderboard")
async def get_leaderboard(sort_by: str = "faithfulness"):
    """
    Returns experiment runs ranked by a specific metric.
    Allowed sort_by values: faithfulness, precision, recall, relevancy, cost, latency
    """
    allowed_sorts = {
        "faithfulness": ExperimentMetrics.avg_faithfulness.desc(),
        "precision": ExperimentMetrics.avg_precision.desc(),
        "recall": ExperimentMetrics.avg_recall.desc(),
        "relevancy": ExperimentMetrics.avg_relevancy.desc(),
        "cost": ExperimentMetrics.total_cost.asc(),
        "latency": ExperimentMetrics.avg_latency.asc()
    }
    
    if sort_by not in allowed_sorts:
        raise HTTPException(status_code=400, detail=f"Invalid sort metric. Choose from: {list(allowed_sorts.keys())}")
        
    db = SessionLocal()
    try:
        results = (
            db.query(ExperimentRun, ExperimentMetrics, Experiment)
            .join(ExperimentMetrics, ExperimentRun.id == ExperimentMetrics.run_id)
            .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
            .filter(ExperimentRun.status == "completed")
            .order_by(allowed_sorts[sort_by])
            .all()
        )
        
        leaderboard = []
        for run, metrics, exp in results:
            leaderboard.append({
                "experiment_id": exp.id,
                "experiment_name": exp.name,
                "run_id": run.id,
                "start_time": run.start_time,
                "metrics": {
                    "avg_faithfulness": metrics.avg_faithfulness,
                    "avg_precision": metrics.avg_precision,
                    "avg_recall": metrics.avg_recall,
                    "avg_relevancy": metrics.avg_relevancy,
                    "total_cost": metrics.total_cost,
                    "avg_latency": metrics.avg_latency
                },
                "config": exp.config
            })
            
        return {"leaderboard": leaderboard}
    finally:
        db.close()

class OptimizationRequest(BaseModel):
    max_trials: int = 5
    constraints: List[str] = []

@router.post("/{experiment_id}/optimize")
async def optimize_experiment(experiment_id: str, request: OptimizationRequest):
    db = SessionLocal()
    try:
        experiment = db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        optimizer = RandomOptimizer()
        candidates = optimizer.generate_candidates(
            base_config=experiment.config, 
            n=request.max_trials, 
            constraints=request.constraints
        )
        
        executor = ParallelExecutor()
        run_ids = await executor.execute(
            candidates=candidates,
            parent_experiment_id=experiment_id,
            search_strategy="random",
            seed="default"
        )
        
        return {"status": "optimization_started", "trials_dispatched": len(run_ids), "run_ids": run_ids}
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

class ParetoRequest(BaseModel):
    objectives: dict = {"avg_faithfulness": "maximize", "avg_latency": "minimize", "total_cost": "minimize"}

@router.post("/leaderboard/pareto")
async def get_pareto_leaderboard(request: ParetoRequest):
    """
    Returns the non-dominated set of experiment runs based on the provided objectives.
    Objectives format: {"avg_faithfulness": "maximize", "avg_latency": "minimize"}
    """
    db = SessionLocal()
    try:
        results = (
            db.query(ExperimentRun, ExperimentMetrics, Experiment)
            .join(ExperimentMetrics, ExperimentRun.id == ExperimentMetrics.run_id)
            .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
            .filter(ExperimentRun.status == "completed")
            .all()
        )
        
        runs_data = []
        for run, metrics, exp in results:
            runs_data.append({
                "experiment_id": exp.id,
                "experiment_name": exp.name,
                "run_id": run.id,
                "start_time": run.start_time,
                "metrics": {
                    "avg_faithfulness": metrics.avg_faithfulness,
                    "avg_precision": metrics.avg_precision,
                    "avg_recall": metrics.avg_recall,
                    "avg_relevancy": metrics.avg_relevancy,
                    "total_cost": metrics.total_cost,
                    "avg_latency": metrics.avg_latency
                },
                "config": exp.config
            })
            
        frontier = ParetoCalculator.get_pareto_frontier(runs_data, request.objectives)
        return {"pareto_frontier": frontier}
    finally:
        db.close()
