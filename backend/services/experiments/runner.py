import asyncio
import hashlib
import time
import logging
from typing import List, Dict, Any
import json
import uuid

from ...core.database import SessionLocal
from ...core.models import ExperimentRun, ExperimentMetrics, DatasetItem, Experiment
from ...services.rag.retriever import Retriever
from ...services.rag.reranker import CrossEncoderReranker
from ...services.rag.compressor import ExtractiveContextCompressor
from ...services.rag.vector_store import VectorStore
from ...services.ingestion.embedder import Embedder
from ...services.llm.router import LLMRouter
from ...services.rag.context_builder import ContextBuilder
from ...core.prompts import PromptManager
from ...services.evaluation.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ...services.evaluation.judge import LLMJudge
from .schema import ExperimentConfig
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, run_id: str):
        self.run_id = run_id
        
    def _get_collection_name(self, config: ExperimentConfig) -> str:
        # Create a deterministic hash for the collection name based on parameters that affect ingestion
        seed_str = f"{config.dataset_id}_{config.chunking.chunk_size}_{config.chunking.overlap}_{config.chunking.strategy}_{config.embedding.model_name}"
        hash_str = hashlib.md5(seed_str.encode()).hexdigest()
        return f"exp_{hash_str}"

    async def run(self):
        db = SessionLocal()
        run = db.query(ExperimentRun).filter_by(id=self.run_id).first()
        if not run:
            logger.error(f"Run {self.run_id} not found.")
            db.close()
            return
            
        experiment = run.experiment
        config = ExperimentConfig.parse_obj(experiment.config)
        
        try:
            run.status = "running"
            db.commit()
            
            # Setup services according to config
            collection_name = self._get_collection_name(config)
            
            # TODO: In a full system, we would check if this collection exists,
            # and if not, we would fetch documents related to the dataset and ingest them
            # using the specified chunking/embedding logic.
            # For now, we assume the vector store points to the right place or is pre-populated.
            vector_store = VectorStore(collection_name=collection_name)
            embedder = Embedder() # In reality, configure with config.embedding.model_name
            retriever = Retriever()
            retriever.vector_store = vector_store
            retriever.embedder = embedder
            
            reranker = None
            if config.reranker and config.reranker.model_name:
                reranker = CrossEncoderReranker()
                
            compressor = ExtractiveContextCompressor()
            context_builder = ContextBuilder()
            llm_router = LLMRouter()
            
            # Setup evaluators
            judge = LLMJudge()
            evaluators = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]
            
            dataset_items = db.query(DatasetItem).filter_by(dataset_id=config.dataset_id).all()
            
            total_latency = 0
            total_cost = 0.0
            metric_sums = {
                "ContextPrecision": 0.0,
                "ContextRecall": 0.0,
                "Faithfulness": 0.0,
                "AnswerRelevancy": 0.0
            }
            
            processed_count = 0
            
            for item in dataset_items:
                start_time = time.time()
                
                # 1. Retrieve
                chunks = retriever.retrieve(
                    query=item.query, 
                    top_k=config.retrieval.top_k * (2 if reranker else 1), 
                    strategy=config.retrieval.strategy
                )
                
                # 2. Rerank
                if reranker and chunks:
                    chunks = reranker.rerank(item.query, chunks)
                    chunks = chunks[:config.retrieval.top_k]
                
                # 3. Generate
                context = context_builder.build(item.query, chunks)
                prompt, _ = PromptManager.get_prompt("rag_default", query=item.query, context=context.formatted_context)
                
                answer = await llm_router.generate(prompt, model=config.generation.model_name)
                
                latency_ms = (time.time() - start_time) * 1000
                total_latency += latency_ms
                
                # Assume average token counts for estimation if not streaming full token data
                # In production, router should return token usage
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(answer.split()) * 1.3
                cost = CostTracker.calculate_cost(config.generation.model_name, int(prompt_tokens), int(completion_tokens))
                total_cost += cost
                
                # 4. Evaluate
                contexts_list = [c.text for c in chunks]
                for evaluator in evaluators:
                    score = await evaluator.compute(
                        query=item.query,
                        contexts=contexts_list,
                        answer=answer,
                        judge=judge
                    )
                    metric_sums[evaluator.__class__.__name__] += (score or 0.0)
                    
                processed_count += 1
                
            if processed_count > 0:
                metrics = ExperimentMetrics(
                    id=str(uuid.uuid4()),
                    run_id=self.run_id,
                    avg_latency=total_latency / processed_count,
                    total_cost=total_cost,
                    avg_precision=metric_sums["ContextPrecision"] / processed_count,
                    avg_recall=metric_sums["ContextRecall"] / processed_count,
                    avg_faithfulness=metric_sums["Faithfulness"] / processed_count,
                    avg_relevancy=metric_sums["AnswerRelevancy"] / processed_count
                )
                db.add(metrics)
                
            import datetime
            run.status = "completed"
            run.end_time = datetime.datetime.utcnow()
            db.commit()
            logger.info(f"Experiment run {self.run_id} completed successfully.")
            
        except Exception as e:
            logger.error(f"Experiment run {self.run_id} failed: {e}")
            run.status = "failed"
            db.commit()
        finally:
            db.close()
