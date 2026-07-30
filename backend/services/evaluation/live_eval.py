import asyncio
import random
import logging
from typing import List, Dict
from ...core.database import SessionLocal
from ...core.models import QueryResult
from .metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from .judge import LLMJudge

logger = logging.getLogger(__name__)

class LiveEvaluator:
    def __init__(self, sample_rate: float = 0.2):
        self.sample_rate = sample_rate
        self.judge = LLMJudge()
        self.metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy()
        ]

    async def schedule_evaluation(self, query_id: str, query_text: str, answer_text: str, contexts: List[str]):
        if random.random() <= self.sample_rate:
            logger.info(f"Scheduling live evaluation for query {query_id}")
            # Run asynchronously
            asyncio.create_task(self._run_evaluation(query_id, query_text, answer_text, contexts))

    async def _run_evaluation(self, query_id: str, query_text: str, answer_text: str, contexts: List[str]):
        try:
            scores = {}
            for metric in self.metrics:
                score = await metric.compute(
                    query=query_text,
                    contexts=contexts,
                    answer=answer_text,
                    judge=self.judge
                )
                scores[metric.__class__.__name__] = score

            # Store in DB
            db = SessionLocal()
            try:
                # Assuming query_result already created by the generation step
                query_result = db.query(QueryResult).filter_by(query_id=query_id).first()
                if query_result:
                    query_result.context_precision = scores.get("ContextPrecision")
                    query_result.context_recall = scores.get("ContextRecall")
                    query_result.faithfulness = scores.get("Faithfulness")
                    query_result.answer_relevancy = scores.get("AnswerRelevancy")
                    db.commit()
            finally:
                db.close()
                
            logger.info(f"Completed live evaluation for query {query_id}: {scores}")
        except Exception as e:
            logger.error(f"Live evaluation failed for query {query_id}: {e}")
