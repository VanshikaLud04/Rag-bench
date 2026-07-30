import logging
from sqlalchemy import func
from ...core.database import SessionLocal
from ...core.models import Query, QueryResult, Feedback

logger = logging.getLogger(__name__)

class AnalyticsJobs:
    @staticmethod
    def aggregate_nightly_metrics():
        db = SessionLocal()
        try:
            # Average Latency
            avg_latency = db.query(func.avg(Query.latency_ms)).scalar()
            
            # Average Evaluation Scores
            avg_precision = db.query(func.avg(QueryResult.context_precision)).scalar()
            avg_recall = db.query(func.avg(QueryResult.context_recall)).scalar()
            avg_faithfulness = db.query(func.avg(QueryResult.faithfulness)).scalar()
            avg_relevancy = db.query(func.avg(QueryResult.answer_relevancy)).scalar()
            
            # Feedback counts
            positive_feedback = db.query(Feedback).filter(Feedback.rating == 1).count()
            negative_feedback = db.query(Feedback).filter(Feedback.rating == -1).count()
            
            logger.info("Nightly Analytics Aggregation Completed:")
            logger.info(f"Avg Latency: {avg_latency} ms")
            logger.info(f"Avg Precision: {avg_precision}, Avg Recall: {avg_recall}")
            logger.info(f"Avg Faithfulness: {avg_faithfulness}, Avg Relevancy: {avg_relevancy}")
            logger.info(f"Feedback: +{positive_feedback} / -{negative_feedback}")
            
            # In a real system, these would be saved to an analytics table or metrics store
        except Exception as e:
            logger.error(f"Analytics aggregation failed: {e}")
        finally:
            db.close()
