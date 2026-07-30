import uuid
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from ...core.database import SessionLocal
from ...core.models import Feedback
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int

@router.post("/")
async def submit_feedback(request: FeedbackRequest = Body(...)):
    if request.rating not in [-1, 1]:
        raise HTTPException(status_code=400, detail="Rating must be -1 or 1")
        
    db = SessionLocal()
    try:
        feedback_entry = Feedback(
            id=str(uuid.uuid4()),
            query_id=request.query_id,
            rating=request.rating
        )
        db.add(feedback_entry)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")
    finally:
        db.close()
