from fastapi import APIRouter
from ...core.config import settings

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "ok", "app": settings.app_name, "version": settings.version}