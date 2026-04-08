import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from ...services.ingestion.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingestion_service.ingest(str(dest))
    return {"message": "Ingested successfully", **result}