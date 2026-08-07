from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ...core.database import SessionLocal
from ...core.models import Dataset, DatasetItem
import uuid
import json
import csv
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), name: str = Form(...), description: str = Form("")):
    if not file.filename.endswith(('.json', '.csv')):
        raise HTTPException(status_code=400, detail="Only JSON and CSV files are supported.")
        
    content = await file.read()
    items_data = []
    
    try:
        if file.filename.endswith('.json'):
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError("JSON must be a list of objects")
            items_data = data
        elif file.filename.endswith('.csv'):
            decoded = content.decode('utf-8')
            reader = csv.DictReader(decoded.splitlines())
            items_data = list(reader)
    except Exception as e:
        logger.error(f"Failed to parse dataset file: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid file format: {e}")
        
    db = SessionLocal()
    try:
        dataset_id = str(uuid.uuid4())
        dataset = Dataset(id=dataset_id, name=name, description=description)
        db.add(dataset)
        
        for item in items_data:
            if 'query' not in item or 'ground_truth' not in item:
                continue # Skip invalid rows
                
            db_item = DatasetItem(
                id=str(uuid.uuid4()),
                dataset_id=dataset_id,
                query=item['query'],
                ground_truth=item['ground_truth'],
                document_id=item.get('document_id')
            )
            db.add(db_item)
            
        db.commit()
        return {"status": "success", "dataset_id": dataset_id, "items_processed": len(items_data)}
    except Exception as e:
        logger.error(f"Database error during dataset upload: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save dataset")
    finally:
        db.close()

@router.get("/")
async def list_datasets():
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).all()
        return [
            {
                "id": d.id, 
                "name": d.name, 
                "description": d.description, 
                "created_at": d.created_at,
                "item_count": db.query(DatasetItem).filter_by(dataset_id=d.id).count()
            } 
            for d in datasets
        ]
    finally:
        db.close()
