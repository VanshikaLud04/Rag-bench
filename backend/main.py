from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.routes import ingest, query, evaluate, health, feedback, datasets, experiments

app = FastAPI(
    title=settings.app_name,
    version=settings.version
)

from .core.database import engine
from .core.models import Base

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluation"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])