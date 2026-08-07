from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True)
    source_type = Column(String)
    source_uri = Column(String)
    title = Column(String)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String)
    
    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'))
    position = Column(Integer)
    heading_path = Column(String)
    token_count = Column(Integer)
    
    document = relationship("Document", back_populates="chunks")

class Query(Base):
    __tablename__ = 'queries'
    id = Column(String, primary_key=True)
    query_text = Column(String)
    rewritten_queries = Column(JSON) # Store list of strings
    strategy = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    latency_ms = Column(Integer)
    
    results = relationship("QueryResult", back_populates="query")

class QueryResult(Base):
    __tablename__ = 'query_results'
    id = Column(String, primary_key=True)
    query_id = Column(String, ForeignKey('queries.id'))
    model = Column(String)
    answer_text = Column(String)
    citations = Column(JSON) # Store citation mappings
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    
    query = relationship("Query", back_populates="results")

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(String, primary_key=True)
    query_id = Column(String, ForeignKey('queries.id'))
    rating = Column(Integer) # e.g. 1 (thumbs up) or -1 (thumbs down)
    created_at = Column(DateTime, default=datetime.utcnow)

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    items = relationship("DatasetItem", back_populates="dataset")

class DatasetItem(Base):
    __tablename__ = 'dataset_items'
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey('datasets.id'))
    query = Column(String)
    ground_truth = Column(String)
    document_id = Column(String, nullable=True) # Optional reference to source document
    
    dataset = relationship("Dataset", back_populates="items")

class Experiment(Base):
    __tablename__ = 'experiments'
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    runs = relationship("ExperimentRun", back_populates="experiment")

class ExperimentRun(Base):
    __tablename__ = 'experiment_runs'
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey('experiments.id'))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String) # 'running', 'completed', 'failed'
    
    # Provenance fields for optimization
    parent_experiment_id = Column(String, ForeignKey('experiments.id'), nullable=True)
    trial_number = Column(Integer, nullable=True)
    search_strategy = Column(String, nullable=True)
    seed = Column(String, nullable=True)
    config_delta = Column(JSON, nullable=True)
    
    experiment = relationship("Experiment", back_populates="runs", foreign_keys=[experiment_id])
    metrics = relationship("ExperimentMetrics", back_populates="run", uselist=False)

class ExperimentMetrics(Base):
    __tablename__ = 'experiment_metrics'
    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey('experiment_runs.id'))
    avg_latency = Column(Float, nullable=True)
    total_cost = Column(Float, nullable=True)
    avg_precision = Column(Float, nullable=True)
    avg_recall = Column(Float, nullable=True)
    avg_faithfulness = Column(Float, nullable=True)
    avg_relevancy = Column(Float, nullable=True)
    
    run = relationship("ExperimentRun", back_populates="metrics")
