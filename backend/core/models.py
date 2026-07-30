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
