from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
import json
import logging
import uuid
import time
from ...services.rag.retriever import CoreRetrievalPipeline
from ...services.rag.context_builder import ContextBuilder
from ...services.rag.citation import SemanticCitationEngine
from ...services.llm.router import LLMRouter
from ..schemas.requests import QueryRequest
from ...core.prompts import PromptManager
from ...services.cache.semantic_cache import SemanticCache
from ...services.ingestion.embedder import Embedder
from ...services.evaluation.live_eval import LiveEvaluator
from ...core.database import SessionLocal
from ...core.models import Query, QueryResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Instantiate services once
pipeline = CoreRetrievalPipeline()
context_builder = ContextBuilder()
citation_engine = SemanticCitationEngine()
llm_router = LLMRouter()
semantic_cache = SemanticCache()
embedder = Embedder()
live_evaluator = LiveEvaluator(sample_rate=0.2)

@router.post("/stream")
async def query_rag_stream(request: QueryRequest = Body(...)):
    model = request.model or "phi3"
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    async def generator():
        db = SessionLocal()
        try:
            query_vec = embedder.embed_single(request.query)
            cached_answer = semantic_cache.get_cache(query_vec, model)
            
            if cached_answer:
                yield f"data: {json.dumps({'type': 'token', 'content': cached_answer})}\n\n"
                meta = {
                    "query_id": query_id,
                    "cached": True
                }
                yield f"data: {json.dumps({'type': 'metadata', 'content': meta})}\n\n"
                return
                
            # Run the new RetrievalPipeline
            chunks = await pipeline.run(request.query)
            
            context = context_builder.build(request.query, chunks)
            prompt, version = PromptManager.get_prompt("rag_default", query=request.query, context=context.formatted_context)
            
            full_answer = ""
            async for chunk in llm_router.generate_stream(prompt, model=model):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                
            semantic_cache.set_cache(request.query, query_vec, full_answer, model)
            
            # Ground citations post-hoc
            citations = citation_engine.ground(full_answer, chunks)
            citation_data = [
                {
                    "sentence": c.answer_sentence,
                    "source": c.chunk.source,
                    "score": c.similarity_score
                }
                for c in citations
            ]
            yield f"data: {json.dumps({'type': 'citations', 'content': citation_data})}\n\n"
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Save query and result to DB
            db_query = Query(id=query_id, query_text=request.query, latency_ms=latency_ms)
            db.add(db_query)
            db.commit()
            
            db_result = QueryResult(
                id=str(uuid.uuid4()),
                query_id=query_id,
                model=model,
                answer_text=full_answer,
                citations=citation_data
            )
            db.add(db_result)
            db.commit()
            
            # Trigger async sampled evaluation
            await live_evaluator.schedule_evaluation(
                query_id=query_id, 
                query_text=request.query, 
                answer_text=full_answer, 
                contexts=[c.text for c in chunks]
            )
            
            meta = {
                "query_id": query_id,
                "model": model,
                "cached": False,
                "latency_ms": latency_ms,
                "chunks_used": len(chunks)
            }
            yield f"data: {json.dumps({'type': 'metadata', 'content': meta})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            db.close()

    return StreamingResponse(generator(), media_type="text/event-stream")