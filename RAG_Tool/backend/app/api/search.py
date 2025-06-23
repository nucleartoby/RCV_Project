from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time
import logging

from app.models.document import SearchQuery, SearchResult
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

retrieval_service = RetrievalService()
generation_service = GenerationService()

@router.post("/", response_model=SearchResult)
async def search_documents(query: SearchQuery):
    start_time = time.time()
    
    try:
        results = await retrieval_service.hybrid_search(query)
        
        query_time = time.time() - start_time
        
        return SearchResult(
            chunks=results,
            total_results=len(results),
            query_time=query_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_response(query: SearchQuery):
    try:
        search_results = await retrieval_service.hybrid_search(query)

        response = await generation_service.generate_with_citations(
            query.query, search_results
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))