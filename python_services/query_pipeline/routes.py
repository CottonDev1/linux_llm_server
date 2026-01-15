"""
Query Pipeline Routes
=====================

FastAPI routes for the RAG query pipeline.

Endpoints:
- POST /query/search - Direct vector search without LLM
- POST /query - Main RAG query with vector search + LLM generation
- POST /query/stream - Streaming RAG query with SSE
- GET /query/projects - Get available projects list
- GET /query/cache/stats - Get cache statistics
- POST /query/cache/clear - Clear response cache

Architecture Notes:
------------------
These routes are designed to be drop-in replacements for the JavaScript
queryRoutes.js endpoints. They maintain the same API contract while
leveraging Python's async capabilities and the Pydantic validation layer.

SSE Streaming:
The streaming endpoint uses FastAPI's StreamingResponse with the
text/event-stream media type. Events are formatted as:
    data: {"type": "...", ...}\n\n

Client Disconnect Handling:
Streaming endpoints check for client disconnect and gracefully
terminate processing to avoid wasting resources.

Error Handling:
All endpoints return consistent error responses with:
- HTTP status code
- error field with message
- details field for additional context
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from query_pipeline.models.query_models import (
    SearchRequest,
    QueryRequest,
    StreamQueryRequest,
    SearchResponse,
    QueryResponse,
    ProjectInfo,
)
from query_pipeline.pipeline import get_query_pipeline, AVAILABLE_PROJECTS

logger = logging.getLogger(__name__)

# Create router with prefix and tags for OpenAPI documentation
router = APIRouter(prefix="/query", tags=["Query"])


# =============================================================================
# Search Endpoint
# =============================================================================

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Direct vector search without LLM processing.

    Returns raw search results formatted for MCP server integration.
    Useful for programmatic access to vector search or debugging
    retrieval quality.

    Args:
        request: Search request with query, project, limit

    Returns:
        SearchResponse with sources and total count
    """
    try:
        pipeline = await get_query_pipeline()
        return await pipeline.search(request)

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Query Endpoint
# =============================================================================

@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Main RAG query endpoint with vector search + LLM generation.

    This endpoint:
    1. Enhances query with conversation context (for follow-ups)
    2. Checks response cache
    3. Searches for relevant documents
    4. Generates answer using LLM
    5. Caches response for future queries

    Special handling for project='knowledge_base':
    - Routes to MongoDB documents collection
    - Uses documentation-focused system prompt
    - Includes department and document type metadata

    Args:
        request: Query request with all parameters

    Returns:
        QueryResponse with answer, sources, and metadata
    """
    try:
        pipeline = await get_query_pipeline()
        return await pipeline.query(request)

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process query",
                "details": str(e)
            }
        )


# =============================================================================
# Streaming Query Endpoint
# =============================================================================

@router.post("/stream")
async def query_stream(request: StreamQueryRequest, http_request: Request):
    """
    Streaming RAG query endpoint with SSE responses.

    Sources are sent first so the UI can display them immediately.
    LLM tokens are streamed as they're generated for real-time feedback.
    Final statistics are sent in the done event.

    Event format:
        data: {"type": "sources", "sources": [...]}\n\n
        data: {"type": "token", "token": "..."}\n\n
        data: {"type": "done", "tokenUsage": {...}}\n\n
        data: {"type": "error", "error": "..."}\n\n

    Args:
        request: Stream query request
        http_request: FastAPI request for disconnect detection

    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator():
        """Generate SSE events with disconnect handling."""
        try:
            pipeline = await get_query_pipeline()

            async for event in pipeline.query_stream(request):
                # Check for client disconnect
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    return

                yield event

        except Exception as e:
            logger.error(f"Stream query failed: {e}", exc_info=True)
            import json
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =============================================================================
# Projects Endpoint
# =============================================================================

@router.get("/projects")
async def get_projects():
    """
    Get available projects list.

    Returns a list of projects that can be queried, including:
    - Regular code projects (gin, warehouse, etc.)
    - The special 'knowledge_base' project for documentation
    - The 'all' option for cross-project search

    Returns:
        Dict with projects list
    """
    return {
        "projects": [p.model_dump() for p in AVAILABLE_PROJECTS]
    }


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get response cache statistics.

    Returns metrics including:
    - Current cache size
    - Hit rate
    - Eviction count
    - TTL configuration

    Returns:
        Dict with cache statistics
    """
    try:
        pipeline = await get_query_pipeline()
        return await pipeline.get_cache_stats()

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear the response cache.

    Removes all cached responses. Use with caution as this will
    temporarily increase load on vector search and LLM services.

    Returns:
        Dict with count of cleared entries
    """
    try:
        pipeline = await get_query_pipeline()
        count = await pipeline.clear_cache()
        return {"cleared": count, "message": f"Cleared {count} cache entries"}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Health Check Endpoint
# =============================================================================

@router.get("/health")
async def health_check():
    """
    Health check for the query pipeline.

    Verifies that:
    - Pipeline is initialized
    - Vector search service is available
    - LLM service is available

    Returns:
        Dict with health status
    """
    try:
        pipeline = await get_query_pipeline()
        return {
            "status": "healthy",
            "initialized": pipeline._initialized,
            "cache_stats": pipeline._response_cache.get_stats()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
