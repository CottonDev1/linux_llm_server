"""
Document Pipeline API Routes
=============================

FastAPI endpoints for the document retrieval pipeline.
Provides RAG (Retrieval-Augmented Generation) query capabilities with:
- Direct vector search without LLM
- RAG query with LLM generation and CRAG validation
- Streaming RAG query with SSE
- User feedback collection

These routes use the KnowledgeBaseOrchestrator for full CRAG pattern:
- Query Understanding
- Hybrid Retrieval
- Document Grading
- Answer Generation
- Answer Validation
- Self-Correction
- Learning Feedback
"""

import logging
import time
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query as QueryParam
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mongodb import MongoDBService
from orchestrator.document_orchestrator import get_orchestrator, KnowledgeBaseOrchestrator
from orchestrator.models import (
    QueryRequest,
    QueryResponse,
    FeedbackRecord,
    FeedbackType,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/documents", tags=["Document Pipeline"])


# ============================================================================
# Request/Response Models (API Layer)
# ============================================================================

class DocumentQueryRequest(BaseModel):
    """Request model for document RAG query."""
    query: str = Field(..., min_length=1, description="Natural language query")
    project: Optional[str] = Field(default=None, description="Project filter (e.g., 'knowledge_base', 'gin', 'EWRLibrary')")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of documents to retrieve")
    department: Optional[str] = Field(default=None, description="Filter by department")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type")
    subject: Optional[str] = Field(default=None, description="Filter by subject")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="Maximum response tokens")
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history for context")
    skip_validation: bool = Field(default=False, description="Skip answer validation step")


class DocumentSource(BaseModel):
    """Source document model."""
    id: Optional[str] = Field(default=None, description="Document chunk ID")
    parent_id: Optional[str] = Field(default=None, description="Parent document ID")
    project: Optional[str] = Field(default=None, description="Project name")
    department: Optional[str] = Field(default=None, description="Department")
    type: Optional[str] = Field(default=None, description="Document type")
    file: Optional[str] = Field(default=None, description="File name")
    title: Optional[str] = Field(default=None, description="Document title")
    snippet: str = Field(..., description="Content snippet")
    relevance: float = Field(..., description="Relevance score (0-1)")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index")
    total_chunks: Optional[int] = Field(default=None, description="Total chunks in document")


class DocumentQueryResponse(BaseModel):
    """Response model for document RAG query."""
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[DocumentSource] = Field(default_factory=list, description="Source documents")
    query: str = Field(..., description="Original query")
    model: str = Field(..., description="LLM model used")
    search_strategy: str = Field(default="crag-hybrid", description="Search strategy used")
    cached: bool = Field(default=False, description="Whether response was cached")
    timing: Optional[Dict[str, float]] = Field(default=None, description="Timing breakdown")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
    confidence: float = Field(default=0.0, description="Answer confidence score")
    validation_passed: bool = Field(default=True, description="Whether validation checks passed")
    query_intent: Optional[str] = Field(default=None, description="Detected query intent")


class SearchRequest(BaseModel):
    """Request model for direct vector search."""
    query: str = Field(..., min_length=1, description="Search query")
    project: Optional[str] = Field(default=None, description="Project filter")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    department: Optional[str] = Field(default=None, description="Filter by department")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type")
    subject: Optional[str] = Field(default=None, description="Filter by subject")


class SearchResponse(BaseModel):
    """Response model for direct vector search."""
    sources: List[DocumentSource] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
    search_time_ms: int = Field(..., description="Search time in milliseconds")


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    query_id: Optional[str] = Field(default=None, description="Optional query ID")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    feedback: str = Field(..., description="Feedback type: 'positive' or 'negative'")
    comment: Optional[str] = Field(default=None, description="Optional comment")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sources used")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool = Field(..., description="Whether feedback was stored")
    message: str = Field(..., description="Result message")
    feedback_id: Optional[str] = Field(default=None, description="Feedback record ID")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status: healthy or unhealthy")
    service: str = Field(default="document_pipeline", description="Service name")
    mongodb: bool = Field(..., description="MongoDB connection status")
    llm: bool = Field(..., description="LLM service status")
    orchestrator: bool = Field(default=False, description="Orchestrator status")
    timestamp: str = Field(..., description="Timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


# ============================================================================
# Helper Functions
# ============================================================================

def format_sources_from_orchestrator(sources: List[Dict[str, Any]], project: Optional[str] = None) -> List[DocumentSource]:
    """Format orchestrator sources into DocumentSource models."""
    formatted = []
    for source in sources:
        formatted.append(DocumentSource(
            id=source.get("document_id"),
            parent_id=source.get("document_id"),
            project=project or source.get("project") or "knowledge_base",
            department=source.get("department"),
            type=source.get("type"),
            file=source.get("source_file") or source.get("title"),
            title=source.get("title"),
            snippet=source.get("content_preview", "")[:200] + "...",
            relevance=source.get("score", 0),
            chunk_index=source.get("chunk_index"),
            total_chunks=source.get("total_chunks")
        ))
    return formatted


def format_sources_from_mongodb(results: List[Dict], project: Optional[str] = None) -> List[DocumentSource]:
    """Format MongoDB search results into DocumentSource models."""
    sources = []
    for result in results:
        sources.append(DocumentSource(
            id=result.get("id"),
            parent_id=result.get("parent_id"),
            project=project or result.get("project") or "knowledge_base",
            department=result.get("department"),
            type=result.get("type"),
            file=result.get("file_name") or result.get("title"),
            title=result.get("title"),
            snippet=(result.get("content") or "")[:200] + "...",
            relevance=result.get("relevance_score", 0),
            chunk_index=result.get("chunk_index"),
            total_chunks=result.get("total_chunks")
        ))
    return sources


def build_filters(request: DocumentQueryRequest) -> Dict[str, Any]:
    """Build filters dictionary from request parameters."""
    filters = {}
    if request.project:
        filters["project"] = request.project
    if request.department:
        filters["department"] = request.department
    if request.doc_type:
        filters["type"] = request.doc_type
    if request.subject:
        filters["subject"] = request.subject
    return filters


# ============================================================================
# Search Endpoints
# ============================================================================

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Direct vector search without LLM processing.

    Returns raw search results formatted with relevance scores.
    Use this for fast document retrieval without answer generation.

    **Request Body:**
    - `query`: Search query text
    - `project`: Optional project filter
    - `limit`: Maximum results to return
    - `department`: Optional department filter
    - `doc_type`: Optional document type filter
    - `subject`: Optional subject filter

    **Returns:**
    - Search results with relevance scores
    - Total result count
    - Search timing
    """
    start_time = time.time()

    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        logger.info(f"Direct search: '{request.query}' (project={request.project}, limit={request.limit})")

        # Perform vector search
        results = await mongo_service.search_documents(
            query=request.query,
            limit=request.limit,
            department=request.department,
            doc_type=request.doc_type,
            subject=request.subject
        )

        # Format sources
        sources = format_sources_from_mongodb(results, request.project)

        search_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Found {len(sources)} documents in {search_time_ms}ms")

        return SearchResponse(
            sources=sources,
            total_results=len(sources),
            query=request.query,
            search_time_ms=search_time_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ============================================================================
# Query Endpoints (Using Orchestrator)
# ============================================================================

@router.post("/query", response_model=DocumentQueryResponse)
async def query_documents(request: DocumentQueryRequest):
    """
    Main RAG query endpoint with LLM generation using CRAG pattern.

    Uses KnowledgeBaseOrchestrator for full pipeline:
    1. Query Understanding - Classify intent, expand query
    2. Hybrid Retrieval - Vector + BM25 search (when enabled)
    3. Document Grading - CRAG relevance filtering
    4. Answer Generation - LLM synthesis with context
    5. Answer Validation - Check relevancy, faithfulness, completeness
    6. Self-Correction - Retry if validation fails

    **Request Body:**
    - `query`: Natural language question
    - `project`: Optional project filter
    - `limit`: Maximum documents to retrieve (default: 5)
    - `department`: Optional department filter
    - `doc_type`: Optional document type filter
    - `temperature`: LLM temperature (default: 0.1)
    - `max_tokens`: Maximum response tokens (default: 500)
    - `history`: Conversation history for context
    - `skip_validation`: Skip answer validation step

    **Returns:**
    - Generated answer with CRAG validation
    - Source documents with relevance scores
    - Token usage statistics
    - Timing breakdown per stage
    - Confidence score and validation status
    """
    start_time = time.time()

    try:
        # Get orchestrator instance
        orchestrator = await get_orchestrator()

        logger.info(f"Query via orchestrator: '{request.query}' (project={request.project}, limit={request.limit})")

        # Build filters from request
        filters = build_filters(request)

        # Build previous queries from history for follow-up detection
        previous_queries = [
            msg.get("content", "")
            for msg in request.history
            if msg.get("role") == "user"
        ]

        # Create orchestrator request
        orchestrator_request = QueryRequest(
            query=request.query,
            filters=filters,
            previous_queries=previous_queries,
            conversation_history=request.history,  # Pass full conversation history
            max_documents=request.limit,
            skip_validation=request.skip_validation,
        )

        # Process through orchestrator pipeline
        result: QueryResponse = await orchestrator.process_query(orchestrator_request)

        if result.error:
            logger.error(f"Orchestrator error: {result.error}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {result.error}")

        # Format sources for API response
        sources = format_sources_from_orchestrator(result.sources, request.project)

        logger.info(
            f"Answer generated via CRAG pipeline in {result.total_time_ms}ms "
            f"(confidence: {result.confidence:.2f}, validation: {result.validation_passed})"
        )

        return DocumentQueryResponse(
            answer=result.answer,
            sources=sources,
            query=request.query,
            model=result.token_usage.get("model", "general"),
            search_strategy="crag-hybrid",
            cached=False,  # TODO: Check semantic cache
            timing=result.stage_timings,
            token_usage={
                "promptTokens": result.token_usage.get("prompt_tokens", 0),
                "responseTokens": result.token_usage.get("completion_tokens", 0),
                "totalTokens": result.token_usage.get("total_tokens", 0)
            },
            confidence=result.confidence,
            validation_passed=result.validation_passed,
            query_intent=result.query_intent.value if result.query_intent else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query-stream")
async def query_documents_stream(request: DocumentQueryRequest):
    """
    Streaming RAG query endpoint with Server-Sent Events (SSE).

    Streams pipeline stages, document retrieval, and LLM tokens in real-time.
    Uses KnowledgeBaseOrchestrator for full CRAG pattern with streaming.

    **Event Types:**
    - `stage_start`: Pipeline stage starting
    - `stage_complete`: Pipeline stage completed
    - `document_found`: Retrieved document
    - `generation_token`: Individual LLM token
    - `validation_check`: Validation check result
    - `complete`: Processing complete with final response
    - `error`: Error information

    **Request Body:**
    - Same as /query endpoint

    **Returns:**
    - StreamingResponse with text/event-stream content type
    """
    try:
        # Get orchestrator instance
        orchestrator = await get_orchestrator()

        logger.info(f"Streaming query via orchestrator: '{request.query}'")

        async def event_generator():
            """Generate SSE events from orchestrator stream."""
            try:
                # Build filters from request
                filters = build_filters(request)

                # Build previous queries from history
                previous_queries = [
                    msg.get("content", "")
                    for msg in request.history
                    if msg.get("role") == "user"
                ]

                # Create orchestrator request with streaming enabled
                orchestrator_request = QueryRequest(
                    query=request.query,
                    filters=filters,
                    previous_queries=previous_queries,
                    max_documents=request.limit,
                    skip_validation=request.skip_validation,
                    stream=True,
                )

                # Stream events from orchestrator
                async for event in orchestrator.process_query_stream(orchestrator_request):
                    # Convert to SSE format
                    yield event.to_sse()

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                error_event = {
                    "event_type": "error",
                    "data": {"error": str(e)},
                    "message": str(e)
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Failed to initialize streaming: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming initialization failed: {str(e)}")


# ============================================================================
# Feedback Endpoint
# ============================================================================

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a query response.

    Stores feedback via orchestrator's learning system.

    **Request Body:**
    - `query_id`: Optional query ID for tracking
    - `query`: Original user query
    - `answer`: Generated answer
    - `feedback`: 'positive' or 'negative'
    - `comment`: Optional user comment
    - `sources`: Optional sources used

    **Returns:**
    - Success status and feedback ID
    """
    try:
        orchestrator = await get_orchestrator()

        # Map feedback string to FeedbackType
        feedback_type = (
            FeedbackType.THUMBS_UP if request.feedback == "positive"
            else FeedbackType.THUMBS_DOWN
        )

        # Create feedback record
        feedback_record = FeedbackRecord(
            query_id=request.query_id or "",
            query=request.query,
            answer=request.answer,
            feedback_type=feedback_type,
            comment=request.comment,
            sources_used=[s.get("id", "") for s in (request.sources or [])],
        )

        # Record via orchestrator
        success = await orchestrator.record_feedback(feedback_record)

        if success:
            logger.info(f"Feedback submitted: {request.feedback} for query '{request.query[:50]}...'")
            return FeedbackResponse(
                success=True,
                message="Feedback submitted successfully",
                feedback_id=feedback_record.feedback_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to store feedback")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


# ============================================================================
# Project List Endpoint
# ============================================================================

@router.get("/projects")
async def list_projects():
    """
    Get list of available projects for document filtering.

    Returns project metadata from MongoDB or static fallback.
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        # Try to get unique projects from documents collection
        try:
            collection = mongo_service.db["documents"]
            projects = await collection.distinct("project")

            if projects:
                return {
                    "projects": [
                        {"id": "all", "name": "All Projects", "description": "Search across all projects"},
                        *[{"id": p, "name": p, "description": f"Project: {p}"} for p in sorted(projects) if p]
                    ]
                }
        except Exception as e:
            logger.warning(f"Failed to get projects from MongoDB: {e}")

        # Fallback to static list
        return {
            "projects": [
                {"id": "all", "name": "All Projects", "description": "Search across all projects"},
                {"id": "gin", "name": "Gin", "description": "Cotton Gin application"},
                {"id": "EWRLibrary", "name": "EWR Library", "description": "EWR shared library"},
                {"id": "warehouse", "name": "Warehouse", "description": "Warehouse management"},
                {"id": "marketing", "name": "Marketing", "description": "Marketing application"},
                {"id": "knowledge_base", "name": "Knowledge Base", "description": "EWR Documentation"}
            ]
        }

    except Exception as e:
        logger.error(f"Failed to list projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


# ============================================================================
# Cache Management
# ============================================================================

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get semantic cache statistics.

    **Returns:**
    - Cache availability status
    - Entry counts by type (embeddings, results, responses)
    - Hit counts
    - TTL settings
    """
    try:
        orchestrator = await get_orchestrator()

        if orchestrator._semantic_cache and orchestrator._semantic_cache.is_available:
            stats = await orchestrator._semantic_cache.get_stats()
            return {"success": True, **stats}
        else:
            return {
                "success": True,
                "available": False,
                "message": "Semantic cache not available (Redis not running)"
            }

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/cache/clear")
async def clear_cache(pattern: Optional[str] = None):
    """
    Clear semantic cache entries.

    **Query Parameters:**
    - `pattern`: Optional pattern to match (e.g., "emb:*" for embeddings only)

    **Returns:**
    - Number of entries cleared
    """
    try:
        orchestrator = await get_orchestrator()

        if orchestrator._semantic_cache and orchestrator._semantic_cache.is_available:
            deleted = await orchestrator._semantic_cache.clear(pattern)
            return {
                "success": True,
                "entries_cleared": deleted,
                "pattern": pattern or "all"
            }
        else:
            return {
                "success": False,
                "message": "Semantic cache not available"
            }

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for document pipeline.

    **Returns:**
    - Service status and component availability
    """
    try:
        mongo_service = MongoDBService.get_instance()
        mongo_initialized = mongo_service.is_initialized

        # Check orchestrator
        orchestrator_healthy = False
        llm_healthy = False
        cache_available = False
        try:
            orchestrator = await get_orchestrator()
            orchestrator_healthy = orchestrator._initialized

            # Check LLM via orchestrator
            if orchestrator._llm_service:
                health = await orchestrator._llm_service.health_check()
                llm_healthy = health.get("healthy", False)

            # Check cache
            if orchestrator._semantic_cache:
                cache_available = orchestrator._semantic_cache.is_available
        except Exception as e:
            logger.warning(f"Orchestrator health check failed: {e}")

        status = "healthy" if (mongo_initialized and orchestrator_healthy and llm_healthy) else "degraded"

        return HealthResponse(
            status=status,
            service="document_pipeline",
            mongodb=mongo_initialized,
            llm=llm_healthy,
            orchestrator=orchestrator_healthy,
            timestamp=datetime.utcnow().isoformat(),
            details={
                "mongodb_initialized": mongo_initialized,
                "orchestrator_initialized": orchestrator_healthy,
                "llm_available": llm_healthy,
                "cache_available": cache_available,
                "pipeline": "crag-hybrid"
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            service="document_pipeline",
            mongodb=False,
            llm=False,
            orchestrator=False,
            timestamp=datetime.utcnow().isoformat(),
            details={"error": str(e)}
        )
