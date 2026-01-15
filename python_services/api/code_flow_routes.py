"""
Code Flow Analysis API Routes
=============================

FastAPI endpoints for code flow analysis using the Code Flow Pipeline.
Provides endpoints for:
- POST /api/code-flow - Complex multi-hop code flow queries
- POST /api/code-flow/stream - Streaming code flow analysis
- GET /api/method-lookup - Find methods by name, class, or signature
- POST /api/call-chain - Build execution paths from entry points

Design Rationale:
-----------------
These endpoints expose the Code Flow Pipeline functionality as REST APIs.
The design follows the same patterns used in the SQL pipeline routes:

1. **Request Validation**: Pydantic models for automatic validation and documentation
2. **Async Operations**: All operations are async for optimal performance
3. **SSE Streaming**: Long-running analyses can stream progress updates
4. **Error Handling**: Comprehensive error responses with status codes

Integration Notes:
-----------------
To integrate these routes into main.py, add:

    from api.code_flow_routes import router as code_flow_router
    app.include_router(code_flow_router)

This will register all routes under the /api prefix.
"""

import logging
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query as QueryParam
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from code_flow_pipeline import (
    CodeFlowPipeline,
    CodeFlowRequest,
    CodeFlowResponse,
    MethodLookupRequest,
    MethodLookupResponse,
    CallChainRequest,
    CallChainResponse,
    get_code_flow_pipeline,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["Code Flow Analysis"])

# Singleton pipeline instance
_pipeline_instance: Optional[CodeFlowPipeline] = None


async def get_pipeline() -> CodeFlowPipeline:
    """Get or create singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = await get_code_flow_pipeline()
    return _pipeline_instance


# ==============================================================================
# Request/Response Models for FastAPI
# ==============================================================================

class CodeFlowQueryRequest(BaseModel):
    """Request model for code flow analysis (API-specific)."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question about code flow",
        examples=[
            "How are bales committed to purchase subcontract in Gin?",
            "What's the execution path from UI to database for load creation?",
        ]
    )
    project: Optional[str] = Field(
        default=None,
        description="Project scope (e.g., 'gin', 'warehouse')",
    )
    maxHops: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum depth for call chain traversal"
    )
    includeCallGraph: bool = Field(
        default=True,
        description="Whether to build call chains from retrieved methods"
    )
    detailed: bool = Field(
        default=False,
        description="Include additional details in response"
    )


class MethodLookupQueryRequest(BaseModel):
    """Request model for method lookup (query params to model)."""
    methodName: str = Field(
        ...,
        min_length=1,
        description="Method name to search for"
    )
    className: Optional[str] = Field(
        default=None,
        description="Class name filter"
    )
    project: Optional[str] = Field(
        default=None,
        description="Project scope"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results"
    )


class CallChainBuildRequest(BaseModel):
    """Request model for call chain building."""
    entryPoint: str = Field(
        ...,
        min_length=1,
        description="Entry point method (e.g., 'BaleCommitmentWindow.btnCommit_Click')",
    )
    project: Optional[str] = Field(
        default=None,
        description="Project scope"
    )
    maxDepth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum call chain depth"
    )
    targetMethod: Optional[str] = Field(
        default=None,
        description="Optional target method to find paths to"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline_initialized: bool
    cache_size: int = 0
    version: str = "0.1.0"


# ==============================================================================
# Endpoints
# ==============================================================================

@router.get("/code-flow/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health of the code flow pipeline.

    Returns the initialization status and cache information.
    """
    try:
        pipeline = await get_pipeline()
        return HealthResponse(
            status="healthy",
            pipeline_initialized=True,
            cache_size=len(pipeline._cache) if hasattr(pipeline, "_cache") else 0,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            pipeline_initialized=False,
        )


@router.post("/code-flow", response_model=CodeFlowResponse)
async def analyze_code_flow(request: CodeFlowQueryRequest) -> CodeFlowResponse:
    """
    Analyze code flow for a natural language question.

    This endpoint performs multi-hop retrieval to understand:
    - Business process documentation
    - Method implementations
    - UI event handlers
    - Call chains from entry points to database operations

    The response includes a synthesized answer along with supporting
    sources and call chain visualizations.

    Example questions:
    - "How are bales committed to purchase subcontract in Gin?"
    - "What's the execution path from UI to database for load creation?"
    - "Show me the complete flow for processing inbound files"
    """
    try:
        pipeline = await get_pipeline()

        # Convert API request to pipeline request
        pipeline_request = CodeFlowRequest(
            query=request.question,
            project=request.project,
            max_hops=request.maxHops,
            include_call_graph=request.includeCallGraph,
            detailed=request.detailed,
        )

        # Execute analysis
        response = await pipeline.analyze(pipeline_request)

        return response

    except Exception as e:
        logger.error(f"Code flow analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Code flow analysis failed",
                "message": str(e),
            }
        )


@router.post("/code-flow/stream")
async def analyze_code_flow_stream(request: CodeFlowQueryRequest):
    """
    Analyze code flow with streaming progress updates.

    Returns Server-Sent Events (SSE) stream with progress updates
    for each analysis stage:
    - classification: Query type detection
    - retrieval: Multi-stage vector search
    - chains: Call chain construction
    - synthesis: LLM answer generation
    - result: Final response

    This endpoint is ideal for long-running analyses where the client
    wants to show real-time progress feedback.
    """
    try:
        pipeline = await get_pipeline()

        # Convert API request to pipeline request
        pipeline_request = CodeFlowRequest(
            query=request.question,
            project=request.project,
            max_hops=request.maxHops,
            include_call_graph=request.includeCallGraph,
            detailed=request.detailed,
        )

        async def event_generator():
            """Generate SSE events from pipeline stream."""
            try:
                async for event in pipeline.analyze_stream(pipeline_request):
                    yield event.to_sse()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                error_event = {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
                }
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"Stream setup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Stream setup failed",
                "message": str(e),
            }
        )


@router.get("/method-lookup", response_model=MethodLookupResponse)
async def lookup_methods(
    methodName: str = QueryParam(
        ...,
        min_length=1,
        description="Method name to search for"
    ),
    className: Optional[str] = QueryParam(
        default=None,
        description="Class name filter"
    ),
    project: Optional[str] = QueryParam(
        default=None,
        description="Project scope"
    ),
    limit: int = QueryParam(
        default=10,
        ge=1,
        le=100,
        description="Maximum results"
    ),
) -> MethodLookupResponse:
    """
    Find methods by name, class, or signature.

    This endpoint searches the code methods collection for matching
    methods. Results include:
    - Method signature and return type
    - Purpose summary
    - File location
    - Called methods and callers
    - Database tables accessed

    Use this endpoint to explore the codebase or find specific
    method implementations.
    """
    try:
        pipeline = await get_pipeline()

        request = MethodLookupRequest(
            method_name=methodName,
            class_name=className,
            project=project,
            limit=limit,
        )

        response = await pipeline.lookup_method(request)

        return response

    except Exception as e:
        logger.error(f"Method lookup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Method lookup failed",
                "message": str(e),
            }
        )


@router.post("/call-chain", response_model=CallChainResponse)
async def build_call_chain(request: CallChainBuildRequest) -> CallChainResponse:
    """
    Build execution paths from an entry point.

    This endpoint traces the call graph starting from a method
    (typically a UI event handler) and builds:
    - Call tree: Hierarchical view of method calls
    - Call chains: Linear execution paths

    Use this to understand:
    - What code runs when a user clicks a button
    - How data flows from UI to database
    - Which methods are involved in a feature

    Example entry points:
    - "btnCommit_Click"
    - "BaleCommitmentWindow.btnCommit_Click"
    - "ProcessInboundFile"
    """
    try:
        pipeline = await get_pipeline()

        pipeline_request = CallChainRequest(
            entry_point=request.entryPoint,
            project=request.project,
            max_depth=request.maxDepth,
            target_method=request.targetMethod,
        )

        response = await pipeline.build_call_chain(pipeline_request)

        return response

    except Exception as e:
        logger.error(f"Call chain build failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Call chain build failed",
                "message": str(e),
            }
        )


@router.delete("/code-flow/cache")
async def clear_cache():
    """
    Clear the code flow pipeline cache.

    Clears both response cache and method metadata cache.
    Use this after updating code context in MongoDB.
    """
    try:
        pipeline = await get_pipeline()
        pipeline.clear_cache()

        return {
            "success": True,
            "message": "Cache cleared successfully",
        }

    except Exception as e:
        logger.error(f"Cache clear failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Cache clear failed",
                "message": str(e),
            }
        )


# ==============================================================================
# Legacy v1 Endpoints (for backward compatibility)
# ==============================================================================

@router.post("/code-flow-v1", response_model=CodeFlowResponse)
async def analyze_code_flow_v1(request: CodeFlowQueryRequest) -> CodeFlowResponse:
    """
    Legacy v1 code flow analysis endpoint.

    This endpoint is provided for backward compatibility with existing
    clients. It uses the same pipeline as the main endpoint.

    Deprecated: Use POST /api/code-flow instead.
    """
    return await analyze_code_flow(request)


@router.get("/method-lookup-v1", response_model=MethodLookupResponse)
async def lookup_methods_v1(
    method: Optional[str] = QueryParam(default=None, description="Method name"),
    className: Optional[str] = QueryParam(default=None, alias="class", description="Class name"),
    signature: Optional[str] = QueryParam(default=None, description="Method signature"),
    project: Optional[str] = QueryParam(default=None, description="Project scope"),
    limit: int = QueryParam(default=20, ge=1, le=100, description="Max results"),
) -> MethodLookupResponse:
    """
    Legacy v1 method lookup endpoint.

    Deprecated: Use GET /api/method-lookup instead.
    """
    # Map legacy parameter names
    method_name = method or ""

    if not method_name and not className and not signature:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "At least one search parameter required",
                "message": "Provide method, class, or signature",
            }
        )

    return await lookup_methods(
        methodName=method_name or signature or "",
        className=className,
        project=project,
        limit=limit,
    )


@router.get("/call-chain-v1", response_model=CallChainResponse)
async def build_call_chain_v1(
    startMethod: Optional[str] = QueryParam(default=None, description="Starting method"),
    eventHandler: Optional[str] = QueryParam(default=None, description="UI event handler"),
    endMethod: Optional[str] = QueryParam(default=None, description="Target method"),
    project: Optional[str] = QueryParam(default=None, description="Project scope"),
    maxDepth: int = QueryParam(default=10, ge=1, le=50, description="Max depth"),
) -> CallChainResponse:
    """
    Legacy v1 call chain endpoint (GET).

    Deprecated: Use POST /api/call-chain instead.
    """
    if not project:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Project scope required",
                "message": "Provide project parameter",
            }
        )

    if not startMethod and not eventHandler:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Starting point required",
                "message": "Provide startMethod or eventHandler",
            }
        )

    entry_point = startMethod or eventHandler

    request = CallChainBuildRequest(
        entryPoint=entry_point,
        project=project,
        maxDepth=maxDepth,
        targetMethod=endMethod,
    )

    return await build_call_chain(request)
