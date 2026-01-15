"""
Code Flow API Routes
====================

FastAPI routes for the code flow analysis pipeline.

Endpoints:
- POST /code-flow - Complex multi-hop code flow queries
- GET /method-lookup - Find methods by name, class, or signature
- POST /call-chain - Build execution paths from entry points
- POST /code-flow/stream - Streaming code flow analysis with SSE

Migrated from: src/routes/codeFlowRoutes.js
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from code_flow_pipeline.models.query_models import (
    CodeFlowRequest,
    CodeFlowResponse,
    MethodLookupRequest,
    MethodLookupResponse,
    CallChainRequest,
    CallChainResponse,
    SSEEvent,
)
from code_flow_pipeline.pipeline import (
    CodeFlowPipeline,
    get_code_flow_pipeline,
)

logger = logging.getLogger(__name__)


def create_code_flow_routes() -> APIRouter:
    """
    Create FastAPI router with code flow analysis endpoints.

    Returns:
        APIRouter with all code flow routes configured
    """
    router = APIRouter()

    @router.post(
        "/code-flow",
        response_model=CodeFlowResponse,
        summary="Code Flow Analysis",
        description="Complex multi-hop code flow queries using multi-stage retrieval."
    )
    async def analyze_code_flow(request: CodeFlowRequest) -> CodeFlowResponse:
        """Analyze code flow for a natural language question."""
        try:
            pipeline = await get_code_flow_pipeline()
            response = await pipeline.analyze(request)
            return response
        except ValueError as e:
            logger.warning(f"Invalid code flow request: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Code flow analysis failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/code-flow/stream",
        summary="Streaming Code Flow Analysis",
        response_class=StreamingResponse
    )
    async def analyze_code_flow_stream(request: CodeFlowRequest, http_request: Request):
        """Analyze code flow with streaming response."""
        async def generate_events():
            try:
                pipeline = await get_code_flow_pipeline()
                async for event in pipeline.analyze_stream(request):
                    if await http_request.is_disconnected():
                        return
                    yield event.to_sse()
            except Exception as e:
                logger.error(f"Streaming analysis failed: {e}", exc_info=True)
                error_event = SSEEvent(event="error", data={"error": str(e)})
                yield error_event.to_sse()

        return StreamingResponse(
            generate_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    @router.get(
        "/method-lookup",
        response_model=MethodLookupResponse,
        summary="Method Lookup"
    )
    async def lookup_method(
        methodName: str = Query(..., description="Method name to search for"),
        className: Optional[str] = Query(None, description="Class name filter"),
        project: Optional[str] = Query(None, description="Project scope"),
        limit: int = Query(10, ge=1, le=100, description="Maximum results")
    ) -> MethodLookupResponse:
        """Look up methods by name, class, or signature."""
        try:
            pipeline = await get_code_flow_pipeline()
            request = MethodLookupRequest(
                method_name=methodName,
                class_name=className,
                project=project,
                limit=limit
            )
            return await pipeline.lookup_method(request)
        except Exception as e:
            logger.error(f"Method lookup failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/call-chain",
        response_model=CallChainResponse,
        summary="Build Call Chain"
    )
    async def build_call_chain(request: CallChainRequest) -> CallChainResponse:
        """Build call chains from an entry point."""
        try:
            pipeline = await get_code_flow_pipeline()
            return await pipeline.build_call_chain(request)
        except Exception as e:
            logger.error(f"Call chain building failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/code-flow/health", summary="Health Check")
    async def health_check():
        """Check code flow pipeline health."""
        try:
            pipeline = await get_code_flow_pipeline()
            return {"healthy": True, "pipeline": "code_flow"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    return router


def register_code_flow_routes(app, prefix: str = "/api"):
    """Register code flow routes with a FastAPI app."""
    router = create_code_flow_routes()
    app.include_router(router, prefix=prefix, tags=["Code Flow"])
    logger.info(f"Code flow routes registered at {prefix}")
