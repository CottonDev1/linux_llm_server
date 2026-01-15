"""
Code Assistance API Routes
==========================

FastAPI routes for the code assistance pipeline.

Endpoints:
- POST /query - Non-streaming code assistance query
- POST /query/stream - Streaming code assistance with SSE
- POST /feedback - Submit feedback on responses
- GET /stats - Get code entity statistics

Design Rationale:
-----------------
These routes follow RESTful conventions and mirror the JavaScript
implementation in codeRoutes.js. Key design decisions:

1. Separate endpoints for streaming/non-streaming to enable
   optimal client integration patterns

2. SSE streaming for real-time UI updates with proper
   content-type and connection headers

3. Comprehensive error handling with appropriate HTTP status codes

4. Request validation via Pydantic models

Usage:
    from fastapi import FastAPI
    from code_assistance_pipeline.routes import create_code_routes

    app = FastAPI()
    router = create_code_routes()
    app.include_router(router, prefix="/api/code", tags=["Code Assistance"])
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from code_assistance_pipeline.models.query_models import (
    CodeQueryRequest,
    CodeQueryResponse,
    CodeFeedbackRequest,
    CodeFeedbackResponse,
    CodeStatsResponse,
    SSEEvent,
)
from code_assistance_pipeline.pipeline import (
    CodeAssistancePipeline,
    get_code_assistance_pipeline,
)
from code_assistance_pipeline.services.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


def create_code_routes() -> APIRouter:
    """
    Create FastAPI router with code assistance endpoints.

    Returns:
        APIRouter with all code assistance routes configured

    Usage:
        app = FastAPI()
        router = create_code_routes()
        app.include_router(router, prefix="/api/code", tags=["Code Assistance"])
    """
    router = APIRouter()

    @router.post(
        "/query",
        response_model=CodeQueryResponse,
        summary="Code Assistance Query",
        description="""
        Process a natural language question about the C# codebase.

        This endpoint performs RAG (Retrieval-Augmented Generation):
        1. Searches for relevant methods, classes, and event handlers
        2. Retrieves call chain relationships
        3. Generates an LLM response with code context

        The response includes:
        - Generated answer citing specific code elements
        - List of source code entities used
        - Call chain showing code flow
        - Performance metrics (retrieval time, generation time)

        Example questions:
        - "How does the Save button work in the Bale Entry screen?"
        - "What methods call BaleService.GetBales?"
        - "Show me the class that handles shipping orders"
        """
    )
    async def query_code(request: CodeQueryRequest) -> CodeQueryResponse:
        """
        Process a code assistance query (non-streaming).

        Args:
            request: Code query request with question and options

        Returns:
            CodeQueryResponse with answer, sources, and metrics

        Raises:
            HTTPException: If query processing fails
        """
        try:
            pipeline = await get_code_assistance_pipeline()
            response = await pipeline.process_query(request)
            return response

        except ValueError as e:
            logger.warning(f"Invalid query request: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            logger.error(f"Code query failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/query/stream",
        summary="Streaming Code Assistance Query",
        description="""
        Process a code assistance query with Server-Sent Events streaming.

        Returns a stream of SSE events for real-time UI updates:

        1. `status` - Pipeline stage updates (retrieving, generating)
        2. `sources` - Retrieved code sources (methods, classes)
        3. `streaming` - LLM tokens as they are generated
        4. `complete` - Final response with full answer and metrics
        5. `error` - Error information if something fails

        Example event format:
        ```
        event: status
        data: {"status": "retrieving", "message": "Searching codebase..."}

        event: sources
        data: {"sources": [...], "call_chain": [...], "response_id": "..."}

        event: streaming
        data: {"status": "streaming", "token": "The"}

        event: complete
        data: {"status": "complete", "answer": "...", "timing": {...}}
        ```
        """,
        response_class=StreamingResponse
    )
    async def query_code_stream(request: CodeQueryRequest):
        """
        Process a code assistance query with streaming response.

        Args:
            request: Code query request with question and options

        Returns:
            StreamingResponse with SSE events

        Design Note:
        Uses async generator to yield SSE events. The StreamingResponse
        handles proper HTTP chunked transfer encoding.
        """
        async def generate_events():
            """Async generator for SSE events."""
            try:
                pipeline = await get_code_assistance_pipeline()

                async for event in pipeline.process_query_stream(request):
                    # Format as SSE
                    yield event.to_sse()

            except Exception as e:
                logger.error(f"Streaming query failed: {e}", exc_info=True)
                error_event = SSEEvent(
                    event="error",
                    data={"status": "error", "error": str(e)}
                )
                yield error_event.to_sse()

        return StreamingResponse(
            generate_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    @router.post(
        "/feedback",
        response_model=CodeFeedbackResponse,
        summary="Submit Response Feedback",
        description="""
        Submit feedback on a code assistance response.

        Feedback is used to:
        - Track response quality over time
        - Identify retrieval failures
        - Improve the system through learning

        Feedback types:
        - **Helpful**: Response answered the question correctly
        - **Not Helpful**: Response was incorrect or incomplete

        For unhelpful responses, you can specify:
        - Error category (wrong_method, wrong_class, missing_context, etc.)
        - Expected methods that should have been referenced
        - Comment explaining the issue
        """
    )
    async def submit_feedback(
        feedback: CodeFeedbackRequest
    ) -> CodeFeedbackResponse:
        """
        Submit feedback on a code assistance response.

        Args:
            feedback: Feedback data with response_id and rating

        Returns:
            CodeFeedbackResponse confirming feedback was recorded

        Raises:
            HTTPException: If feedback submission fails
        """
        try:
            pipeline = await get_code_assistance_pipeline()
            response = await pipeline.submit_feedback(feedback)
            return response

        except ValueError as e:
            logger.warning(f"Invalid feedback request: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            logger.error(f"Feedback submission failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/stats",
        response_model=CodeStatsResponse,
        summary="Get Code Entity Statistics",
        description="""
        Get statistics about indexed code entities.

        Returns counts for:
        - Classes
        - Methods
        - Call graph edges
        - Event handlers
        - Database operations
        - Total entities

        Useful for monitoring index health and understanding codebase size.
        """
    )
    async def get_stats() -> CodeStatsResponse:
        """
        Get statistics about indexed code entities.

        Returns:
            CodeStatsResponse with entity counts

        Raises:
            HTTPException: If stats retrieval fails
        """
        try:
            pipeline = await get_code_assistance_pipeline()
            stats = await pipeline.get_stats()
            return stats

        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/health",
        summary="Health Check",
        description="Check if the code assistance pipeline is healthy."
    )
    async def health_check():
        """
        Check code assistance pipeline health.

        Returns:
            Dict with health status and component information
        """
        try:
            pipeline = await get_code_assistance_pipeline()

            # Check LLM health (ResponseGenerator imported at module level)
            generator = ResponseGenerator()
            await generator.initialize()
            llm_health = await generator.health_check()

            # Get stats
            stats = await pipeline.get_stats()

            return {
                "healthy": True,
                "components": {
                    "pipeline": "healthy",
                    "llm": llm_health,
                    "mongodb": {
                        "healthy": True,
                        "entities": stats.code_entities
                    }
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }

    return router


# =============================================================================
# Convenience function for main.py integration
# =============================================================================

def register_code_routes(app, prefix: str = "/api/code"):
    """
    Register code assistance routes with a FastAPI app.

    This is a convenience function for easy integration into
    the main FastAPI application.

    Args:
        app: FastAPI application instance
        prefix: URL prefix for all routes (default: /api/code)

    Usage:
        from fastapi import FastAPI
        from code_assistance_pipeline.routes import register_code_routes

        app = FastAPI()
        register_code_routes(app)
    """
    router = create_code_routes()
    app.include_router(router, prefix=prefix, tags=["Code Assistance"])
    logger.info(f"Code assistance routes registered at {prefix}")
