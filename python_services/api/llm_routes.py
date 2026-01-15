"""
LLM Routes - FastAPI router for LLM operations.

Endpoints:
- POST /llm/generate - Text generation (non-streaming)
- POST /llm/generate-stream - SSE streaming generation
- GET /llm/health - Health check for LLM service
- GET /llm/cache/stats - Cache statistics
- DELETE /llm/cache - Clear cache

Architecture Notes:
------------------
These endpoints provide a clean REST API over the LLMService.
The streaming endpoint uses Server-Sent Events (SSE) for real-time
response streaming, which is ideal for chat interfaces.

SSE Design:
- Uses text/event-stream content type
- Yields JSON-encoded chunks
- Final chunk contains done=true and token usage
- Properly handles client disconnection
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


# ==============================================================================
# Request/Response Models
# ==============================================================================

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="User prompt for generation")
    system: str = Field("", description="System prompt for context")
    model: Optional[str] = Field(None, description="Override model (uses configured default if None)")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens for completion")
    use_cache: bool = Field(True, description="Whether to use response cache")
    use_sql_model: bool = Field(False, description="Use the dedicated SQL model")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    success: bool
    response: str = ""
    error: Optional[str] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)
    model: str = ""
    generation_time_ms: int = 0
    cached: bool = False


class StreamChunkResponse(BaseModel):
    """Model for a single stream chunk."""
    content: str
    done: bool = False
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    healthy: bool
    host: str = ""
    models_available: int = 0
    models: List[str] = Field(default_factory=list)
    configured_model: str = ""
    sql_model: str = ""
    error: Optional[str] = None
    use_dedicated_endpoints: bool = True
    endpoints: Dict[str, Any] = Field(default_factory=dict)


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    cache_size: int
    cache_max_size: int
    cache_ttl: int


# ==============================================================================
# Endpoints
# ==============================================================================

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text using the configured LLM.

    This endpoint provides synchronous text generation. For real-time
    streaming responses, use the /generate-stream endpoint instead.

    The response includes token usage statistics and generation time,
    which are useful for monitoring and cost tracking.
    """
    from services.llm_service import get_llm_service

    try:
        service = await get_llm_service()

        result = await service.generate(
            prompt=request.prompt,
            system=request.system,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_cache=request.use_cache,
            use_sql_model=request.use_sql_model,
        )

        return GenerateResponse(
            success=result.success,
            response=result.response,
            error=result.error,
            token_usage=result.token_usage,
            model=result.model,
            generation_time_ms=result.generation_time_ms,
            cached=result.cached,
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerateResponse(
            success=False,
            error=str(e)
        )


@router.post("/generate-stream")
async def generate_stream(request: GenerateRequest, http_request: Request):
    """
    Generate text with streaming response via Server-Sent Events (SSE).

    This endpoint streams the LLM response in real-time, which provides
    a better user experience for chat interfaces. Each chunk is sent
    as a JSON-encoded SSE event.

    Event format:
    ```
    data: {"content": "Hello", "done": false}

    data: {"content": " world", "done": false}

    data: {"content": "", "done": true, "token_usage": {...}}
    ```

    The final event has done=true and includes token usage statistics.
    """
    from services.llm_service import get_llm_service

    async def event_generator():
        """Generate SSE events from LLM stream."""
        try:
            service = await get_llm_service()

            async for chunk in service.generate_stream(
                prompt=request.prompt,
                system=request.system,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_sql_model=request.use_sql_model,
            ):
                # Check if client disconnected
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    break

                # Format as SSE event
                event_data = {
                    "content": chunk.content,
                    "done": chunk.done,
                }
                if chunk.error:
                    event_data["error"] = chunk.error
                if chunk.token_usage:
                    event_data["token_usage"] = chunk.token_usage

                yield f"data: {json.dumps(event_data)}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'content': '', 'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """
    Get LLM response cache statistics.

    Returns current cache size, maximum size, and TTL configuration.
    """
    from services.llm_service import get_llm_service

    try:
        service = await get_llm_service()
        stats = await service.get_cache_stats()

        return CacheStatsResponse(
            cache_size=stats.get("cache_size", 0),
            cache_max_size=stats.get("cache_max_size", 100),
            cache_ttl=stats.get("cache_ttl", 300),
        )

    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_cache():
    """
    Clear the LLM response cache.

    Useful when you want to force fresh generations for all prompts.
    """
    from services.llm_service import get_llm_service

    try:
        service = await get_llm_service()
        await service.clear_cache()

        return {"success": True, "message": "Cache cleared"}

    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
