"""
LLM Monitoring API routes.

Provides REST endpoints for the LLM monitoring dashboard:
- Real-time metrics
- Trace statistics
- Trace querying and filtering
- Timeline data for charts
- Trace cleanup
"""
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import (
    LLMService,
    Pipeline,
    TraceFilter,
    TraceStats,
    RealtimeMetrics,
    LLMTrace,
    TraceStatus
)
from config import MONGODB_URI

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm-monitoring", tags=["LLM Monitoring"])


def get_llm_service() -> LLMService:
    """Dependency to get LLM service instance."""
    return LLMService.get_instance(MONGODB_URI)


# Response Models

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class PipelineInfo(BaseModel):
    id: str
    name: str


class CleanupResponse(BaseModel):
    deleted_count: int
    days_kept: int


# Endpoints

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check LLM monitoring service health."""
    return HealthResponse(status="healthy", timestamp=datetime.utcnow())


@router.get("/realtime", response_model=RealtimeMetrics)
async def get_realtime_metrics(
    service: LLMService = Depends(get_llm_service)
):
    """
    Get real-time metrics for dashboard.
    
    Returns:
    - Requests in last minute/hour
    - Error rate
    - Average latency
    - Token usage
    - Endpoint health status
    - Recent errors
    """
    try:
        return await service.get_realtime_metrics()
    except Exception as e:
        logger.error(f"Failed to get realtime metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=TraceStats)
async def get_stats(
    hours: int = Query(24, ge=1, le=720, description="Number of hours to look back"),
    pipeline: Optional[Pipeline] = Query(None, description="Filter by pipeline"),
    service: LLMService = Depends(get_llm_service)
):
    """
    Get aggregated statistics for traces.
    
    Returns:
    - Total traces, success/error counts
    - Token statistics (total, average)
    - Latency statistics (avg, min, max, percentiles)
    - Breakdown by pipeline and status
    """
    try:
        return await service.get_stats(hours=hours, pipeline=pipeline)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces", response_model=List[LLMTrace])
async def get_traces(
    pipeline: Optional[Pipeline] = Query(None, description="Filter by pipeline"),
    status: Optional[TraceStatus] = Query(None, description="Filter by status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    operation: Optional[str] = Query(None, description="Filter by operation name"),
    min_duration_ms: Optional[float] = Query(None, ge=0, description="Minimum duration in ms"),
    max_duration_ms: Optional[float] = Query(None, ge=0, description="Maximum duration in ms"),
    hours: int = Query(24, ge=1, le=720, description="Number of hours to look back"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    service: LLMService = Depends(get_llm_service)
):
    """
    Query traces with filters.
    
    Supports filtering by:
    - Pipeline type
    - Status (success, error, timeout)
    - User ID
    - Operation name
    - Duration range
    - Time range
    
    Supports pagination via skip/limit.
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        filter = TraceFilter(
            pipeline=pipeline,
            status=status,
            user_id=user_id,
            operation=operation,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            start_time=start_time,
            end_time=end_time,
            skip=skip,
            limit=limit,
        )
        
        return await service.get_traces(filter)
    except Exception as e:
        logger.error(f"Failed to get traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/count")
async def count_traces(
    pipeline: Optional[Pipeline] = Query(None),
    status: Optional[TraceStatus] = Query(None),
    hours: int = Query(24, ge=1, le=720),
    service: LLMService = Depends(get_llm_service)
):
    """Get count of traces matching filter."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        filter = TraceFilter(
            pipeline=pipeline,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )
        
        count = await service.count_traces(filter)
        return {"count": count}
    except Exception as e:
        logger.error(f"Failed to count traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/{trace_id}", response_model=LLMTrace)
async def get_trace(
    trace_id: str,
    service: LLMService = Depends(get_llm_service)
):
    """Get single trace by ID."""
    try:
        trace = await service.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/latency")
async def get_latency_timeline(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    interval_minutes: int = Query(5, ge=1, le=60, description="Interval between data points"),
    pipeline: Optional[Pipeline] = Query(None, description="Filter by pipeline"),
    service: LLMService = Depends(get_llm_service)
):
    """
    Get latency over time for charts.
    
    Returns time-series data with:
    - Timestamp
    - Average, min, max latency
    - Request count per interval
    """
    try:
        return await service.get_latency_timeline(
            hours=hours,
            interval_minutes=interval_minutes,
            pipeline=pipeline,
        )
    except Exception as e:
        logger.error(f"Failed to get latency timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/tokens")
async def get_token_timeline(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    interval_minutes: int = Query(60, ge=5, le=360, description="Interval between data points"),
    service: LLMService = Depends(get_llm_service)
):
    """
    Get token usage over time for charts.
    
    Returns time-series data with:
    - Timestamp
    - Prompt tokens
    - Completion tokens
    - Total tokens
    - Request count per interval
    """
    try:
        return await service.get_token_timeline(
            hours=hours,
            interval_minutes=interval_minutes,
        )
    except Exception as e:
        logger.error(f"Failed to get token timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines", response_model=List[PipelineInfo])
async def get_pipeline_list():
    """Get list of available pipelines."""
    return [
        PipelineInfo(id=p.value, name=p.value.replace("_", " ").title())
        for p in Pipeline
    ]


@router.delete("/traces/cleanup", response_model=CleanupResponse)
async def cleanup_old_traces(
    days_to_keep: int = Query(30, ge=1, le=365, description="Days of traces to keep"),
    service: LLMService = Depends(get_llm_service)
):
    """
    Delete traces older than specified days.
    
    Use with caution - this permanently deletes data.
    """
    try:
        deleted = await service.cleanup_old_traces(days_to_keep)
        return CleanupResponse(deleted_count=deleted, days_kept=days_to_keep)
    except Exception as e:
        logger.error(f"Failed to cleanup traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/recent")
async def get_recent_errors(
    limit: int = Query(20, ge=1, le=100),
    service: LLMService = Depends(get_llm_service)
):
    """Get recent error traces."""
    try:
        filter = TraceFilter(
            status=TraceStatus.ERROR,
            limit=limit,
            sort_by="timestamp",
            sort_order="desc",
        )
        return await service.get_traces(filter)
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slow-requests")
async def get_slow_requests(
    threshold_ms: float = Query(5000, ge=100, description="Minimum duration to consider slow"),
    limit: int = Query(20, ge=1, le=100),
    hours: int = Query(24, ge=1, le=168),
    service: LLMService = Depends(get_llm_service)
):
    """Get traces that took longer than threshold."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        filter = TraceFilter(
            min_duration_ms=threshold_ms,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            sort_by="total_duration_ms",
            sort_order="desc",
        )
        return await service.get_traces(filter)
    except Exception as e:
        logger.error(f"Failed to get slow requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))
