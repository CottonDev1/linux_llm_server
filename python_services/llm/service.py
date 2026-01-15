"""
High-level LLM Service interface.

Provides a singleton service for accessing LLM clients and monitoring data.
"""
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from .client import TracedLLMClient
from .repository import LLMTraceRepository
from .models import Pipeline, TraceFilter, TraceStats, RealtimeMetrics, LLMTrace
from .endpoints import get_all_endpoint_health


class LLMService:
    """
    Centralized LLM service for all pipelines.
    
    Usage:
        service = LLMService(mongodb_uri="mongodb://EWRSPT-AI:27018")
        
        # Get client for a pipeline
        sql_client = service.get_client(Pipeline.SQL)
        
        # Get monitoring data
        stats = await service.get_stats(hours=24)
        metrics = await service.get_realtime_metrics()
    """
    
    _instance: Optional["LLMService"] = None
    
    def __init__(self, mongodb_uri: str, database_name: str = "llm_website"):
        """Initialize LLM service."""
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.repository = LLMTraceRepository(mongodb_uri, database_name)
        self._clients: Dict[Pipeline, TracedLLMClient] = {}
    
    @classmethod
    def get_instance(cls, mongodb_uri: Optional[str] = None, database_name: str = "llm_website") -> "LLMService":
        """Get singleton instance."""
        if cls._instance is None:
            if mongodb_uri is None:
                raise ValueError("mongodb_uri required for first initialization")
            cls._instance = cls(mongodb_uri, database_name)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
    
    def get_client(self, pipeline: Pipeline) -> TracedLLMClient:
        """Get or create traced client for pipeline."""
        if pipeline not in self._clients:
            self._clients[pipeline] = TracedLLMClient(
                mongodb_uri=self.mongodb_uri,
                pipeline=pipeline,
                database_name=self.database_name,
            )
        return self._clients[pipeline]
    
    # Convenience methods for getting pipeline-specific clients
    
    def get_sql_client(self) -> TracedLLMClient:
        """Get client for SQL pipeline."""
        return self.get_client(Pipeline.SQL)
    
    def get_audio_client(self) -> TracedLLMClient:
        """Get client for Audio pipeline."""
        return self.get_client(Pipeline.AUDIO)
    
    def get_query_client(self) -> TracedLLMClient:
        """Get client for Query/RAG pipeline."""
        return self.get_client(Pipeline.QUERY)
    
    def get_git_client(self) -> TracedLLMClient:
        """Get client for Git pipeline."""
        return self.get_client(Pipeline.GIT)
    
    def get_code_flow_client(self) -> TracedLLMClient:
        """Get client for Code Flow pipeline."""
        return self.get_client(Pipeline.CODE_FLOW)
    
    def get_code_assistance_client(self) -> TracedLLMClient:
        """Get client for Code Assistance pipeline."""
        return self.get_client(Pipeline.CODE_ASSISTANCE)
    
    def get_document_client(self) -> TracedLLMClient:
        """Get client for Document Agent pipeline."""
        return self.get_client(Pipeline.DOCUMENT_AGENT)
    
    # Monitoring methods
    
    async def get_stats(
        self,
        hours: int = 24,
        pipeline: Optional[Pipeline] = None,
    ) -> TraceStats:
        """Get statistics for the last N hours."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        return await self.repository.get_stats(start_time, end_time, pipeline)
    
    async def get_realtime_metrics(self) -> RealtimeMetrics:
        """Get real-time metrics."""
        metrics = await self.repository.get_realtime_metrics()
        metrics.endpoint_status = await get_all_endpoint_health()
        return metrics
    
    async def get_traces(self, filter: TraceFilter) -> List[LLMTrace]:
        """Get traces matching filter."""
        return await self.repository.query_traces(filter)
    
    async def get_trace(self, trace_id: str) -> Optional[LLMTrace]:
        """Get single trace by ID."""
        return await self.repository.get_trace(trace_id)
    
    async def count_traces(self, filter: TraceFilter) -> int:
        """Count traces matching filter."""
        return await self.repository.count_traces(filter)
    
    async def get_latency_timeline(
        self,
        hours: int = 24,
        interval_minutes: int = 5,
        pipeline: Optional[Pipeline] = None,
    ) -> List[dict]:
        """Get latency over time for charts."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        return await self.repository.get_latency_timeline(
            start_time, end_time, interval_minutes, pipeline
        )
    
    async def get_token_timeline(
        self,
        hours: int = 24,
        interval_minutes: int = 60,
    ) -> List[dict]:
        """Get token usage over time for charts."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        return await self.repository.get_token_usage_timeline(
            start_time, end_time, interval_minutes
        )
    
    async def cleanup_old_traces(self, days_to_keep: int = 30) -> int:
        """Delete old traces to manage storage."""
        return await self.repository.delete_old_traces(days_to_keep)
