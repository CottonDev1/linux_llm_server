"""
Traced LLM Client for llama.cpp endpoints.

All LLM calls go through this client, which automatically:
- Logs full request/response to MongoDB
- Captures timing metrics from llama.cpp
- Handles errors gracefully
- Supports both async and sync operations
"""
import time
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx

from .models import (
    LLMRequest, LLMResponse, LLMTrace, LLMTimings,
    TraceContext, TraceOutcome, TraceStatus,
    Pipeline
)
from .repository import LLMTraceRepository
from .endpoints import get_endpoint_for_pipeline, get_model_for_endpoint


class TracedLLMClient:
    """
    LLM Client with automatic MongoDB tracing.
    
    Usage:
        client = TracedLLMClient(
            mongodb_uri="mongodb://localhost:27017",
            pipeline=Pipeline.SQL
        )
        
        response = await client.generate(
            prompt="Generate SQL for...",
            context=TraceContext(user_id="admin", user_question="Show all users")
        )
    """
    
    def __init__(
        self,
        mongodb_uri: str,
        pipeline: Pipeline,
        database_name: str = "llm_website",
        timeout: int = 300,
        endpoint_override: Optional[str] = None,
    ):
        """
        Initialize traced LLM client.
        
        Args:
            mongodb_uri: MongoDB connection string
            pipeline: Pipeline using this client
            database_name: MongoDB database name
            timeout: Request timeout in seconds
            endpoint_override: Override default endpoint for pipeline
        """
        self.pipeline = pipeline
        self.endpoint = endpoint_override or get_endpoint_for_pipeline(pipeline)
        self.model = get_model_for_endpoint(self.endpoint)
        self.timeout = timeout
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        
        # Initialize repository
        self.repository = LLMTraceRepository(mongodb_uri, database_name)
        
        # Validate endpoint is local
        if not self.endpoint.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                f"Only local LLM endpoints allowed. Got: {self.endpoint}. "
                "External APIs are not permitted."
            )
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"trace_{timestamp}_{self.pipeline.value}_{unique}"
    
    def _parse_llama_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse llama.cpp response into LLMResponse model (supports both legacy and OpenAI formats)."""
        # Check if this is OpenAI-compatible format
        if "choices" in data:
            return self._parse_openai_response(data)

        # Legacy /completion format
        timings_data = data.get("timings", {})
        timings = LLMTimings(
            prompt_n=timings_data.get("prompt_n", 0),
            prompt_ms=timings_data.get("prompt_ms", 0),
            predicted_n=timings_data.get("predicted_n", 0),
            predicted_ms=timings_data.get("predicted_ms", 0),
            predicted_per_second=timings_data.get("predicted_per_second", 0),
        )

        # Determine stop reason
        stop_reason = None
        if data.get("stopped_eos"):
            stop_reason = "eos"
        elif data.get("stopped_word"):
            stop_reason = "stop_sequence"
        elif data.get("stopped_limit"):
            stop_reason = "max_tokens"

        return LLMResponse(
            content=data.get("content", ""),
            model=data.get("model"),
            tokens_evaluated=data.get("tokens_evaluated", 0),
            tokens_predicted=data.get("tokens_predicted", 0),
            truncated=data.get("truncated", False),
            stopped_eos=data.get("stopped_eos", False),
            stopped_limit=data.get("stopped_limit", False),
            stopped_word=data.get("stopped_word", False),
            stop_reason=stop_reason,
            timings=timings,
        )

    def _parse_openai_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Parse OpenAI-compatible response format."""
        choices = data.get("choices", [])
        content = ""
        finish_reason = None

        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason")

        # Map finish_reason to our stop reasons
        stop_reason = None
        stopped_eos = False
        stopped_limit = False
        stopped_word = False

        if finish_reason == "stop":
            stopped_eos = True
            stop_reason = "eos"
        elif finish_reason == "length":
            stopped_limit = True
            stop_reason = "max_tokens"

        # Extract usage info
        usage = data.get("usage", {})

        # Create empty timings (OpenAI format doesn't include detailed timings)
        timings = LLMTimings(
            prompt_n=usage.get("prompt_tokens", 0),
            prompt_ms=0,
            predicted_n=usage.get("completion_tokens", 0),
            predicted_ms=0,
            predicted_per_second=0,
        )

        return LLMResponse(
            content=content,
            model=data.get("model"),
            tokens_evaluated=usage.get("prompt_tokens", 0),
            tokens_predicted=usage.get("completion_tokens", 0),
            truncated=False,
            stopped_eos=stopped_eos,
            stopped_limit=stopped_limit,
            stopped_word=stopped_word,
            stop_reason=stop_reason,
            timings=timings,
        )
    
    async def generate(
        self,
        prompt: str,
        operation: str = "generate",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        context: Optional[TraceContext] = None,
        tags: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate completion with automatic tracing.
        
        Args:
            prompt: Input prompt
            operation: Operation name for trace
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            context: Additional context for trace
            tags: Tags for filtering
        
        Returns:
            LLMResponse with content and metrics
        """
        trace_id = self._generate_trace_id()
        start_time = time.time()
        
        request = LLMRequest(
            prompt=prompt,
            n_predict=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )
        
        trace = LLMTrace(
            trace_id=trace_id,
            pipeline=self.pipeline,
            operation=operation,
            endpoint=self.endpoint,
            model=self.model,
            request=request,
            context=context or TraceContext(),
            tags=tags or [],
        )
        
        llm_response = None
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use OpenAI-compatible API endpoint
                response = await client.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop": stop or None,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            llm_response = self._parse_llama_response(data)
            trace.response = llm_response
            trace.outcome = TraceOutcome(status=TraceStatus.SUCCESS)
            
        except httpx.TimeoutException as e:
            trace.outcome = TraceOutcome(
                status=TraceStatus.TIMEOUT,
                error_message=f"Request timed out after {self.timeout}s",
                error_type="TimeoutError",
            )
            trace.total_duration_ms = (time.time() - start_time) * 1000
            self._store_trace_safe(trace)
            raise
        
        except httpx.HTTPStatusError as e:
            trace.outcome = TraceOutcome(
                status=TraceStatus.ERROR,
                error_message=str(e),
                error_type="HTTPError",
            )
            trace.total_duration_ms = (time.time() - start_time) * 1000
            self._store_trace_safe(trace)
            raise
        
        except Exception as e:
            trace.outcome = TraceOutcome(
                status=TraceStatus.ERROR,
                error_message=str(e),
                error_type=type(e).__name__,
            )
            trace.total_duration_ms = (time.time() - start_time) * 1000
            self._store_trace_safe(trace)
            raise
        
        trace.total_duration_ms = (time.time() - start_time) * 1000
        self._store_trace_safe(trace)
        
        # Attach trace_id to response for later outcome updates
        llm_response._trace_id = trace_id
        
        return llm_response
    
    def _store_trace_safe(self, trace: LLMTrace):
        """Store trace without failing the LLM call if storage fails."""
        try:
            self.repository.insert_trace_sync(trace)
        except Exception as storage_error:
            # Log storage error but don't fail the LLM call
            print(f"Warning: Failed to store trace {trace.trace_id}: {storage_error}")
    
    def generate_sync(
        self,
        prompt: str,
        operation: str = "generate",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        context: Optional[TraceContext] = None,
        tags: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Synchronous version of generate()."""
        return asyncio.run(
            self.generate(
                prompt=prompt,
                operation=operation,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                context=context,
                tags=tags,
            )
        )
    
    def update_trace_outcome(
        self,
        trace_id: str,
        validation_passed: Optional[bool] = None,
        executed: Optional[bool] = None,
        rows_returned: Optional[int] = None,
        user_feedback: Optional[str] = None,
    ):
        """
        Update trace with outcome information after initial generation.
        
        Call this after validation, execution, or receiving user feedback.
        """
        self.repository.update_trace_outcome(
            trace_id=trace_id,
            validation_passed=validation_passed,
            executed=executed,
            rows_returned=rows_returned,
            user_feedback=user_feedback,
        )


# Factory functions for each pipeline

def create_sql_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for SQL pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.SQL, **kwargs)


def create_audio_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for Audio pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.AUDIO, **kwargs)


def create_query_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for Query/RAG pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.QUERY, **kwargs)


def create_git_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for Git pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.GIT, **kwargs)


def create_code_flow_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for Code Flow pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.CODE_FLOW, **kwargs)


def create_code_assistance_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for Code Assistance pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.CODE_ASSISTANCE, **kwargs)


def create_document_client(mongodb_uri: str, **kwargs) -> TracedLLMClient:
    """Create traced client for Document Agent pipeline."""
    return TracedLLMClient(mongodb_uri=mongodb_uri, pipeline=Pipeline.DOCUMENT_AGENT, **kwargs)
