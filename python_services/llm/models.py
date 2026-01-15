"""
Pydantic models for LLM tracing and monitoring.

Defines data structures for:
- LLM requests and responses
- Trace documents stored in MongoDB
- Query filters for trace retrieval
- Aggregated statistics
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class LLMEndpoint(str, Enum):
    """Available LLM endpoints."""
    SQL = "http://localhost:8080"
    GENERAL = "http://localhost:8081"
    CODE = "http://localhost:8082"


class Pipeline(str, Enum):
    """Pipeline identifiers."""
    SQL = "sql"
    AUDIO = "audio"
    QUERY = "query"
    GIT = "git"
    CODE_FLOW = "code_flow"
    CODE_ASSISTANCE = "code_assistance"
    DOCUMENT_AGENT = "document_agent"


class LLMRequest(BaseModel):
    """Request sent to llama.cpp endpoint."""
    prompt: str
    max_tokens: int = Field(default=2048, alias="n_predict")
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = Field(default_factory=list, alias="stop")
    
    class Config:
        populate_by_name = True


class LLMTimings(BaseModel):
    """Timing information from llama.cpp response."""
    prompt_n: int = 0                    # Number of prompt tokens
    prompt_ms: float = 0.0               # Time to process prompt
    predicted_n: int = 0                 # Number of generated tokens
    predicted_ms: float = 0.0            # Time to generate
    predicted_per_second: float = 0.0    # Generation speed


class LLMResponse(BaseModel):
    """Response from llama.cpp endpoint."""
    content: str
    model: Optional[str] = None
    tokens_evaluated: int = 0            # Prompt tokens
    tokens_predicted: int = 0            # Completion tokens
    truncated: bool = False
    stopped_eos: bool = False
    stopped_limit: bool = False
    stopped_word: bool = False
    stop_reason: Optional[str] = None
    timings: Optional[LLMTimings] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_evaluated + self.tokens_predicted

    @property
    def success(self) -> bool:
        """Returns True if response has content (for compatibility)."""
        return bool(self.content)

    @property
    def text(self) -> str:
        """Alias for content (for compatibility)."""
        return self.content


class TraceStatus(str, Enum):
    """Trace outcome status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TraceContext(BaseModel):
    """Contextual information for the trace."""
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Pipeline context
    user_question: Optional[str] = None
    database: Optional[str] = None
    tables_used: List[str] = Field(default_factory=list)
    audio_file: Optional[str] = None
    code_file: Optional[str] = None
    retrieved_docs: List[str] = Field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceOutcome(BaseModel):
    """Outcome/result information."""
    status: TraceStatus = TraceStatus.SUCCESS
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Pipeline-specific outcomes
    validation_passed: Optional[bool] = None
    executed: Optional[bool] = None
    rows_returned: Optional[int] = None
    user_feedback: Optional[Literal["positive", "negative", "none"]] = None


class LLMTrace(BaseModel):
    """
    Complete trace document for MongoDB storage.
    
    Captures full request/response cycle with timing, context, and outcome.
    """
    # Identifiers
    trace_id: str = Field(..., description="Unique trace identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Pipeline info
    pipeline: Pipeline
    operation: str = Field(..., description="Operation name (e.g., 'generate_sql_query')")
    endpoint: str
    model: Optional[str] = None
    
    # Request
    request: LLMRequest
    
    # Response (None if error before response)
    response: Optional[LLMResponse] = None
    
    # Timing
    total_duration_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None
    
    # Context
    context: TraceContext = Field(default_factory=TraceContext)
    
    # Outcome
    outcome: TraceOutcome = Field(default_factory=TraceOutcome)
    
    # Tags for filtering
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


# Query/Filter Models

class TraceFilter(BaseModel):
    """Filters for querying traces."""
    pipeline: Optional[Pipeline] = None
    status: Optional[TraceStatus] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Performance filters
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    min_tokens: Optional[int] = None
    
    # Pagination
    skip: int = 0
    limit: int = 100
    
    # Sort
    sort_by: str = "timestamp"
    sort_order: Literal["asc", "desc"] = "desc"


class TraceStats(BaseModel):
    """Aggregated statistics for traces."""
    total_traces: int = 0
    success_count: int = 0
    error_count: int = 0
    
    # Token stats
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    
    # Latency stats
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    
    # Throughput
    avg_tokens_per_second: float = 0.0
    
    # By pipeline
    by_pipeline: Dict[str, int] = Field(default_factory=dict)
    
    # By status
    by_status: Dict[str, int] = Field(default_factory=dict)
    
    # Time range
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class RealtimeMetrics(BaseModel):
    """Real-time metrics for dashboard."""
    # Current state
    active_requests: int = 0
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    
    # Error rate
    error_rate_last_hour: float = 0.0
    
    # Latency (last hour)
    avg_latency_ms: float = 0.0
    
    # Tokens (last hour)
    total_tokens_last_hour: int = 0
    
    # By endpoint health
    endpoint_status: Dict[str, bool] = Field(default_factory=dict)
    
    # Recent errors
    recent_errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Last updated
    updated_at: datetime = Field(default_factory=datetime.utcnow)
