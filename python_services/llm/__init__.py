"""
LLM Service Module

Provides traced LLM clients and monitoring for all pipelines.
All LLM calls are automatically logged to MongoDB with full
request/response capture, timing metrics, and context.
"""
from .models import (
    Pipeline,
    LLMEndpoint,
    LLMRequest,
    LLMResponse,
    LLMTrace,
    LLMTimings,
    TraceContext,
    TraceOutcome,
    TraceStatus,
    TraceFilter,
    TraceStats,
    RealtimeMetrics,
)
from .client import (
    TracedLLMClient,
    create_sql_client,
    create_audio_client,
    create_query_client,
    create_git_client,
    create_code_flow_client,
    create_code_assistance_client,
    create_document_client,
)
from .service import LLMService
from .repository import LLMTraceRepository
from .integration import (
    get_llm_service,
    reset_llm_service,
    # SQL helpers
    generate_sql,
    validate_sql_with_llm,
    # Audio helpers
    summarize_transcription,
    analyze_call_content,
    # Query/RAG helpers
    generate_rag_response,
    # Code Flow helpers
    analyze_code_flow,
    # Code Assistance helpers
    generate_code_completion,
    explain_code,
    # Git helpers
    analyze_git_diff,
    generate_commit_message,
    # Document Agent helpers
    process_document,
    answer_document_question,
    # Generic helper
    generate_text,
)

__all__ = [
    # Models
    "Pipeline",
    "LLMEndpoint",
    "LLMRequest",
    "LLMResponse",
    "LLMTrace",
    "LLMTimings",
    "TraceContext",
    "TraceOutcome",
    "TraceStatus",
    "TraceFilter",
    "TraceStats",
    "RealtimeMetrics",
    # Client
    "TracedLLMClient",
    "create_sql_client",
    "create_audio_client",
    "create_query_client",
    "create_git_client",
    "create_code_flow_client",
    "create_code_assistance_client",
    "create_document_client",
    # Service
    "LLMService",
    # Repository
    "LLMTraceRepository",
    # Integration helpers
    "get_llm_service",
    "reset_llm_service",
    "generate_sql",
    "validate_sql_with_llm",
    "summarize_transcription",
    "analyze_call_content",
    "generate_rag_response",
    "analyze_code_flow",
    "generate_code_completion",
    "explain_code",
    "analyze_git_diff",
    "generate_commit_message",
    "process_document",
    "answer_document_question",
    "generate_text",
]
