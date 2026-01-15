"""
Document Orchestrator Package
=============================

Central coordination layer for the knowledge base retrieval pipeline.

This package provides:
- KnowledgeBaseOrchestrator: Multi-agent coordinator for query processing
- Pipeline models: Data classes for each pipeline stage
- Streaming support: SSE events for real-time UI updates

Architecture:
-------------
The orchestrator implements a CRAG (Corrective RAG) pattern:

    Query --> Understanding --> Retrieval --> Grading --> Generation --> Validation
                                    |            |                          |
                                    +-- Correction Loop (if grading/validation fails)

Usage:
------
    from orchestrator import get_orchestrator, QueryRequest

    # Initialize
    orchestrator = await get_orchestrator()

    # Process query (sync)
    response = await orchestrator.process_query(
        QueryRequest(query="How do I configure the system?")
    )

    # Process query (streaming)
    async for event in orchestrator.process_query_stream(
        QueryRequest(query="...", stream=True)
    ):
        yield event.to_sse()

    # Record feedback
    from orchestrator import FeedbackRecord, FeedbackType
    await orchestrator.record_feedback(
        FeedbackRecord(
            query_id=response.query_id,
            query=response.query,
            answer=response.answer,
            feedback_type=FeedbackType.THUMBS_UP
        )
    )

Configuration:
--------------
    from orchestrator import OrchestratorConfig

    config = OrchestratorConfig(
        enable_hybrid_search=True,
        enable_document_grading=True,
        enable_validation=True,
        max_correction_attempts=2,
    )

    orchestrator = KnowledgeBaseOrchestrator(config=config)
"""

from .models import (
    # Enums
    QueryIntent,
    PipelineStage,
    ValidationStatus,
    StreamEventType,
    FeedbackType,

    # Request/Response
    QueryRequest,
    QueryResponse,

    # Stage Results
    QueryAnalysisResult,
    ExtractedEntity,
    RetrievedDocument,
    RetrievalResult,
    GradedDocument,
    GradingResult,
    GenerationRequest,
    GenerationResult,
    ValidationResult,
    RelevancyCheck,
    FaithfulnessCheck,
    CompletenessCheck,

    # Pipeline State
    PipelineState,

    # Streaming
    StreamEvent,

    # Feedback
    FeedbackRecord,

    # Configuration
    OrchestratorConfig,
)

from .document_orchestrator import (
    KnowledgeBaseOrchestrator,
    get_orchestrator,
    close_orchestrator,
)

__version__ = "1.0.0"

__all__ = [
    # Orchestrator
    "KnowledgeBaseOrchestrator",
    "get_orchestrator",
    "close_orchestrator",

    # Enums
    "QueryIntent",
    "PipelineStage",
    "ValidationStatus",
    "StreamEventType",
    "FeedbackType",

    # Request/Response
    "QueryRequest",
    "QueryResponse",

    # Stage Results
    "QueryAnalysisResult",
    "ExtractedEntity",
    "RetrievedDocument",
    "RetrievalResult",
    "GradedDocument",
    "GradingResult",
    "GenerationRequest",
    "GenerationResult",
    "ValidationResult",
    "RelevancyCheck",
    "FaithfulnessCheck",
    "CompletenessCheck",

    # Pipeline State
    "PipelineState",

    # Streaming
    "StreamEvent",

    # Feedback
    "FeedbackRecord",

    # Configuration
    "OrchestratorConfig",
]
