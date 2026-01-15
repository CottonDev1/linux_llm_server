"""
Document Orchestrator Models
============================

Pydantic models for the knowledge base retrieval pipeline orchestration.

Design Rationale:
-----------------
These models define the data structures that flow through the multi-agent
orchestration pipeline. Each step of the pipeline (query understanding,
retrieval, grading, generation, validation) has dedicated models to ensure
type safety and clear interfaces between components.

Key Design Decisions:
1. All models inherit from BaseModel for validation and serialization
2. Enums provide type-safe classification of query types and pipeline states
3. Timing metrics are embedded in each step for performance monitoring
4. Models support both streaming and batch processing modes
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


# =============================================================================
# Query Classification
# =============================================================================

class QueryIntent(str, Enum):
    """
    Classification of user query intent for intelligent routing.

    Routing Strategy:
    - SIMPLE: Skip retrieval, direct LLM answer (definitions, yes/no)
    - FACTUAL: Standard single-hop retrieval
    - ANALYTICAL: Multi-hop retrieval with synthesis
    - TEMPORAL: Time-filtered retrieval with recency weighting
    - PROCEDURAL: Step-by-step retrieval with ordering
    - COMPARISON: Multi-entity retrieval with side-by-side formatting
    - AGGREGATION: Full-collection scan with statistics
    """
    SIMPLE = "simple"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"


class PipelineStage(str, Enum):
    """Current stage of the pipeline execution."""
    PENDING = "pending"
    QUERY_UNDERSTANDING = "query_understanding"
    RETRIEVAL = "retrieval"
    GRADING = "grading"
    GENERATION = "generation"
    VALIDATION = "validation"
    CORRECTION = "correction"
    COMPLETE = "complete"
    FAILED = "failed"


class ValidationStatus(str, Enum):
    """Result of answer validation checks."""
    PASSED = "passed"
    FAILED_RELEVANCY = "failed_relevancy"
    FAILED_FAITHFULNESS = "failed_faithfulness"
    FAILED_COMPLETENESS = "failed_completeness"
    SKIPPED = "skipped"


# =============================================================================
# Pipeline Input/Output
# =============================================================================

class QueryRequest(BaseModel):
    """
    Incoming query request from user.

    This is the entry point for the orchestration pipeline.
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    query: str = Field(..., description="User's natural language query")
    user_id: Optional[str] = Field(default=None, description="User identifier for personalization")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID")

    # Optional context
    previous_queries: List[str] = Field(default_factory=list, description="Previous queries in session")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Full conversation history [{role, content}]")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Pre-defined search filters")

    # Options
    stream: bool = Field(default=False, description="Enable SSE streaming")
    max_documents: int = Field(default=5, ge=1, le=20, description="Max documents to retrieve")
    skip_validation: bool = Field(default=False, description="Skip answer validation step")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QueryResponse(BaseModel):
    """
    Final response from the orchestration pipeline.

    Includes the answer, sources, confidence scores, and pipeline metadata.
    """
    query_id: str
    query: str
    answer: str = Field(default="", description="Generated answer")

    # Source attribution
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents used")
    citations: List[str] = Field(default_factory=list, description="Inline citations [1], [2], etc.")

    # Confidence and quality
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence score")
    validation_passed: bool = Field(default=True, description="Whether validation checks passed")
    validation_details: Optional[Dict[str, Any]] = Field(default=None, description="Validation check results")

    # Routing info
    query_intent: QueryIntent = Field(default=QueryIntent.FACTUAL)
    retrieval_used: bool = Field(default=True, description="Whether retrieval was performed")
    correction_applied: bool = Field(default=False, description="Whether self-correction was applied")

    # Timing
    total_time_ms: int = Field(default=0, description="Total processing time")
    stage_timings: Dict[str, int] = Field(default_factory=dict, description="Time per pipeline stage")

    # Token usage
    token_usage: Dict[str, int] = Field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    })

    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if failed")
    completed_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Query Understanding Stage
# =============================================================================

class ExtractedEntity(BaseModel):
    """Entity extracted from user query."""
    text: str
    entity_type: str  # person, organization, date, table, column, etc.
    normalized: Optional[str] = None
    confidence: float = 1.0


class QueryAnalysisResult(BaseModel):
    """
    Result of query understanding stage.

    Contains classified intent, extracted entities, and query expansions.
    """
    original_query: str
    query_intent: QueryIntent
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Entity extraction
    entities: List[ExtractedEntity] = Field(default_factory=list)

    # Query transformations
    rewritten_query: Optional[str] = Field(default=None, description="Query rewritten for retrieval")
    expanded_queries: List[str] = Field(default_factory=list, description="Alternative query formulations")

    # Contextual signals
    is_follow_up: bool = Field(default=False, description="Part of multi-turn conversation")
    requires_retrieval: bool = Field(default=True, description="Whether retrieval is needed")
    suggested_filters: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    processing_time_ms: int = 0


# =============================================================================
# Retrieval Stage
# =============================================================================

class RetrievedDocument(BaseModel):
    """
    A document chunk retrieved from the knowledge base.

    Includes both vector similarity score and optional BM25 score for
    hybrid search with Reciprocal Rank Fusion (RRF).
    """
    document_id: str
    chunk_id: str
    content: str
    title: Optional[str] = None

    # Source metadata
    source_file: Optional[str] = None
    department: Optional[str] = None
    doc_type: Optional[str] = None

    # Scores
    vector_score: float = Field(default=0.0, description="Cosine similarity score")
    bm25_score: float = Field(default=0.0, description="BM25 keyword score")
    rrf_score: float = Field(default=0.0, description="Combined RRF score")

    # Position for context assembly
    chunk_index: int = 0
    total_chunks: int = 1

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """
    Result of the retrieval stage.

    Contains retrieved documents from both vector and BM25 search,
    combined using Reciprocal Rank Fusion.
    """
    documents: List[RetrievedDocument] = Field(default_factory=list)

    # Search statistics
    vector_candidates: int = Field(default=0, description="Documents from vector search")
    bm25_candidates: int = Field(default=0, description="Documents from BM25 search")
    total_unique: int = Field(default=0, description="Unique documents after fusion")

    # Query used
    search_query: str = ""
    expanded_queries_used: List[str] = Field(default_factory=list)

    # Timing
    vector_search_ms: int = 0
    bm25_search_ms: int = 0
    fusion_ms: int = 0
    total_time_ms: int = 0


# =============================================================================
# Grading Stage
# =============================================================================

class GradedDocument(BaseModel):
    """
    A document with relevance grade from the grading agent.

    CRAG Pattern: Each document is graded as relevant/ambiguous/irrelevant
    to determine if corrective retrieval is needed.
    """
    document: RetrievedDocument

    # Grading result
    relevance: str = Field(default="relevant", description="relevant|ambiguous|irrelevant")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(default=None, description="Why this grade was assigned")

    # Keep for context
    include_in_context: bool = Field(default=True)


class GradingResult(BaseModel):
    """
    Result of the document grading stage.

    If average relevance is too low, triggers corrective retrieval.
    """
    graded_documents: List[GradedDocument] = Field(default_factory=list)

    # Aggregated scores
    average_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    relevant_count: int = 0
    ambiguous_count: int = 0
    irrelevant_count: int = 0

    # Corrective action
    needs_correction: bool = Field(default=False, description="Trigger corrective retrieval")
    correction_reason: Optional[str] = None

    # Timing
    processing_time_ms: int = 0


# =============================================================================
# Generation Stage
# =============================================================================

class GenerationRequest(BaseModel):
    """Request to the LLM for answer generation."""
    query: str
    context: str = Field(description="Formatted context from retrieved documents")
    system_prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False


class GenerationResult(BaseModel):
    """Result from answer generation."""
    answer: str = ""

    # Token tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Timing
    generation_time_ms: int = 0

    # Model info
    model_used: str = ""

    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None


# =============================================================================
# Validation Stage
# =============================================================================

class RelevancyCheck(BaseModel):
    """Check if answer addresses the user's question."""
    passed: bool = False
    score: float = 0.0
    reasoning: Optional[str] = None


class FaithfulnessCheck(BaseModel):
    """Check if answer is grounded in retrieved context (no hallucinations)."""
    passed: bool = False
    score: float = 0.0
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class CompletenessCheck(BaseModel):
    """Check if answer fully addresses all aspects of the query."""
    passed: bool = False
    score: float = 0.0
    missing_aspects: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class ValidationResult(BaseModel):
    """
    Complete validation result from the validation agent.

    Multi-stage validation ensures answer quality before returning to user.
    """
    is_valid: bool = Field(default=True)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Individual checks
    relevancy: RelevancyCheck = Field(default_factory=RelevancyCheck)
    faithfulness: FaithfulnessCheck = Field(default_factory=FaithfulnessCheck)
    completeness: CompletenessCheck = Field(default_factory=CompletenessCheck)

    # Issues found
    issues: List[str] = Field(default_factory=list)

    # Recommendation
    needs_correction: bool = False
    correction_hints: List[str] = Field(default_factory=list)

    # Timing
    processing_time_ms: int = 0


# =============================================================================
# Pipeline State
# =============================================================================

class PipelineState(BaseModel):
    """
    Complete state of the orchestration pipeline.

    Tracks all intermediate results and enables retry/recovery.
    Used for both sync and streaming modes.
    """
    request: QueryRequest
    current_stage: PipelineStage = PipelineStage.PENDING

    # Stage results
    query_analysis: Optional[QueryAnalysisResult] = None
    retrieval_result: Optional[RetrievalResult] = None
    grading_result: Optional[GradingResult] = None
    generation_result: Optional[GenerationResult] = None
    validation_result: Optional[ValidationResult] = None

    # Correction tracking
    correction_attempts: int = 0
    max_corrections: int = 2

    # Final output
    response: Optional[QueryResponse] = None

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Error tracking
    error: Optional[str] = None
    error_stage: Optional[PipelineStage] = None


# =============================================================================
# Streaming Events
# =============================================================================

class StreamEventType(str, Enum):
    """Types of events sent via SSE during pipeline execution."""
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    PROGRESS = "progress"
    DOCUMENT_FOUND = "document_found"
    GENERATION_TOKEN = "generation_token"
    VALIDATION_CHECK = "validation_check"
    ERROR = "error"
    COMPLETE = "complete"


class StreamEvent(BaseModel):
    """
    Server-Sent Event for real-time pipeline updates.

    Enables UI to show progress, retrieved documents, and streaming answer.
    """
    event_type: StreamEventType
    query_id: str
    stage: PipelineStage

    # Event-specific data
    data: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    elapsed_ms: int = 0

    def to_sse(self) -> str:
        """Format as SSE message."""
        import json
        payload = self.model_dump(mode='json')
        return f"event: {self.event_type.value}\ndata: {json.dumps(payload)}\n\n"


# =============================================================================
# Feedback Models
# =============================================================================

class FeedbackType(str, Enum):
    """Type of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    RATING = "rating"


class FeedbackRecord(BaseModel):
    """
    User feedback on a query response.

    Used by the learning agent for continuous improvement.
    """
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    query_id: str
    feedback_type: FeedbackType

    # Original query and response
    query: str
    answer: str

    # Feedback details
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    correction: Optional[str] = Field(default=None, description="User-provided correct answer")
    comment: Optional[str] = None

    # Context for learning
    sources_used: List[str] = Field(default_factory=list)
    query_intent: Optional[QueryIntent] = None

    # Metadata
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Configuration
# =============================================================================

class OrchestratorConfig(BaseModel):
    """
    Configuration for the knowledge base orchestrator.

    Tunable parameters for each pipeline stage.
    """
    # Query understanding
    enable_query_expansion: bool = True
    max_expanded_queries: int = 3

    # Retrieval
    default_top_k: int = 5
    vector_weight: float = 0.5  # RRF fusion weight
    bm25_weight: float = 0.5
    min_similarity_threshold: float = 0.4
    enable_hybrid_search: bool = True

    # Grading
    enable_document_grading: bool = True
    min_relevance_threshold: float = 0.5
    trigger_correction_threshold: float = 0.3  # Average relevance below this triggers correction

    # Generation
    max_context_tokens: int = 4000
    generation_max_tokens: int = 512
    generation_temperature: float = 0.0

    # Validation
    enable_validation: bool = True
    enable_self_correction: bool = True
    max_correction_attempts: int = 2

    # Timeouts (ms)
    query_analysis_timeout: int = 5000
    retrieval_timeout: int = 10000
    grading_timeout: int = 10000
    generation_timeout: int = 60000
    validation_timeout: int = 10000

    # Routing thresholds
    simple_query_max_length: int = 50  # Skip retrieval for very short queries

    # Caching
    enable_semantic_cache: bool = False
    cache_ttl_seconds: int = 3600
