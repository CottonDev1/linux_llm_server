"""
Pydantic models for data validation and serialization
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(str):
    """Custom type for MongoDB ObjectId"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str):
            return v
        raise ValueError("Invalid ObjectId")


# ============================================================================
# Document Models (replaces DocumentationDatabase)
# ============================================================================

class DocumentBase(BaseModel):
    """Base model for documents"""
    title: str
    content: str
    department: str = "general"
    type: str = "documentation"
    subject: Optional[str] = None
    file_name: Optional[str] = None
    file_size: int = 0
    tags: List[str] = Field(default_factory=lambda: ["untagged"])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    """Model for creating a new document"""
    pass


class DocumentUpdate(BaseModel):
    """Model for updating document metadata"""
    title: Optional[str] = None
    department: Optional[str] = None
    type: Optional[str] = None
    subject: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentChunk(BaseModel):
    """Model for a document chunk stored in MongoDB"""
    id: str
    parent_id: str
    chunk_index: int
    total_chunks: int
    title: str
    content: str
    department: str
    type: str
    subject: Optional[str] = None
    file_name: Optional[str] = None
    file_size: int = 0
    upload_date: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    vector: List[float]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentResponse(BaseModel):
    """Model for document response"""
    id: str
    title: str
    content: str
    department: str
    type: str
    subject: Optional[str] = None
    file_name: Optional[str] = None
    file_size: int = 0
    upload_date: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    chunks: int = 1


class DocumentSearchResult(BaseModel):
    """Model for search results"""
    id: str
    parent_id: str
    title: str
    content: str
    department: str
    type: str
    subject: Optional[str] = None
    file_name: Optional[str] = None
    upload_date: datetime
    tags: List[str]
    relevance_score: float
    chunk_index: int
    total_chunks: int


# ============================================================================
# Code Context Models
# ============================================================================

class CodeContextBase(BaseModel):
    """Base model for code context entries"""
    document: str
    database: Optional[str] = None  # Project scope
    category: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    language: str = "csharp"


class CodeContextCreate(CodeContextBase):
    """Model for creating code context"""
    pass


class CodeContextDocument(CodeContextBase):
    """Model for code context stored in MongoDB"""
    id: str
    timestamp: int
    vector: List[float]


class CodeContextSearchResult(BaseModel):
    """Model for code context search results"""
    id: str
    content: str
    similarity: float
    distance: float
    metadata: Dict[str, Any]
    timestamp: int


# ============================================================================
# SQL Knowledge Models (replaces SQLKnowledgeDB)
# ============================================================================

class SQLKnowledgeBase(BaseModel):
    """Base model for SQL knowledge entries"""
    content: str
    knowledge_type: str  # schema, stored_procedure, example, pattern
    database_name: Optional[str] = None
    table_name: Optional[str] = None
    procedure_name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class SQLKnowledgeCreate(SQLKnowledgeBase):
    """Model for creating SQL knowledge"""
    pass


class SQLKnowledgeDocument(SQLKnowledgeBase):
    """Model for SQL knowledge stored in MongoDB"""
    id: str
    timestamp: int
    vector: List[float]


class SQLKnowledgeSearchResult(BaseModel):
    """Model for SQL knowledge search results"""
    id: str
    content: str
    knowledge_type: str
    similarity: float
    database_name: Optional[str] = None
    table_name: Optional[str] = None
    procedure_name: Optional[str] = None
    description: Optional[str] = None


# ============================================================================
# API Response Models
# ============================================================================

class StoreResponse(BaseModel):
    """Response for store operations"""
    success: bool
    document_id: str
    chunks: int = 1
    message: str


class DeleteResponse(BaseModel):
    """Response for delete operations"""
    success: bool
    message: str


class StatsResponse(BaseModel):
    """Response for statistics"""
    total_documents: int
    total_chunks: int
    departments: Dict[str, int]
    types: Dict[str, int]
    subjects: Dict[str, int]


class VectorSearchStatus(BaseModel):
    """Vector search status info"""
    enabled: bool
    native_available: bool
    status: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for health check"""
    status: str
    mongodb_connected: bool
    mongodb_version: Optional[str] = None
    embedding_model_loaded: bool
    vector_search: Optional[VectorSearchStatus] = None
    collections: Dict[str, int]


# ============================================================================
# SQL Extraction Models
# ============================================================================

class DatabaseConfigRequest(BaseModel):
    """Request model for database connection configuration"""
    name: str
    server: str
    database: str
    lookup_key: str
    user: str
    password: str
    port: int = 1433


class ExtractionRequest(BaseModel):
    """Request model for single database extraction"""
    config: DatabaseConfigRequest


class ExtractionFromConfigRequest(BaseModel):
    """Request model for config file extraction"""
    config_path: str
    only: Optional[str] = None  # Extract only specific database


class ExtractionStatsResponse(BaseModel):
    """Response model for extraction statistics"""
    database: str
    tables: int = 0
    procedures: int = 0
    errors: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Response model for extraction operations"""
    success: bool
    databases: List[ExtractionStatsResponse]
    total_stats: Dict[str, int]


class ExtractionJobResponse(BaseModel):
    """Response model for async extraction job"""
    success: bool
    job_id: str
    message: str


# ============================================================================
# RAG Pipeline Models
# ============================================================================

class PipelineRunRequest(BaseModel):
    """Request model for running RAG pipeline"""
    database: str
    llm_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    skip_extraction: bool = True
    skip_summarization: bool = False
    skip_embedding: bool = False


class PipelineSummarizeRequest(BaseModel):
    """Request model for running summarization stage"""
    database: str
    llm_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"


class PipelineEmbedRequest(BaseModel):
    """Request model for running embedding stage"""
    database: str


# ============================================================================
# Feedback System Models
# ============================================================================

class FeedbackType(str, Enum):
    """
    Types of feedback users can provide on RAG system responses.

    - rating: Simple thumbs up/down or star rating
    - correction: User provides corrected SQL or information
    - refinement: User refines their query with additional context
    - resolution: User marks the query as resolved/answered
    """
    RATING = "rating"
    CORRECTION = "correction"
    REFINEMENT = "refinement"
    RESOLUTION = "resolution"


class FeedbackRating(BaseModel):
    """
    Rating feedback - simple quality assessment.

    Supports both binary (thumbs up/down) and star ratings.
    Used for quick feedback on response quality.
    """
    is_helpful: bool = Field(..., description="Whether the response was helpful (thumbs up/down)")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Optional star rating (1-5)")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional comment explaining the rating")


class FeedbackCorrection(BaseModel):
    """
    Correction feedback - user provides the correct answer.

    Critical for improving RAG accuracy. When users provide corrections,
    we can update document quality scores and potentially add corrected
    examples to the few-shot learning collection.
    """
    original_response: str = Field(..., description="The original response that was incorrect")
    corrected_response: str = Field(..., description="The correct response provided by user")
    error_type: Optional[str] = Field(None, description="Type of error (e.g., 'schema_error', 'logic_error', 'incomplete')")
    affected_tables: Optional[List[str]] = Field(default_factory=list, description="Tables affected by the correction")
    comment: Optional[str] = Field(None, max_length=1000, description="Additional context about the correction")


class FeedbackRefinement(BaseModel):
    """
    Refinement feedback - user clarifies their original query.

    Helps understand user intent better and can improve query
    understanding for similar future queries.
    """
    original_query: str = Field(..., description="The original query that needed clarification")
    refined_query: str = Field(..., description="The refined/clarified query")
    additional_context: Optional[str] = Field(None, description="Additional context provided by user")


class FeedbackCreate(BaseModel):
    """
    Request model for creating feedback.

    Supports multiple feedback types with their specific payloads.
    Links feedback to the original query session for context.
    """
    feedback_type: FeedbackType = Field(..., description="Type of feedback being provided")

    # Query context - links feedback to the original query
    query_id: Optional[str] = Field(None, description="ID of the original query session (if available)")
    query: str = Field(..., description="The original query text")
    response: str = Field(..., description="The response that feedback is about")

    # Database context
    database: Optional[str] = Field(None, description="Database that was queried")

    # Type-specific feedback payloads (only one should be populated based on feedback_type)
    rating: Optional[FeedbackRating] = Field(None, description="Rating feedback details")
    correction: Optional[FeedbackCorrection] = Field(None, description="Correction feedback details")
    refinement: Optional[FeedbackRefinement] = Field(None, description="Refinement feedback details")

    # Document references - IDs of documents used in generating the response
    document_ids: List[str] = Field(default_factory=list, description="IDs of documents used in response")

    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier (for tracking user-specific patterns)")
    session_id: Optional[str] = Field(None, description="Session identifier (for tracking session context)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeedbackResponse(BaseModel):
    """Response model for feedback operations."""
    success: bool
    feedback_id: str
    message: str
    quality_scores_updated: int = Field(0, description="Number of document quality scores updated")


class FeedbackStatsResponse(BaseModel):
    """
    Aggregated feedback statistics.

    Provides insights into RAG system performance based on user feedback.
    """
    total_feedback: int = Field(..., description="Total number of feedback entries")

    # Breakdown by type
    by_type: Dict[str, int] = Field(default_factory=dict, description="Feedback count by type")

    # Rating statistics
    total_ratings: int = Field(0, description="Total number of rating feedbacks")
    helpful_count: int = Field(0, description="Number of helpful ratings (thumbs up)")
    not_helpful_count: int = Field(0, description="Number of not helpful ratings (thumbs down)")
    helpfulness_rate: float = Field(0.0, description="Percentage of helpful ratings (0.0-1.0)")
    average_rating: Optional[float] = Field(None, description="Average star rating (if star ratings used)")

    # Correction statistics
    total_corrections: int = Field(0, description="Total number of corrections submitted")
    corrections_by_error_type: Dict[str, int] = Field(default_factory=dict, description="Corrections by error type")

    # Database-specific stats
    by_database: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Feedback stats by database")

    # Time-based stats
    feedback_last_24h: int = Field(0, description="Feedback received in last 24 hours")
    feedback_last_7d: int = Field(0, description="Feedback received in last 7 days")

    # Period for the stats
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class LowPerformingDocument(BaseModel):
    """
    Model for documents with low quality scores based on feedback.

    Used to identify documents that need improvement or review.
    """
    document_id: str = Field(..., description="Document ID")
    document_type: str = Field(..., description="Type of document (schema, procedure, example)")
    title: Optional[str] = Field(None, description="Document title or name")
    database: Optional[str] = Field(None, description="Associated database")

    # Quality metrics
    quality_score: float = Field(..., description="Current quality score (0.0-1.0)")
    total_feedback: int = Field(..., description="Total feedback count for this document")
    negative_feedback: int = Field(..., description="Count of negative feedback")
    correction_count: int = Field(..., description="Number of corrections involving this document")

    # Recent feedback
    last_feedback_date: Optional[datetime] = Field(None, description="Date of most recent feedback")
    recent_issues: List[str] = Field(default_factory=list, description="Recent issues reported")


# ============================================================================
# Category Models (for document classification)
# ============================================================================

class CategoryType(str, Enum):
    """Valid category types for document classification."""
    DEPARTMENTS = "departments"
    TYPES = "types"
    SUBJECTS = "subjects"


class CategoryBase(BaseModel):
    """Base model for category entries."""
    id: str = Field(..., description="Unique category identifier (e.g., 'it', 'hr')")
    name: str = Field(..., description="Display name for the category")
    description: Optional[str] = Field(None, description="Description of the category")
    order: int = Field(0, description="Display order (lower = first)")

    model_config = {"extra": "ignore"}  # Ignore extra fields like created_at, updated_at


class CategoryCreate(CategoryBase):
    """Model for creating a new category."""
    pass


class CategoryUpdate(BaseModel):
    """Model for updating a category (all fields optional)."""
    name: Optional[str] = Field(None, description="Display name for the category")
    description: Optional[str] = Field(None, description="Description of the category")
    order: Optional[int] = Field(None, description="Display order (lower = first)")


class CategoryResponse(BaseModel):
    """Response model for category operations."""
    success: bool
    category: Optional[CategoryBase] = None
    message: Optional[str] = None


class CategoryListResponse(BaseModel):
    """Response model for listing categories."""
    success: bool
    categories: List[CategoryBase]


# ============================================================================
# Audio Analysis Models
# ============================================================================

class AudioMetadata(BaseModel):
    """Audio file metadata"""
    duration_seconds: float = Field(..., description="Duration of audio in seconds")
    sample_rate: int = Field(..., description="Audio sample rate in Hz")
    channels: int = Field(..., description="Number of audio channels")
    format: str = Field(..., description="Audio format (wav, mp3, etc.)")
    file_size_bytes: int = Field(..., description="File size in bytes")


class EmotionTimestamp(BaseModel):
    """Emotion detection timestamp"""
    emotion: str = Field(..., description="Detected emotion")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class EmotionResult(BaseModel):
    """Emotion analysis result"""
    primary: str = Field(..., description="Primary detected emotion")
    detected: List[str] = Field(default_factory=list, description="All detected emotions")
    timestamps: List[EmotionTimestamp] = Field(default_factory=list, description="Emotion timestamps")


class AudioEventTimestamp(BaseModel):
    """Audio event timestamp"""
    event: str = Field(..., description="Detected audio event")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class AudioEventResult(BaseModel):
    """Audio event detection result"""
    detected: List[str] = Field(default_factory=list, description="Detected audio events")
    timestamps: List[AudioEventTimestamp] = Field(default_factory=list, description="Event timestamps")


class SpeakerSegment(BaseModel):
    """A segment of speech from a specific speaker"""
    speaker: str = Field(..., description="Speaker identifier (e.g., 'Speaker 1')")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(default="", description="Transcribed text for this segment")


class SpeakerStatistics(BaseModel):
    """Speaking statistics for a single speaker"""
    total_duration: float = Field(..., description="Total speaking duration in seconds")
    segment_count: int = Field(..., description="Number of speaking segments")
    word_count: int = Field(default=0, description="Number of words spoken")
    percentage: float = Field(default=0, description="Percentage of total speaking time")


class SpeakerDiarizationResult(BaseModel):
    """Speaker diarization results"""
    enabled: bool = Field(default=False, description="Whether diarization was enabled/performed")
    segments: List[SpeakerSegment] = Field(default_factory=list, description="Speaker segments with timestamps")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Per-speaker statistics")
    num_speakers: int = Field(default=0, description="Number of unique speakers detected")


class CallMetadata(BaseModel):
    """
    Metadata parsed from RingCentral call recording filename.

    Pattern: yyyymmdd-hhmmss_EXT_PHONE_DIRECTION_AUTO_RECORDINGID.mp3
    Example: 20251209-123034_302_(843)858-0749_Outgoing_Auto_2265804682051.mp3
    """
    call_date: Optional[str] = Field(None, description="Parsed date (YYYY-MM-DD)")
    call_time: Optional[str] = Field(None, description="Parsed time (HH:MM:SS)")
    extension: Optional[str] = Field(None, description="Support staff extension")
    phone_number: Optional[str] = Field(None, description="Customer phone number")
    direction: Optional[str] = Field(None, description="Call direction (Incoming/Outgoing)")
    auto_flag: Optional[str] = Field(None, description="Auto indicator")
    recording_id: Optional[str] = Field(None, description="RingCentral Recording ID")
    parsed: bool = Field(False, description="Whether filename was successfully parsed")


class CallContentAnalysis(BaseModel):
    """
    LLM-analyzed content from call transcription.

    Extracted by analyzing the transcription with an LLM to detect:
    - Subject: What the call was about
    - Outcome: How the call ended (resolved, unresolved, etc.)
    - Customer Name: Name mentioned in the transcription
    """
    subject: Optional[str] = Field(None, description="LLM-detected call subject")
    outcome: Optional[str] = Field(None, description="LLM-detected outcome")
    customer_name: Optional[str] = Field(None, description="LLM-extracted customer name")
    confidence: float = Field(0.0, description="Confidence score (0-1)")
    analysis_model: str = Field("", description="Model used for analysis")


class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis"""
    language: str = Field(default="auto", description="Language code (en, zh, ja, etc.) or 'auto'")


class AudioAnalysisResponse(BaseModel):
    """Response model for audio analysis"""
    success: bool
    transcription: str = Field(..., description="Clean transcription text")
    transcription_summary: Optional[str] = Field(None, description="LLM-generated summary for long transcriptions (>2 min)")
    raw_transcription: str = Field(..., description="Raw transcription with tags")
    emotions: EmotionResult
    audio_events: AudioEventResult
    language: str = Field(..., description="Detected or specified language")
    audio_metadata: AudioMetadata
    call_metadata: Optional[CallMetadata] = Field(None, description="Parsed filename metadata (RingCentral)")
    call_content: Optional[CallContentAnalysis] = Field(None, description="LLM-analyzed call content")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


class AudioStoreRequest(BaseModel):
    """Request model for storing audio analysis"""
    customer_support_staff: str = Field(..., description="Customer support staff name or ID")
    ewr_customer: str = Field(..., description="EWR customer name or ID")
    mood: str = Field(..., description="Overall mood: Negative, Positive, or Neutral")
    outcome: str = Field(..., description="Call outcome")
    filename: str = Field(..., description="Original audio filename")
    transcription: str = Field(..., description="Clean transcription text")
    transcription_summary: Optional[str] = Field(None, description="Summary for long transcriptions (>2 min)")
    raw_transcription: str = Field(..., description="Raw transcription with tags")
    emotions: EmotionResult
    audio_events: AudioEventResult
    language: str = Field(..., description="Detected language")
    audio_metadata: AudioMetadata
    # New fields for call metadata (parsed from filename)
    call_metadata: Optional[CallMetadata] = Field(None, description="Parsed call metadata from filename")
    # New fields for LLM-analyzed content
    call_content: Optional[CallContentAnalysis] = Field(None, description="LLM-analyzed call content")
    # Related ticket IDs from EWRCentral (can be updated later)
    related_ticket_ids: List[int] = Field(default_factory=list, description="CentralTicketIDs that may be related to this call")
    # Primary linked ticket (single ticket assignment)
    linked_ticket_id: Optional[int] = Field(None, description="Primary CentralTicketID linked to this call")
    # Speaker diarization results
    speaker_diarization: Optional[SpeakerDiarizationResult] = Field(None, description="Speaker diarization results")
    # Optional pending filename to delete after successful save
    pending_filename: Optional[str] = Field(None, description="Pending JSON filename to delete after save")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_support_staff": "John Doe",
                "ewr_customer": "Jane Smith",
                "mood": "Positive",
                "outcome": "Issue Resolved",
                "filename": "call_2024_01_15.wav",
                "transcription": "Hello, how can I help you today?",
                "transcription_summary": None,
                "raw_transcription": "<|HAPPY|><|Speech|>Hello, how can I help you today?",
                "emotions": {
                    "primary": "HAPPY",
                    "detected": ["HAPPY"],
                    "timestamps": []
                },
                "audio_events": {
                    "detected": ["Speech"],
                    "timestamps": []
                },
                "language": "en",
                "audio_metadata": {
                    "duration_seconds": 45.5,
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "wav",
                    "file_size_bytes": 1440000
                }
            }
        }


class AudioSearchRequest(BaseModel):
    """Request model for searching audio analyses"""
    query: Optional[str] = Field(None, description="Search query text (optional)")
    customer_support_staff: Optional[str] = Field(None, description="Filter by customer support staff")
    ewr_customer: Optional[str] = Field(None, description="Filter by EWR customer")
    mood: Optional[str] = Field(None, description="Filter by mood")
    outcome: Optional[str] = Field(None, description="Filter by outcome")
    emotion: Optional[str] = Field(None, description="Filter by detected emotion")
    language: Optional[str] = Field(None, description="Filter by language")
    date_from: Optional[str] = Field(None, description="Filter by date range start (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="Filter by date range end (YYYY-MM-DD)")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")


class AudioAnalysisDocument(BaseModel):
    """Model for audio analysis stored in MongoDB"""
    id: str
    customer_support_staff: str
    ewr_customer: str
    mood: str
    outcome: str
    filename: str
    transcription: str
    transcription_summary: Optional[str] = None
    raw_transcription: str
    emotions: EmotionResult
    audio_events: AudioEventResult
    language: str
    audio_metadata: AudioMetadata
    embedding_text: str
    created_at: datetime
    updated_at: datetime
    analyzed_by: str
    analysis_version: str
    relevance_score: Optional[float] = None
    # Ticket linking fields
    related_ticket_ids: List[int] = Field(default_factory=list, description="Related ticket IDs")
    linked_ticket_id: Optional[int] = Field(None, description="Primary linked ticket ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AudioStatsResponse(BaseModel):
    """Response model for audio analysis statistics"""
    total_analyses: int = Field(..., description="Total number of audio analyses")
    by_mood: Dict[str, int] = Field(default_factory=dict, description="Count by mood")
    by_outcome: Dict[str, int] = Field(default_factory=dict, description="Count by outcome")
    by_language: Dict[str, int] = Field(default_factory=dict, description="Count by language")
    by_emotion: Dict[str, int] = Field(default_factory=dict, description="Count by primary emotion")
    total_duration_hours: float = Field(..., description="Total audio duration in hours")
    average_duration_seconds: float = Field(..., description="Average audio duration in seconds")


# ==================== Ticket Match History Models ====================

class TicketMatchCandidate(BaseModel):
    """A candidate ticket from the matching process"""
    ticket_id: int = Field(..., description="CentralTicketID")
    ticket_title: Optional[str] = Field(None, description="Ticket title")
    semantic_score: float = Field(..., description="Semantic similarity score (0-1)")
    combined_score: float = Field(..., description="Combined weighted score (0-1)")
    match_reasons: List[str] = Field(default_factory=list, description="Reasons for the match")
    phone_match: bool = Field(False, description="Whether phone number matched")
    staff_match: bool = Field(False, description="Whether staff member matched")
    time_proximity_minutes: Optional[int] = Field(None, description="Minutes between call and ticket creation")


class TicketMatchHistoryRecord(BaseModel):
    """Record of a ticket matching attempt for ML training"""
    id: str = Field(..., description="UUID for this match record")
    analysis_id: str = Field(..., description="Audio analysis ID")
    matched_at: datetime = Field(..., description="When match was attempted")
    match_method: str = Field(..., description="Method used: 'semantic', 'phone', 'combined'")
    search_text: str = Field(..., description="Text used for semantic search (summary or transcription)")
    search_text_type: str = Field(..., description="Type of text: 'summary' or 'transcription'")
    candidates: List[TicketMatchCandidate] = Field(default_factory=list, description="All candidates considered")
    selected_ticket_id: Optional[int] = Field(None, description="User's final selection")
    auto_linked: bool = Field(False, description="Was this auto-linked?")
    auto_link_confidence: Optional[float] = Field(None, description="Confidence score if auto-linked")
    user_override: bool = Field(False, description="Did user change auto-link selection?")
    feedback: Optional[str] = Field(None, description="User feedback on match quality")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SemanticMatchRequest(BaseModel):
    """Request for semantic ticket matching"""
    analysis_id: str = Field(..., description="Audio analysis ID to find matches for")
    use_summary: bool = Field(True, description="Use summary first, fall back to transcription")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum combined score threshold")
    auto_link_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Threshold for automatic linking")
    auto_link: bool = Field(False, description="Automatically link if confidence exceeds threshold")


class SemanticMatchResponse(BaseModel):
    """Response from semantic ticket matching"""
    success: bool
    analysis_id: str
    search_text_type: str = Field(..., description="What was searched: 'summary' or 'transcription'")
    total_candidates: int = Field(0, description="Number of tickets considered")
    matches: List[TicketMatchCandidate] = Field(default_factory=list, description="Matched tickets sorted by score")
    best_match: Optional[TicketMatchCandidate] = Field(None, description="Highest scoring match")
    auto_linked: bool = Field(False, description="Whether auto-linking occurred")
    linked_ticket_id: Optional[int] = Field(None, description="Ticket ID if linked")
    error: Optional[str] = Field(None, description="Error message if failed")


# ==================== Phone Customer Mapping Models ====================

class PhoneCustomerMapping(BaseModel):
    """
    Maps phone numbers to customer IDs for unknown number identification.

    Tracks phone numbers discovered through ticket linking and other sources
    to help identify customers calling from unregistered numbers.
    """
    phone_number: str = Field(..., description="Normalized phone number (E.164 format preferred)")
    customer_id: int = Field(..., description="EWR Customer ID")
    first_seen: datetime = Field(..., description="When this mapping was first discovered")
    last_seen: datetime = Field(..., description="When this mapping was last confirmed")
    occurrence_count: int = Field(1, description="Number of times this mapping has been seen")
    source: str = Field(..., description="How mapping was discovered (e.g., 'ticket_link', 'manual')")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PhoneCustomerMappingResponse(BaseModel):
    """Response for phone customer mapping operations"""
    success: bool
    phone_number: Optional[str] = None
    customer_id: Optional[int] = None
    mapping: Optional[PhoneCustomerMapping] = None
    message: Optional[str] = None


class PhoneCustomerMappingsListResponse(BaseModel):
    """Response for listing phone mappings"""
    success: bool
    mappings: List[PhoneCustomerMapping] = Field(default_factory=list)
    total: int = 0
