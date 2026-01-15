"""
EWR Agent Core Models
=====================

Pydantic models for agent communication, status tracking, and task management.
These models are the foundation for all agent interactions.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Agent Identity Enums
# =============================================================================

class AgentType(str, Enum):
    """Types of specialized agents in the system."""
    TASK = "task"           # Shell/PowerShell/workflow operations
    CODE = "code"           # Code analysis, generation, review
    DOCUMENT = "document"   # Document processing, search, summarization
    SQL = "sql"             # SQL generation, validation, execution
    PLANNER = "planner"     # Query planning and decomposition (Phase 2)
    CORRECTION = "correction"  # SQL error correction (Phase 2 MAGIC pattern)
    TEST = "test"           # Playwright test generation and execution
    AUDIO = "audio"         # Audio transcription, diarization, analysis
    CUSTOM = "custom"       # User-defined custom agents


class AgentState(str, Enum):
    """Agent operational states."""
    IDLE = "idle"           # Available for new tasks
    BUSY = "busy"           # Currently processing a task
    PAUSED = "paused"       # Temporarily suspended
    OFFLINE = "offline"     # Not running
    ERROR = "error"         # In error state
    STARTING = "starting"   # Initializing
    STOPPING = "stopping"   # Shutting down


class AgentCapability(str, Enum):
    """Standard capabilities that agents can advertise."""
    # Task Agent capabilities
    SHELL_EXEC = "shell_exec"
    POWERSHELL_EXEC = "powershell_exec"
    GIT_OPS = "git_ops"
    WORKFLOW = "workflow"

    # Code Agent capabilities
    CODE_ANALYZE = "code_analyze"
    CODE_GENERATE = "code_generate"
    CODE_REVIEW = "code_review"
    CODE_REFACTOR = "code_refactor"
    CODE_EXPLAIN = "code_explain"
    PROJECT_SCAN = "project_scan"

    # Document Agent capabilities
    DOC_SEARCH = "doc_search"
    DOC_SUMMARIZE = "doc_summarize"
    DOC_FORMAT = "doc_format"
    QUERY_EXPAND = "query_expand"
    MULTI_DOC = "multi_doc"

    # SQL Agent capabilities
    SQL_GENERATE = "sql_generate"
    SQL_VALIDATE = "sql_validate"
    SQL_EXECUTE = "sql_execute"
    SQL_EXPLAIN = "sql_explain"
    SCHEMA_EXPLORE = "schema_explore"
    TRAINING_DATA = "training_data"
    # Phase 3: SQL Agent upgrades
    SQL_DECOMPOSE = "sql_decompose"        # Break complex queries into sub-queries
    SQL_OPTIMIZE = "sql_optimize"          # Suggest query optimizations
    CONFIDENCE_SCORE = "confidence_score"  # Return confidence with generated SQL
    ABSTENTION = "abstention"              # Return "I don't know" when low confidence
    CACHE_INVALIDATE = "cache_invalidate"  # Invalidate schema cache

    # Phase 3: Code Agent upgrades
    SQL_PERFORMANCE = "sql_performance"    # Analyze SQL for performance issues

    # Phase 3: Task Agent upgrades
    SCHEMA_CHANGE = "schema_change"        # Detect schema changes via git

    # Phase 2: Planner Agent capabilities (OraPlan-SQL pattern)
    QUERY_DECOMPOSE = "query_decompose"    # Decompose complex NL queries into steps
    PLAN_GENERATE = "plan_generate"        # Generate execution plans for queries

    # Phase 2: Correction Agent capabilities (MAGIC pattern)
    SQL_CORRECT = "sql_correct"            # Fix SQL based on error feedback
    ERROR_ANALYZE = "error_analyze"        # Categorize and analyze SQL errors
    FEEDBACK_GENERATE = "feedback_generate"  # Create prevention guidelines

    # Test Agent capabilities (Playwright/E2E testing)
    TEST_GENERATE = "test_generate"        # Generate Playwright test scripts
    TEST_EXECUTE = "test_execute"          # Execute Playwright tests
    TEST_FIX = "test_fix"                  # Fix failing test scripts
    SERVICE_MANAGE = "service_manage"      # Start/stop/check web services
    PIPELINE_ANALYZE = "pipeline_analyze"  # Analyze pipeline for test generation
    ERROR_RECORD = "error_record"          # Record webstack errors

    # Audio Agent capabilities
    AUDIO_TRANSCRIBE = "audio_transcribe"    # Transcribe audio to text
    AUDIO_DIARIZE = "audio_diarize"          # Speaker diarization
    AUDIO_ANALYZE = "audio_analyze"          # Full audio analysis pipeline
    AUDIO_SUMMARIZE = "audio_summarize"      # Summarize transcription
    AUDIO_EMOTION = "audio_emotion"          # Detect emotions in audio
    AUDIO_CHUNK = "audio_chunk"              # Chunk audio for processing
    AUDIO_QA = "audio_qa"                    # Quality assurance for transcription
    AUDIO_CONSENSUS = "audio_consensus"      # Multi-model consensus validation


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"           # Request for help/delegation
    RESPONSE = "response"         # Response to a request
    NOTIFICATION = "notification" # One-way notification
    HEARTBEAT = "heartbeat"       # Agent health check
    STATUS_UPDATE = "status"      # Agent status change
    BROADCAST = "broadcast"       # Message to all agents


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


# =============================================================================
# Agent Information Models
# =============================================================================

class AgentInfo(BaseModel):
    """Information about a registered agent."""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType
    name: str
    version: str = "1.0.0"
    description: str = ""
    state: AgentState = AgentState.IDLE
    capabilities: List[AgentCapability] = Field(default_factory=list)
    custom_capabilities: List[str] = Field(default_factory=list)
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def all_capabilities(self) -> List[str]:
        """Get all capabilities as strings."""
        return [c.value for c in self.capabilities] + self.custom_capabilities

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.all_capabilities


# =============================================================================
# Message Models
# =============================================================================

class AgentMessage(BaseModel):
    """Base message for inter-agent communication."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None  # None = broadcast
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For request-response pairing
    ttl_seconds: int = 300  # Time-to-live
    payload: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl_seconds


class DelegationRequest(BaseModel):
    """Request to delegate a task to another agent."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent_id: str
    target_agent_type: Optional[AgentType] = None  # None = any capable agent
    target_agent_id: Optional[str] = None  # Specific agent
    required_capability: str  # Capability needed
    task_description: str
    task_params: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timeout_seconds: int = 300
    callback_required: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DelegationResponse(BaseModel):
    """Response to a delegation request."""
    request_id: str
    responder_agent_id: str
    accepted: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Task Models
# =============================================================================

class TaskResult(BaseModel):
    """Result from task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    duration_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED and self.error is None


# =============================================================================
# Configuration Models
# =============================================================================

class LLMConfig(BaseModel):
    """Configuration for LLM backend."""
    backend: str = "llamacpp"  # llamacpp, openai, anthropic
    model: str = "qwen2.5-coder:1.5b"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 60


class AgentConfigModel(BaseModel):
    """Configuration for an agent instance."""
    agent_type: AgentType
    name: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    capabilities: List[str] = Field(default_factory=list)
    auto_register: bool = True
    heartbeat_interval: int = 10  # seconds
    max_concurrent_tasks: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
