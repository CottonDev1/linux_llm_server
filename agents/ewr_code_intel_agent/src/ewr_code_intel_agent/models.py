"""
EWR Code Intelligence Agent Models
==================================

Pydantic models for deep code analysis and knowledge management.
"""

from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisStatus(str, Enum):
    """Status of analysis."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class EntryPointType(str, Enum):
    """Types of code entry points."""
    API_ENDPOINT = "api_endpoint"
    UI_HANDLER = "ui_handler"
    CLI_COMMAND = "cli_command"
    SCHEDULED_TASK = "scheduled_task"
    EVENT_HANDLER = "event_handler"
    MESSAGE_HANDLER = "message_handler"
    MAIN_FUNCTION = "main_function"
    TEST = "test"
    OTHER = "other"


class CallNode(BaseModel):
    """A node in a call graph."""
    id: str
    name: str
    full_name: str  # Fully qualified name
    file_path: str
    line_number: int
    node_type: str  # function, method, constructor, etc.
    class_name: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_external: bool = False  # External library call
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CallEdge(BaseModel):
    """An edge in a call graph."""
    caller_id: str
    callee_id: str
    call_type: str = "direct"  # direct, indirect, callback, async
    line_number: int = 0
    arguments: List[str] = Field(default_factory=list)


class CallGraph(BaseModel):
    """Complete call graph for a codebase."""
    root_id: Optional[str] = None
    nodes: Dict[str, CallNode] = Field(default_factory=dict)
    edges: List[CallEdge] = Field(default_factory=list)
    depth: int = 0
    total_calls: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_callees(self, node_id: str) -> List[str]:
        """Get all nodes called by the given node."""
        return [e.callee_id for e in self.edges if e.caller_id == node_id]

    def get_callers(self, node_id: str) -> List[str]:
        """Get all nodes that call the given node."""
        return [e.caller_id for e in self.edges if e.callee_id == node_id]


class DataFlowPoint(BaseModel):
    """A point in data flow analysis."""
    variable_name: str
    file_path: str
    line_number: int
    operation: str  # read, write, transform, return, parameter
    value_type: Optional[str] = None
    source: Optional[str] = None  # Where the value came from
    destinations: List[str] = Field(default_factory=list)


class DataFlowTrace(BaseModel):
    """Trace of how data flows through the application."""
    variable_name: str
    start_point: DataFlowPoint
    flow_points: List[DataFlowPoint] = Field(default_factory=list)
    end_points: List[DataFlowPoint] = Field(default_factory=list)
    crosses_boundaries: bool = False  # Crosses module/class boundaries
    affected_functions: List[str] = Field(default_factory=list)
    security_concerns: List[str] = Field(default_factory=list)


class EntryPoint(BaseModel):
    """An entry point into the application."""
    id: str
    name: str
    entry_type: EntryPointType
    file_path: str
    line_number: int
    route: Optional[str] = None  # API route, URL pattern
    http_method: Optional[str] = None
    parameters: List[Dict[str, str]] = Field(default_factory=list)
    description: Optional[str] = None
    calls: List[str] = Field(default_factory=list)  # Functions called from here


class DependencyInfo(BaseModel):
    """Information about a dependency."""
    name: str
    version: Optional[str] = None
    source: str  # internal, external, builtin
    import_count: int = 0
    dependents: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Complete analysis result for a repository."""
    repo_path: str
    repo_name: str
    status: AnalysisStatus = AnalysisStatus.PENDING
    total_files: int = 0
    total_lines: int = 0
    languages: Dict[str, int] = Field(default_factory=dict)
    entry_points: List[EntryPoint] = Field(default_factory=list)
    call_graphs: Dict[str, CallGraph] = Field(default_factory=dict)
    dependencies: List[DependencyInfo] = Field(default_factory=list)
    key_classes: List[str] = Field(default_factory=list)
    key_functions: List[str] = Field(default_factory=list)
    architecture_notes: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStep(BaseModel):
    """A step in a workflow explanation."""
    step_number: int
    description: str
    file_path: str
    function_name: str
    line_number: int
    code_snippet: Optional[str] = None
    notes: Optional[str] = None


class WorkflowExplanation(BaseModel):
    """Explanation of how a workflow works."""
    question: str
    summary: str
    steps: List[WorkflowStep] = Field(default_factory=list)
    entry_points_used: List[str] = Field(default_factory=list)
    key_components: List[str] = Field(default_factory=list)
    data_flow: Optional[str] = None
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeGap(BaseModel):
    """A gap in the agent's knowledge."""
    question: str
    expected_topic: str
    actual_answer: Optional[str] = None
    missing_context: List[str] = Field(default_factory=list)
    suggested_files: List[str] = Field(default_factory=list)
    severity: str = "medium"  # low, medium, high


class ValidationResult(BaseModel):
    """Result from self-validation."""
    total_questions: int
    correct_answers: int
    accuracy_score: float = 0.0
    gaps: List[KnowledgeGap] = Field(default_factory=list)
    needs_refinement: bool = False
    suggestions: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class DeveloperAnswer(BaseModel):
    """Answer to a developer question."""
    question: str
    answer: str
    confidence: float = 0.0
    relevant_files: List[str] = Field(default_factory=list)
    code_examples: List[Dict[str, str]] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)
    workflow: Optional[WorkflowExplanation] = None


class CodeChunk(BaseModel):
    """A chunk of code for embedding."""
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, module, block
    language: str
    context: str = ""  # Additional context about the chunk
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    """Result of embedding code chunks."""
    chunk_id: str
    embedding: List[float] = Field(default_factory=list)
    model: str
    dimensions: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RefinementRequest(BaseModel):
    """Request to refine embeddings based on gaps."""
    gaps: List[KnowledgeGap]
    files_to_reprocess: List[str] = Field(default_factory=list)
    chunk_strategy: str = "default"  # default, fine, coarse
    max_chunk_size: int = 500


class RefinementResult(BaseModel):
    """Result of embedding refinement."""
    chunks_processed: int
    chunks_added: int
    chunks_updated: int
    new_accuracy_score: float = 0.0
    remaining_gaps: List[KnowledgeGap] = Field(default_factory=list)
