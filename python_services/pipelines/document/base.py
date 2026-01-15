"""
Base Pipeline Classes - Foundation for document retrieval pipeline steps

This module provides the abstract base class and data structures used by all
pipeline steps in the document retrieval workflow. The design follows these
principles:

1. **Immutable Context**: Each step receives a context object that contains
   all accumulated state from previous steps. Steps should not mutate this
   context but instead return results that the orchestrator merges.

2. **Explicit Dependencies**: Each step declares what it requires from context
   and what it produces. This enables validation and documentation.

3. **Async-First**: All I/O operations use async/await for efficient concurrency.

4. **Error Isolation**: Each step handles its own errors and returns structured
   results that indicate success/failure without raising exceptions to callers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging
import uuid


class QueryType(str, Enum):
    """
    Classification of query complexity and intent.

    Used by QueryUnderstandingStep to route queries appropriately:
    - SIMPLE: Direct lookups, can often use cached responses
    - FACTUAL: Requires retrieval of specific facts from documents
    - ANALYTICAL: Needs synthesis across multiple documents
    - TEMPORAL: Time-sensitive queries requiring recent information
    """
    SIMPLE = "simple"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"


@dataclass
class StepResult:
    """
    Result returned by a pipeline step.

    This structure provides a consistent interface for step outcomes:
    - success: Whether the step completed without critical errors
    - data: The primary output data from the step
    - metadata: Timing, debugging, and instrumentation data
    - errors: List of error messages if any occurred
    - warnings: Non-fatal issues encountered during processing

    Design Rationale:
    Using a structured result rather than exceptions allows the pipeline
    orchestrator to make intelligent decisions about error recovery,
    retry logic, and degraded operation modes.
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure metadata has required timing fields."""
        if "step_id" not in self.metadata:
            self.metadata["step_id"] = str(uuid.uuid4())[:8]
        if "completed_at" not in self.metadata:
            self.metadata["completed_at"] = datetime.utcnow().isoformat()


@dataclass
class RetrievedDocument:
    """
    A document chunk retrieved from the knowledge base.

    Attributes:
        id: Unique identifier for this chunk
        parent_id: ID of the parent document (for multi-chunk docs)
        content: The actual text content
        title: Document or section title
        score: Retrieval relevance score (0-1)
        source: Origin indicator (vector, bm25, hybrid)
        metadata: Additional document properties
        grading_score: Score assigned by document grading step (0-1)
        grading_reasoning: Explanation for the grading score
    """
    id: str
    parent_id: str
    content: str
    title: str = ""
    score: float = 0.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    grading_score: Optional[float] = None
    grading_reasoning: Optional[str] = None

    @property
    def is_relevant(self) -> bool:
        """Check if document passed grading threshold."""
        if self.grading_score is None:
            return True  # Not graded yet, assume relevant
        return self.grading_score >= 0.5


@dataclass
class PipelineContext:
    """
    Shared context passed through the pipeline.

    This immutable-by-convention context accumulates state as each step
    processes the query. The orchestrator is responsible for merging
    step results into this context.

    Design Rationale:
    A shared context object allows steps to access information from
    previous steps without tight coupling. Each step only reads what
    it needs and writes its outputs, which the orchestrator merges.

    Attributes:
        query_id: Unique identifier for this query session
        original_query: The user's original input
        timestamp: When the query was received

        # Query understanding outputs
        query_type: Classification of query complexity
        rewritten_query: Optimized query for retrieval
        expanded_queries: List of query variants for multi-query retrieval
        extracted_entities: Named entities found in query

        # Retrieval outputs
        retrieved_documents: Documents from hybrid retrieval
        retrieval_method: Which retrieval path was used

        # Grading outputs
        graded_documents: Documents after relevance grading
        average_relevance: Mean relevance score of graded docs

        # User context
        user_id: Optional user identifier for personalization
        session_history: Previous queries in this session
        filters: User-specified filters (department, type, etc.)
    """
    # Query identification
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Query understanding outputs
    query_type: QueryType = QueryType.FACTUAL
    rewritten_query: str = ""
    expanded_queries: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)

    # Retrieval outputs
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    retrieval_method: str = "hybrid"
    vector_results_count: int = 0
    bm25_results_count: int = 0

    # Grading outputs
    graded_documents: List[RetrievedDocument] = field(default_factory=list)
    average_relevance: float = 0.0
    documents_filtered_count: int = 0

    # User context
    user_id: Optional[str] = None
    session_history: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)

    # Pipeline control
    skip_grading: bool = False  # Skip grading for high-confidence retrievals
    require_web_fallback: bool = False  # Flag for corrective retrieval

    def merge_result(self, result: StepResult) -> 'PipelineContext':
        """
        Create a new context with step results merged in.

        This method creates a shallow copy of the context and updates
        fields based on the step result data. Steps should include
        field names in their result data that match context attributes.

        Args:
            result: The StepResult to merge

        Returns:
            A new PipelineContext with merged data
        """
        # Create a copy of all current fields
        new_context = PipelineContext(
            query_id=self.query_id,
            original_query=self.original_query,
            timestamp=self.timestamp,
            query_type=self.query_type,
            rewritten_query=self.rewritten_query,
            expanded_queries=list(self.expanded_queries),
            extracted_entities=dict(self.extracted_entities),
            retrieved_documents=list(self.retrieved_documents),
            retrieval_method=self.retrieval_method,
            vector_results_count=self.vector_results_count,
            bm25_results_count=self.bm25_results_count,
            graded_documents=list(self.graded_documents),
            average_relevance=self.average_relevance,
            documents_filtered_count=self.documents_filtered_count,
            user_id=self.user_id,
            session_history=list(self.session_history),
            filters=dict(self.filters),
            skip_grading=self.skip_grading,
            require_web_fallback=self.require_web_fallback,
        )

        # Merge in result data
        for key, value in result.data.items():
            if hasattr(new_context, key):
                setattr(new_context, key, value)

        return new_context


class PipelineStep(ABC):
    """
    Abstract base class for all pipeline steps.

    Each step in the document retrieval pipeline inherits from this class
    and implements the execute() method. Steps are designed to be:

    1. **Stateless**: All state flows through PipelineContext
    2. **Composable**: Steps can be reordered or replaced
    3. **Observable**: Logging and metrics are built-in
    4. **Resilient**: Errors are captured, not propagated

    Subclasses must implement:
    - name: Human-readable step name
    - requires: Set of context fields this step reads
    - produces: Set of context fields this step writes
    - execute(): The actual step logic
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the pipeline step.

        Args:
            logger: Optional logger instance. If not provided, creates one
                   using the step's class name.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this step."""
        pass

    @property
    @abstractmethod
    def requires(self) -> Set[str]:
        """
        Context fields this step requires as input.

        The orchestrator validates these fields exist before executing.
        """
        pass

    @property
    @abstractmethod
    def produces(self) -> Set[str]:
        """
        Context fields this step produces as output.

        Used for documentation and pipeline validation.
        """
        pass

    @abstractmethod
    async def execute(self, context: PipelineContext) -> StepResult:
        """
        Execute this step with the given context.

        This method should:
        1. Read required data from context
        2. Perform the step's logic (retrieval, grading, etc.)
        3. Return a StepResult with outputs in the data dict

        Args:
            context: The current pipeline context

        Returns:
            StepResult with success status and output data
        """
        pass

    def validate_context(self, context: PipelineContext) -> List[str]:
        """
        Validate that required context fields are present.

        Args:
            context: The context to validate

        Returns:
            List of missing field names (empty if valid)
        """
        missing = []
        for field_name in self.requires:
            if not hasattr(context, field_name):
                missing.append(field_name)
            elif getattr(context, field_name) is None:
                # Allow None for optional fields, but log a warning
                self.logger.debug(f"Field '{field_name}' is None")
        return missing

    async def run(self, context: PipelineContext) -> StepResult:
        """
        Run this step with validation and error handling.

        This is the public entry point that:
        1. Validates required context fields
        2. Records timing metadata
        3. Executes the step
        4. Catches and wraps any exceptions

        Args:
            context: The current pipeline context

        Returns:
            StepResult with success status and output data
        """
        start_time = datetime.utcnow()

        # Validate context
        missing = self.validate_context(context)
        if missing:
            return StepResult(
                success=False,
                errors=[f"Missing required context fields: {missing}"],
                metadata={
                    "step_name": self.name,
                    "started_at": start_time.isoformat(),
                    "duration_ms": 0,
                }
            )

        try:
            self.logger.debug(f"Executing step: {self.name}")
            result = await self.execute(context)

            # Add timing metadata
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            result.metadata["step_name"] = self.name
            result.metadata["started_at"] = start_time.isoformat()
            result.metadata["duration_ms"] = round(duration_ms, 2)

            self.logger.debug(
                f"Step {self.name} completed in {duration_ms:.2f}ms "
                f"(success={result.success})"
            )

            return result

        except Exception as e:
            self.logger.exception(f"Step {self.name} failed with exception")
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            return StepResult(
                success=False,
                errors=[f"Exception in {self.name}: {str(e)}"],
                metadata={
                    "step_name": self.name,
                    "started_at": start_time.isoformat(),
                    "duration_ms": round(duration_ms, 2),
                    "exception_type": type(e).__name__,
                }
            )
