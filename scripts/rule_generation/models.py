"""
Data models for rule generation system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class ProcedureType(Enum):
    REPORTING = "reporting"  # Get*, Report*, Search* - highest NLQ relevance
    CRUD = "crud"            # Insert*, Update*, Delete* - reveals relationships
    BATCH = "batch"          # Batch*, Process*, Sync* - complex operations
    UTILITY = "utility"      # Helper procs, logging, etc.


class ComplexityTier(Enum):
    SIMPLE = "simple"        # Single table, basic SELECT
    AGGREGATE = "aggregate"  # GROUP BY, aggregations
    MULTI_STEP = "multi-step"  # Multiple operations, temp tables


class ActionType(Enum):
    RETRIEVE = "retrieve"
    AGGREGATE = "aggregate"
    SEARCH = "search"
    COMPARE = "compare"
    TREND = "trend"


class TemporalScope(Enum):
    POINT_IN_TIME = "point-in-time"
    RANGE = "range"
    HISTORICAL = "historical"
    CURRENT_STATE = "current-state"


class TableRole(Enum):
    FACT = "fact"           # Main transactional table
    DIMENSION = "dimension"  # Descriptive attributes
    LOOKUP = "lookup"       # Type/Status reference tables
    BRIDGE = "bridge"       # Many-to-many junction tables
    AUDIT = "audit"         # History/audit tables


class ColumnRole(Enum):
    IDENTIFIER = "identifier"
    MEASURE = "measure"
    DIMENSION = "dimension"
    TIMESTAMP = "timestamp"
    STATUS = "status"
    FLAG = "flag"


@dataclass
class ImplicitFilter:
    """A filter that should always be applied but isn't explicitly asked for."""
    column: str
    operator: str  # '=', 'IS NULL', 'IS NOT NULL', '>', '<', etc.
    value: Any
    frequency: float  # How often this filter appears (0.0 - 1.0)
    description: str  # Natural language description


@dataclass
class TableSemantics:
    """Semantic information about a table."""
    name: str
    role: TableRole
    grain: str  # e.g., "per_ticket", "per_user", "per_day"
    primary_key: Optional[str] = None
    common_joins: List[str] = field(default_factory=list)
    implicit_filters: List[ImplicitFilter] = field(default_factory=list)


@dataclass
class ColumnSemantics:
    """Semantic information about a column."""
    name: str
    role: ColumnRole
    display_name: str  # Natural language name
    calculation: Optional[str] = None  # If derived
    is_temporal: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None


@dataclass
class JoinTemplate:
    """A reusable JOIN pattern."""
    source_table: str
    target_table: str
    join_type: str  # INNER, LEFT, RIGHT, etc.
    condition: str
    purpose: str  # e.g., "lookup ticket type", "get creator name"


@dataclass
class AggregationPattern:
    """A pattern for aggregation queries."""
    trigger_phrases: List[str]  # e.g., ["how many", "count of"]
    sql_function: str  # COUNT, SUM, AVG, etc.
    typical_group_by: List[str]  # Common grouping columns


@dataclass
class TemporalPattern:
    """A pattern for date/time filtering."""
    natural_language: str  # e.g., "last year"
    sql_template: str  # e.g., "YEAR({col}) = YEAR(GETDATE()) - 1"
    column_placeholder: str = "{col}"


@dataclass
class ProcedureAnalysis:
    """Complete semantic analysis of a stored procedure."""
    procedure_name: str
    database: str

    # Classification
    procedure_type: ProcedureType
    complexity_tier: ComplexityTier
    nlq_relevance: float  # 0.0 - 1.0

    # Intent Signature
    action_type: ActionType
    entity_focus: str  # Primary noun (Ticket, User, PTO)
    temporal_scope: TemporalScope

    # Tables
    tables_used: List[TableSemantics] = field(default_factory=list)
    primary_table: Optional[str] = None

    # Columns
    columns: List[ColumnSemantics] = field(default_factory=list)

    # Patterns
    join_templates: List[JoinTemplate] = field(default_factory=list)
    aggregation_patterns: List[AggregationPattern] = field(default_factory=list)
    temporal_patterns: List[TemporalPattern] = field(default_factory=list)
    implicit_filters: List[ImplicitFilter] = field(default_factory=list)

    # Parameters
    parameter_mappings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Keywords for triggering
    trigger_keywords: List[str] = field(default_factory=list)

    # Example queries this proc can answer
    example_questions: List[str] = field(default_factory=list)


@dataclass
class GeneratedRule:
    """A rule generated from stored procedure analysis."""
    rule_id: str
    database: str
    rule_type: str  # "assistance" | "constraint" | "example"
    priority: str  # "high" | "normal" | "low"

    # Rule content
    description: str
    rule_text: str

    # Triggers
    trigger_keywords: List[str] = field(default_factory=list)
    trigger_tables: List[str] = field(default_factory=list)
    trigger_columns: List[str] = field(default_factory=list)

    # Auto-fix patterns
    auto_fix: Optional[Dict[str, str]] = None

    # Example (for exact-match rules)
    example_question: Optional[str] = None
    example_sql: Optional[str] = None

    # Metadata
    source_procedure: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_mongodb_doc(self) -> Dict:
        """Convert to MongoDB document format."""
        doc = {
            "rule_id": self.rule_id,
            "database": self.database,
            "type": self.rule_type,
            "priority": self.priority,
            "description": self.description,
            "rule_text": self.rule_text,
            "trigger_keywords": self.trigger_keywords,
            "trigger_tables": self.trigger_tables,
            "trigger_columns": self.trigger_columns,
            "is_active": True,
            "source": f"generated_from_{self.source_procedure}",
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.created_at,
            "version": 1,
        }

        if self.auto_fix:
            doc["auto_fix"] = self.auto_fix

        if self.example_question and self.example_sql:
            doc["example"] = {
                "question": self.example_question,
                "sql": self.example_sql
            }

        return doc


@dataclass
class GeneratedExample:
    """An example (question + SQL) generated from stored procedure."""
    example_id: str
    database: str
    question: str
    sql: str

    # Metadata
    source_procedure: str
    tables_used: List[str] = field(default_factory=list)
    complexity: str = "simple"
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_mongodb_doc(self) -> Dict:
        """Convert to MongoDB document format."""
        return {
            "example_id": self.example_id,
            "database": self.database,
            "question": self.question,
            "sql": self.sql,
            "tables_used": self.tables_used,
            "complexity": self.complexity,
            "source": f"generated_from_{self.source_procedure}",
            "confidence": self.confidence,
            "is_active": True,
            "created_at": self.created_at,
        }


@dataclass
class ValidationResult:
    """Result of validating a rule or example."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of testing a rule via the API."""
    rule_id: str
    test_question: str
    success: bool
    generated_sql: Optional[str] = None
    execution_result: Optional[Dict] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0
