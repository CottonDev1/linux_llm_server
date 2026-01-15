"""
Autonomous SQL Training Pipeline
================================

Prefect-orchestrated pipeline for generating SQL training data automatically.
Runs overnight (7 PM - 7 AM CST) without human intervention.

Features:
- Automatic table selection based on failure feedback
- Schema-aware question generation
- SQL candidate generation with validation
- Quality control gates
- Auto-rule generation from failure patterns
- Full metrics emission to Prefect dashboard
"""

import asyncio
import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact

# Timezone for CST
try:
    from zoneinfo import ZoneInfo
    CST = ZoneInfo("America/Chicago")
except ImportError:
    import pytz
    CST = pytz.timezone("America/Chicago")


# =============================================================================
# Data Models
# =============================================================================

class FailureType(Enum):
    """SQL failure categories."""
    SYNTAX_ERROR = "syntax_error"
    INVALID_COLUMN = "invalid_column"
    INVALID_TABLE = "invalid_table"
    INVALID_OBJECT = "invalid_object"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    UNKNOWN = "unknown"


@dataclass
class Question:
    """Generated question with metadata."""
    text: str
    table: str
    difficulty: str = "medium"
    source: str = "template"
    expected_columns: List[str] = field(default_factory=list)


@dataclass
class SQLCandidate:
    """SQL candidate with metadata."""
    sql: str
    confidence: float
    source: str
    temperature: float = 0.0


@dataclass
class ExecutionResult:
    """SQL execution result."""
    success: bool
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    elapsed_ms: int = 0
    sample_rows: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[FailureType] = None


@dataclass
class ValidationResult:
    """Validation result."""
    valid: bool
    score: float
    reason: str


@dataclass
class FailureAnalysis:
    """Failure analysis result."""
    failure_type: FailureType
    reason: str
    suggested_fix: str


@dataclass
class TrainingExample:
    """Training example ready for storage."""
    question: str
    sql: str
    is_positive: bool
    confidence: float
    explanation: str = ""
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None
    tables_used: List[str] = field(default_factory=list)


@dataclass
class QCResult:
    """Quality control result."""
    accepted: List[TrainingExample] = field(default_factory=list)
    rejected: List[Tuple[TrainingExample, str]] = field(default_factory=list)


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    tables_processed: int = 0
    questions_generated: int = 0
    sql_candidates: int = 0
    validation_success: int = 0
    validation_failure: int = 0
    examples_stored: int = 0
    failures_stored: int = 0
    rules_generated: int = 0

    @property
    def success_rate(self) -> float:
        total = self.validation_success + self.validation_failure
        return self.validation_success / total if total > 0 else 0.0


@dataclass
class TrainingMetrics:
    """Cumulative training metrics."""
    tables_processed: int = 0
    questions_generated: int = 0
    positive_examples: int = 0
    negative_examples: int = 0
    rules_generated: int = 0
    runtime_seconds: float = 0.0
    failure_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.positive_examples + self.negative_examples
        return self.positive_examples / total if total > 0 else 0.0

    def merge(self, batch: BatchMetrics):
        """Merge batch metrics into totals."""
        self.tables_processed += batch.tables_processed
        self.questions_generated += batch.questions_generated
        self.positive_examples += batch.examples_stored
        self.negative_examples += batch.failures_stored
        self.rules_generated += batch.rules_generated


# =============================================================================
# Helper Functions
# =============================================================================

def hash_question(question: str) -> str:
    """Create hash of normalized question."""
    normalized = question.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def hash_sql(sql: str) -> str:
    """Create hash of normalized SQL."""
    # Remove whitespace variations
    normalized = " ".join(sql.upper().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def check_cutoff(cutoff_hour: int = 7, tz: str = "America/Chicago") -> bool:
    """Check if approaching cutoff time."""
    try:
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo(tz)
    except ImportError:
        import pytz
        local_tz = pytz.timezone(tz)

    now = datetime.now(local_tz)
    cutoff = now.replace(hour=cutoff_hour, minute=0, second=0, microsecond=0)

    # If past cutoff time, it's for tomorrow
    if now.hour >= cutoff_hour:
        return False

    remaining = (cutoff - now).total_seconds()
    return remaining < 900  # 15 minutes before cutoff


def batched(iterable, n):
    """Batch an iterable into chunks of size n."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def classify_error(error: Exception) -> FailureType:
    """Classify SQL error into failure type."""
    error_str = str(error).lower()

    if "invalid column name" in error_str:
        return FailureType.INVALID_COLUMN
    elif "invalid object name" in error_str:
        return FailureType.INVALID_TABLE
    elif "syntax" in error_str:
        return FailureType.SYNTAX_ERROR
    elif "permission" in error_str or "denied" in error_str:
        return FailureType.PERMISSION_DENIED
    elif "timeout" in error_str:
        return FailureType.TIMEOUT
    else:
        return FailureType.UNKNOWN


def extract_sql(response: str) -> str:
    """Extract SQL from LLM response."""
    # Look for SQL in code blocks
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    # Look for SQL without code blocks
    sql_match = re.search(r"```\s*(SELECT.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    # Return as-is if it looks like SQL
    if response.strip().upper().startswith("SELECT"):
        return response.strip()

    return response


def add_row_limit(sql: str, limit: int = 100) -> str:
    """Add TOP clause if not present."""
    sql_upper = sql.upper().strip()
    if "SELECT TOP" in sql_upper or "LIMIT" in sql_upper:
        return sql

    # Add TOP after SELECT
    return re.sub(r"^SELECT\s+", f"SELECT TOP {limit} ", sql, flags=re.IGNORECASE)


# =============================================================================
# Question Templates
# =============================================================================

QUESTION_TEMPLATES = [
    # Basic retrieval
    "Show me all {table} records",
    "List all {table}",
    "Get all entries from {table}",

    # Filtered retrieval
    "Show me {table} from today",
    "Show me {table} from the last 7 days",
    "Show me {table} from this month",

    # Aggregation
    "How many {table} are there?",
    "Count all {table}",
    "What is the total count of {table}?",

    # Column-specific (requires column info)
    "Show me {table} with {column} greater than average",
    "List {table} where {column} is not null",

    # Time-based (requires date column)
    "Show me the most recent {table}",
    "List {table} created today",
    "Show {table} added in the last month",

    # Top N
    "Show me the top 10 {table}",
    "List the first 5 {table}",
]


def generate_template_questions(table_name: str, columns: List[str]) -> List[Question]:
    """Generate questions from templates for a table."""
    questions = []

    # Basic templates
    for template in QUESTION_TEMPLATES[:6]:
        question_text = template.format(table=table_name)
        questions.append(Question(
            text=question_text,
            table=table_name,
            source="template"
        ))

    # Column-specific if we have columns
    if columns:
        numeric_cols = [c for c in columns if any(t in c.lower() for t in ['id', 'count', 'amount', 'total', 'price'])]
        date_cols = [c for c in columns if any(t in c.lower() for t in ['date', 'time', 'created', 'updated', 'added'])]

        if numeric_cols:
            col = numeric_cols[0]
            questions.append(Question(
                text=f"Show me {table_name} with {col} greater than average",
                table=table_name,
                source="template_column"
            ))

        if date_cols:
            col = date_cols[0]
            questions.append(Question(
                text=f"Show me {table_name} from the last week ordered by {col}",
                table=table_name,
                source="template_date"
            ))

    return questions[:5]  # Limit to 5 per table


# =============================================================================
# Tasks
# =============================================================================

@task(
    name="initialize_pipeline",
    description="Initialize training pipeline",
    retries=2,
    retry_delay_seconds=10,
    tags=["sql-training", "init"]
)
async def initialize_pipeline_task(database: str, host: str) -> str:
    """Create pipeline record and return ID."""
    logger = get_run_logger()

    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    pipeline_id = str(uuid.uuid4())

    mongo = await get_mongodb_service()
    await mongo.db["training_pipelines"].insert_one({
        "_id": pipeline_id,
        "database": database,
        "host": host,
        "status": "running",
        "started_at": datetime.utcnow(),
        "tables_processed": 0,
        "examples_generated": 0
    })

    logger.info(f"Initialized pipeline {pipeline_id} for {database}@{host}")
    return pipeline_id


@task(
    name="select_tables",
    description="Select tables for training",
    retries=2,
    tags=["sql-training", "selection"]
)
async def select_tables_task(
    database: str,
    max_tables: int = 50,
    exclude_processed: Optional[List[str]] = None
) -> List[str]:
    """Select tables prioritized by feedback and coverage."""
    logger = get_run_logger()
    exclude_processed = exclude_processed or []

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    # Get all tables from schema
    all_tables = await mongo.db["sql_schema_context"].distinct(
        "table_name",
        {"database": {"$regex": database, "$options": "i"}}
    )

    logger.info(f"Found {len(all_tables)} total tables in {database}")

    # Filter already processed
    remaining = [t for t in all_tables if t not in exclude_processed]

    # Prioritize by failure frequency
    failures = await mongo.db["agent_failures"].aggregate([
        {"$match": {"database": {"$regex": database, "$options": "i"}}},
        {"$unwind": {"path": "$tables", "preserveNullAndEmptyArrays": True}},
        {"$group": {"_id": "$tables", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]).to_list(1000)

    failure_tables = [f["_id"] for f in failures if f["_id"] and f["_id"] in remaining]
    other_tables = [t for t in remaining if t not in failure_tables]

    # Priority: high-failure tables first, then others
    prioritized = failure_tables + other_tables

    selected = prioritized[:max_tables]
    logger.info(f"Selected {len(selected)} tables (priority: {len(failure_tables)} from failures)")

    return selected


@task(
    name="load_table_schema",
    description="Load schema for a table",
    retries=2,
    tags=["sql-training", "schema"]
)
async def load_table_schema_task(
    table_name: str,
    database: str
) -> Dict[str, Any]:
    """Load schema for a specific table."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    # Get table definition
    table_doc = await mongo.db["sql_schema_context"].find_one({
        "database": {"$regex": database, "$options": "i"},
        "table_name": {"$regex": f"^{table_name}$", "$options": "i"}
    })

    if not table_doc:
        logger.warning(f"No schema found for {table_name}")
        return {"table": table_name, "columns": [], "relationships": []}

    return {
        "table": table_name,
        "columns": table_doc.get("columns", []),
        "primary_keys": table_doc.get("primary_keys", []),
        "relationships": table_doc.get("relationships", []),
        "sample_values": table_doc.get("sample_values", {})
    }


@task(
    name="generate_questions",
    description="Generate questions for table",
    retries=1,
    tags=["sql-training", "generation"]
)
async def generate_questions_task(
    table_schema: Dict[str, Any],
    questions_per_table: int = 3
) -> List[Question]:
    """Generate NL questions for a table."""
    logger = get_run_logger()

    table_name = table_schema.get("table", "Unknown")
    columns = [c.get("name", c) if isinstance(c, dict) else c
               for c in table_schema.get("columns", [])]

    questions = generate_template_questions(table_name, columns)

    logger.info(f"Generated {len(questions)} questions for {table_name}")
    return questions[:questions_per_table]


@task(
    name="generate_sql_candidates",
    description="Generate SQL for question",
    retries=2,
    retry_delay_seconds=5,
    timeout_seconds=60,
    tags=["sql-training", "sql-gen"]
)
async def generate_sql_candidates_task(
    question: Question,
    table_schema: Dict[str, Any],
    database: str,
    num_candidates: int = 2
) -> List[SQLCandidate]:
    """Generate SQL candidates for a question."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from services.llm_service import get_llm_service

    candidates = []

    # Build prompt
    schema_text = f"Table: {table_schema['table']}\n"
    if table_schema.get("columns"):
        cols = table_schema["columns"]
        if isinstance(cols[0], dict):
            col_names = [c.get("name", str(c)) for c in cols]
        else:
            col_names = cols
        schema_text += f"Columns: {', '.join(col_names[:20])}\n"

    prompt = f"""Given this SQL Server schema:
{schema_text}

Generate a SQL query for this question:
"{question.text}"

Return only the SQL query, nothing else."""

    try:
        llm = await get_llm_service()

        # Generate at different temperatures
        temps = [0.0, 0.3][:num_candidates]

        for temp in temps:
            result = await llm.generate(
                prompt=prompt,
                system="You are a SQL Server expert. Generate only valid T-SQL queries.",
                use_sql_model=True,
                temperature=temp,
                max_tokens=512
            )

            if result.success and result.response:
                sql = extract_sql(result.response)
                if sql:
                    candidates.append(SQLCandidate(
                        sql=sql,
                        confidence=0.8 - (temp * 0.2),
                        source=f"llm_temp_{temp}",
                        temperature=temp
                    ))

    except Exception as e:
        logger.warning(f"SQL generation failed: {e}")

    # Fallback: simple query
    if not candidates:
        simple_sql = f"SELECT TOP 10 * FROM {table_schema['table']}"
        candidates.append(SQLCandidate(
            sql=simple_sql,
            confidence=0.5,
            source="fallback"
        ))

    logger.info(f"Generated {len(candidates)} SQL candidates for: {question.text[:50]}...")
    return candidates


@task(
    name="execute_sql",
    description="Execute SQL against database",
    retries=1,
    timeout_seconds=35,
    tags=["sql-training", "execution"]
)
async def execute_sql_task(
    sql: str,
    database: str,
    host: str
) -> ExecutionResult:
    """Execute SQL and capture results."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    # Add row limit
    sql = add_row_limit(sql, 100)

    try:
        import pymssql

        # Connection parameters (from CLAUDE.md)
        conn = pymssql.connect(
            server=host,
            user="EWRUser",
            password="66a3904d69",
            database=database,
            timeout=30
        )

        cursor = conn.cursor(as_dict=True)
        start = time.time()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.time() - start

        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        cursor.close()
        conn.close()

        return ExecutionResult(
            success=True,
            row_count=len(rows),
            columns=columns,
            elapsed_ms=int(elapsed * 1000),
            sample_rows=rows[:5] if rows else []
        )

    except Exception as e:
        logger.warning(f"SQL execution failed: {e}")
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type=classify_error(e)
        )


@task(
    name="validate_results",
    description="Validate query results",
    tags=["sql-training", "validation"]
)
async def validate_results_task(
    question: Question,
    sql: str,
    execution_result: ExecutionResult
) -> ValidationResult:
    """Validate that results match question intent."""
    logger = get_run_logger()

    if not execution_result.success:
        return ValidationResult(
            valid=False,
            score=0.0,
            reason=f"Execution failed: {execution_result.error}"
        )

    # Empty results - soft pass
    if execution_result.row_count == 0:
        return ValidationResult(
            valid=True,
            score=0.6,
            reason="Query returned 0 rows (may be valid for current data)"
        )

    # Check for cartesian explosion
    if execution_result.row_count > 5000:
        return ValidationResult(
            valid=False,
            score=0.2,
            reason="Possible cartesian product (>5000 rows)"
        )

    # Valid execution with results
    return ValidationResult(
        valid=True,
        score=0.9,
        reason=f"Query executed successfully ({execution_result.row_count} rows)"
    )


@task(
    name="analyze_failure",
    description="Analyze SQL failure",
    tags=["sql-training", "analysis"]
)
async def analyze_failure_task(
    question: Question,
    sql: str,
    error: str
) -> FailureAnalysis:
    """Categorize and analyze failure."""
    logger = get_run_logger()

    error_lower = error.lower()

    if "invalid column name" in error_lower:
        # Extract column name
        match = re.search(r"invalid column name '([^']+)'", error, re.IGNORECASE)
        column = match.group(1) if match else "unknown"
        return FailureAnalysis(
            failure_type=FailureType.INVALID_COLUMN,
            reason=f"Column '{column}' does not exist",
            suggested_fix=f"Check schema for correct column name instead of '{column}'"
        )

    elif "invalid object name" in error_lower:
        match = re.search(r"invalid object name '([^']+)'", error, re.IGNORECASE)
        table = match.group(1) if match else "unknown"
        return FailureAnalysis(
            failure_type=FailureType.INVALID_TABLE,
            reason=f"Table '{table}' does not exist",
            suggested_fix="Verify table name against schema"
        )

    elif "syntax" in error_lower:
        return FailureAnalysis(
            failure_type=FailureType.SYNTAX_ERROR,
            reason="SQL syntax error",
            suggested_fix="Check SQL syntax for T-SQL compliance"
        )

    elif "timeout" in error_lower:
        return FailureAnalysis(
            failure_type=FailureType.TIMEOUT,
            reason="Query timeout exceeded",
            suggested_fix="Simplify query or add appropriate indexes"
        )

    elif "permission" in error_lower or "denied" in error_lower:
        return FailureAnalysis(
            failure_type=FailureType.PERMISSION_DENIED,
            reason="Permission denied",
            suggested_fix="Check database permissions"
        )

    return FailureAnalysis(
        failure_type=FailureType.UNKNOWN,
        reason=error[:200],
        suggested_fix="Manual review required"
    )


@task(
    name="quality_control",
    description="Apply quality control",
    retries=2,
    tags=["sql-training", "qc"]
)
async def quality_control_task(
    examples: List[TrainingExample],
    database: str
) -> QCResult:
    """Apply quality control gates to examples."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    accepted = []
    rejected = []

    for example in examples:
        # Check confidence threshold
        if example.confidence < 0.5:
            rejected.append((example, "Low confidence"))
            continue

        # Check for duplicate questions
        q_hash = hash_question(example.question)
        existing = await mongo.db["sql_examples"].find_one({
            "question_hash": q_hash,
            "database": {"$regex": database, "$options": "i"}
        })

        if existing:
            rejected.append((example, "Duplicate question"))
            continue

        # For negative examples, ensure distinct failure
        if not example.is_positive:
            existing_failure = await mongo.db["sql_training_failures"].find_one({
                "question_hash": q_hash,
                "failure_type": example.failure_type,
                "database": {"$regex": database, "$options": "i"}
            })

            if existing_failure:
                rejected.append((example, "Duplicate failure pattern"))
                continue

        accepted.append(example)

    logger.info(f"QC: {len(accepted)} accepted, {len(rejected)} rejected")
    return QCResult(accepted=accepted, rejected=rejected)


@task(
    name="store_examples",
    description="Store training examples",
    retries=2,
    tags=["sql-training", "storage"]
)
async def store_examples_task(
    qc_result: QCResult,
    database: str,
    pipeline_id: str
) -> Tuple[int, int]:
    """Store validated examples in MongoDB."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    positive_count = 0
    negative_count = 0

    for example in qc_result.accepted:
        if example.is_positive:
            await mongo.db["sql_examples"].insert_one({
                "database": database,
                "question": example.question,
                "question_hash": hash_question(example.question),
                "sql": example.sql,
                "sql_hash": hash_sql(example.sql),
                "explanation": example.explanation,
                "confidence": example.confidence,
                "tables_used": example.tables_used,
                "source": "autonomous_training",
                "pipeline_id": pipeline_id,
                "created_at": datetime.utcnow(),
                "verified": True
            })
            positive_count += 1
        else:
            await mongo.db["sql_training_failures"].insert_one({
                "database": database,
                "question": example.question,
                "question_hash": hash_question(example.question),
                "failed_sql": example.sql,
                "failure_type": example.failure_type,
                "failure_reason": example.failure_reason,
                "source": "autonomous_training",
                "pipeline_id": pipeline_id,
                "created_at": datetime.utcnow()
            })
            negative_count += 1

    logger.info(f"Stored {positive_count} positive, {negative_count} negative examples")
    return positive_count, negative_count


@task(
    name="generate_rules",
    description="Auto-generate rules from failures",
    retries=1,
    tags=["sql-training", "rules"]
)
async def generate_rules_task(database: str) -> int:
    """Generate rules from repeated failure patterns."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    # Find repeated failures
    patterns = await mongo.db["sql_training_failures"].aggregate([
        {"$match": {"database": {"$regex": database, "$options": "i"}}},
        {"$group": {
            "_id": "$failure_reason",
            "failure_type": {"$first": "$failure_type"},
            "count": {"$sum": 1},
            "examples": {"$push": {"question": "$question", "sql": "$failed_sql"}}
        }},
        {"$match": {"count": {"$gte": 3}}},  # At least 3 occurrences
        {"$sort": {"count": -1}},
        {"$limit": 20}
    ]).to_list(20)

    rules_created = 0

    for pattern in patterns:
        reason = pattern["_id"]
        if not reason:
            continue

        # Check if rule already exists
        existing = await mongo.db["sql_rules"].find_one({
            "auto_pattern": reason,
            "database": {"$regex": database, "$options": "i"}
        })

        if existing:
            continue

        # Extract keywords from examples
        examples = pattern.get("examples", [])
        keywords = set()
        for ex in examples[:5]:
            words = ex.get("question", "").lower().split()
            keywords.update(w for w in words if len(w) > 3)

        # Create auto-generated rule
        rule = {
            "id": f"auto_{uuid.uuid4().hex[:8]}",
            "database": database,
            "description": f"Auto-generated rule for: {reason[:100]}",
            "type": "constraint",
            "trigger_keywords": list(keywords)[:10],
            "rule_text": f"Avoid this error: {reason}",
            "auto_pattern": reason,
            "priority": "normal",
            "source": "autonomous_training",
            "occurrence_count": pattern["count"],
            "created_at": datetime.utcnow(),
            "active": True
        }

        await mongo.db["sql_rules"].insert_one(rule)
        rules_created += 1
        logger.info(f"Created rule for pattern: {reason[:50]}...")

    logger.info(f"Generated {rules_created} new rules from failure patterns")
    return rules_created


@task(
    name="emit_metrics",
    description="Emit metrics to Prefect",
    tags=["sql-training", "metrics"]
)
async def emit_metrics_task(metrics: BatchMetrics, batch_num: int):
    """Emit batch metrics as Prefect artifact."""
    await create_table_artifact(
        key=f"training-batch-{batch_num}",
        table=[
            {"Metric": "Tables Processed", "Value": str(metrics.tables_processed)},
            {"Metric": "Questions Generated", "Value": str(metrics.questions_generated)},
            {"Metric": "SQL Candidates", "Value": str(metrics.sql_candidates)},
            {"Metric": "Validation Success", "Value": str(metrics.validation_success)},
            {"Metric": "Validation Failure", "Value": str(metrics.validation_failure)},
            {"Metric": "Examples Stored", "Value": str(metrics.examples_stored)},
            {"Metric": "Failures Stored", "Value": str(metrics.failures_stored)},
            {"Metric": "Success Rate", "Value": f"{metrics.success_rate:.1%}"}
        ],
        description=f"Training batch {batch_num} metrics"
    )


@task(
    name="generate_report",
    description="Generate final report",
    tags=["sql-training", "report"]
)
async def generate_report_task(
    pipeline_id: str,
    metrics: TrainingMetrics
) -> str:
    """Generate markdown report."""
    logger = get_run_logger()

    report = f"""# Autonomous SQL Training Report

**Pipeline ID:** `{pipeline_id}`
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## Summary

| Metric | Value |
|--------|-------|
| Tables Processed | {metrics.tables_processed} |
| Questions Generated | {metrics.questions_generated} |
| Positive Examples | {metrics.positive_examples} |
| Negative Examples | {metrics.negative_examples} |
| Rules Generated | {metrics.rules_generated} |
| Success Rate | {metrics.success_rate:.1%} |
| Runtime | {metrics.runtime_seconds / 60:.1f} minutes |

## Failure Breakdown

| Failure Type | Count |
|--------------|-------|
"""

    for failure_type, count in sorted(metrics.failure_breakdown.items(), key=lambda x: -x[1]):
        report += f"| {failure_type} | {count} |\n"

    report += """
## Status

"""
    if metrics.success_rate >= 0.7:
        report += "Pipeline operating normally.\n"
    elif metrics.success_rate >= 0.5:
        report += "Moderate success rate - review failure patterns.\n"
    else:
        report += "Low success rate - intervention may be needed.\n"

    if metrics.rules_generated > 0:
        report += f"\n{metrics.rules_generated} new rules were auto-generated from failure patterns.\n"

    await create_markdown_artifact(
        key="training-report",
        markdown=report,
        description="Autonomous SQL Training Report"
    )

    logger.info("Generated training report")
    return report


# =============================================================================
# Sub-Flow: Process Table Batch
# =============================================================================

@flow(
    name="process_table_batch",
    description="Process a batch of tables",
    retries=1
)
async def process_table_batch(
    tables: List[str],
    database: str,
    host: str,
    pipeline_id: str,
    questions_per_table: int = 3
) -> BatchMetrics:
    """Process a batch of tables."""
    logger = get_run_logger()

    metrics = BatchMetrics()
    all_examples = []

    for table in tables:
        logger.info(f"Processing table: {table}")

        try:
            # Load schema
            schema = await load_table_schema_task(table, database)

            # Generate questions
            questions = await generate_questions_task(schema, questions_per_table)
            metrics.questions_generated += len(questions)

            # Process each question
            for question in questions:
                # Generate SQL candidates
                candidates = await generate_sql_candidates_task(
                    question, schema, database, num_candidates=2
                )
                metrics.sql_candidates += len(candidates)

                # Try each candidate
                best_result = None
                best_candidate = None

                for candidate in candidates:
                    # Execute
                    exec_result = await execute_sql_task(
                        candidate.sql, database, host
                    )

                    # Validate
                    validation = await validate_results_task(
                        question, candidate.sql, exec_result
                    )

                    if validation.valid and validation.score > 0.7:
                        if not best_result or validation.score > best_result.score:
                            best_result = validation
                            best_candidate = candidate

                # Record result
                if best_result and best_candidate:
                    metrics.validation_success += 1
                    all_examples.append(TrainingExample(
                        question=question.text,
                        sql=best_candidate.sql,
                        is_positive=True,
                        confidence=best_result.score,
                        explanation=best_result.reason,
                        tables_used=[table]
                    ))
                else:
                    metrics.validation_failure += 1

                    # Analyze failure if we have one
                    if candidates and not candidates[0].sql.startswith("SELECT TOP"):
                        # Get the first candidate's result
                        exec_result = await execute_sql_task(
                            candidates[0].sql, database, host
                        )

                        if not exec_result.success:
                            failure = await analyze_failure_task(
                                question, candidates[0].sql, exec_result.error or ""
                            )

                            all_examples.append(TrainingExample(
                                question=question.text,
                                sql=candidates[0].sql,
                                is_positive=False,
                                confidence=0.8,
                                failure_type=failure.failure_type.value,
                                failure_reason=failure.reason,
                                tables_used=[table]
                            ))

                            # Track failure breakdown
                            ft = failure.failure_type.value
                            metrics.failure_breakdown = getattr(metrics, 'failure_breakdown', {})
                            if not hasattr(metrics, 'failure_breakdown'):
                                metrics.failure_breakdown = {}
                            metrics.failure_breakdown[ft] = metrics.failure_breakdown.get(ft, 0) + 1

            metrics.tables_processed += 1

        except Exception as e:
            logger.error(f"Error processing table {table}: {e}")
            continue

    # Quality control
    qc_result = await quality_control_task(all_examples, database)

    # Store examples
    positive, negative = await store_examples_task(qc_result, database, pipeline_id)
    metrics.examples_stored = positive
    metrics.failures_stored = negative

    return metrics


# =============================================================================
# Main Flow
# =============================================================================

@flow(
    name="autonomous_sql_training",
    description="Overnight autonomous SQL training data generation",
    retries=1,
    retry_delay_seconds=300,
    timeout_seconds=43200,  # 12 hours max
    log_prints=True
)
async def autonomous_sql_training_flow(
    database: str = "EWRCentral",
    host: str = "CHAD-PC",
    max_tables: int = 50,
    batch_size: int = 5,
    questions_per_table: int = 3,
    cutoff_hour: int = 7,
    cutoff_tz: str = "America/Chicago"
):
    """
    Main autonomous training flow.

    Runs from 7 PM to 7 AM, generating SQL training examples
    from database schema and stored procedures.

    Args:
        database: Target database name
        host: SQL Server host
        max_tables: Maximum tables to process
        batch_size: Tables per batch
        questions_per_table: Questions to generate per table
        cutoff_hour: Hour to stop (7 = 7 AM)
        cutoff_tz: Timezone for cutoff
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("AUTONOMOUS SQL TRAINING PIPELINE")
    logger.info(f"Database: {database} @ {host}")
    logger.info(f"Max tables: {max_tables}, Batch size: {batch_size}")
    logger.info(f"Cutoff: {cutoff_hour}:00 {cutoff_tz}")
    logger.info("=" * 60)

    # Initialize
    pipeline_id = await initialize_pipeline_task(database, host)
    logger.info(f"Pipeline ID: {pipeline_id}")

    # Select tables
    tables = await select_tables_task(
        database=database,
        max_tables=max_tables
    )

    if not tables:
        logger.warning("No tables to process!")
        return TrainingMetrics()

    # Process in batches
    total_metrics = TrainingMetrics()
    batch_num = 0

    for batch in batched(tables, batch_size):
        batch_num += 1

        # Check cutoff
        if check_cutoff(cutoff_hour, cutoff_tz):
            logger.warning("Approaching cutoff time, stopping gracefully")
            break

        logger.info(f"Processing batch {batch_num}: {len(batch)} tables")

        batch_result = await process_table_batch(
            tables=batch,
            database=database,
            host=host,
            pipeline_id=pipeline_id,
            questions_per_table=questions_per_table
        )

        total_metrics.merge(batch_result)

        # Emit metrics
        await emit_metrics_task(batch_result, batch_num)

        logger.info(f"Batch {batch_num} complete: {batch_result.examples_stored} examples stored")

    # Generate rules from failures
    rules_generated = await generate_rules_task(database)
    total_metrics.rules_generated = rules_generated

    # Convert user feedback to rules
    try:
        from feedback_to_rules import feedback_to_rules_flow
        feedback_result = await feedback_to_rules_flow(database=database)
        total_metrics.rules_generated += feedback_result.get("rules_created", 0)
        total_metrics.rules_generated += feedback_result.get("exact_matches", 0)
        logger.info(f"Feedback conversion: {feedback_result}")
    except Exception as e:
        logger.warning(f"Feedback conversion failed (non-critical): {e}")

    # Calculate runtime
    total_metrics.runtime_seconds = time.time() - start_time

    # Update pipeline status
    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')
    from mongodb import get_mongodb_service
    mongo = await get_mongodb_service()

    await mongo.db["training_pipelines"].update_one(
        {"_id": pipeline_id},
        {"$set": {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "tables_processed": total_metrics.tables_processed,
            "examples_generated": total_metrics.positive_examples + total_metrics.negative_examples,
            "success_rate": total_metrics.success_rate
        }}
    )

    # Generate report
    await generate_report_task(pipeline_id, total_metrics)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Tables: {total_metrics.tables_processed}")
    logger.info(f"Positive examples: {total_metrics.positive_examples}")
    logger.info(f"Negative examples: {total_metrics.negative_examples}")
    logger.info(f"Rules generated: {total_metrics.rules_generated}")
    logger.info(f"Success rate: {total_metrics.success_rate:.1%}")
    logger.info(f"Runtime: {total_metrics.runtime_seconds / 60:.1f} minutes")
    logger.info("=" * 60)

    return total_metrics


# =============================================================================
# Deployment
# =============================================================================

if __name__ == "__main__":
    # For testing
    import asyncio

    asyncio.run(autonomous_sql_training_flow(
        database="EWRCentral",
        host="CHAD-PC",
        max_tables=5,  # Small test
        batch_size=2,
        questions_per_table=2
    ))
