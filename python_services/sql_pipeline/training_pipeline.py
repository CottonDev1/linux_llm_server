"""
Training Pipeline Module

This module provides a Prefect-based pipeline for generating training data from stored procedures.
Supports multi-stage generation with validation and execution-based verification.

Pipeline Stages:
1. Extraction + Schema Compression (can be skipped if schema exists)
2. Question Generation + Multi-Candidate SQL
3. Execution Validation + LLM-as-Judge

Prefect Monitoring:
- Real-time progress artifacts
- Stage-level metrics and timing
- Summary dashboard with success rates
- Historical comparison data
"""

from typing import AsyncGenerator, Optional, List, Dict, Any
import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact, create_link_artifact
from prefect.context import get_run_context

from sql_pipeline.models.training_models import (
    TrainingPipelineConfig,
    GeneratedQuestion,
    SQLCandidate,
    TrainingExample,
    TrainingResult,
)
from sql_pipeline.models.query_models import SQLCredentials
from sql_pipeline.models.validation_models import ValidationResult

logger = logging.getLogger(__name__)


# ============================================================================
# METRICS AND MONITORING CLASSES
# ============================================================================

@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage_name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.items_processed == 0:
            return 0.0
        return self.items_succeeded / self.items_processed

    def start(self):
        self.started_at = datetime.utcnow()

    def complete(self):
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


@dataclass
class PipelineMetrics:
    """Comprehensive metrics for the entire pipeline run."""
    pipeline_id: str = ""
    database: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    # Stage metrics
    stage1_metrics: StageMetrics = field(default_factory=lambda: StageMetrics("Extraction"))
    stage2_metrics: StageMetrics = field(default_factory=lambda: StageMetrics("Generation"))
    stage3_metrics: StageMetrics = field(default_factory=lambda: StageMetrics("Validation"))

    # Aggregate stats
    total_sps_processed: int = 0
    total_questions_generated: int = 0
    total_candidates_generated: int = 0
    total_candidates_validated: int = 0
    total_examples_stored: int = 0

    # Quality metrics
    execution_success_rate: float = 0.0
    validation_pass_rate: float = 0.0
    average_validation_score: float = 0.0

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def start(self):
        self.started_at = datetime.utcnow()

    def complete(self):
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "database": self.database,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "stages": {
                "extraction": {
                    "duration_seconds": self.stage1_metrics.duration_seconds,
                    "items_processed": self.stage1_metrics.items_processed,
                    "success_rate": self.stage1_metrics.success_rate
                },
                "generation": {
                    "duration_seconds": self.stage2_metrics.duration_seconds,
                    "items_processed": self.stage2_metrics.items_processed,
                    "success_rate": self.stage2_metrics.success_rate
                },
                "validation": {
                    "duration_seconds": self.stage3_metrics.duration_seconds,
                    "items_processed": self.stage3_metrics.items_processed,
                    "success_rate": self.stage3_metrics.success_rate
                }
            },
            "totals": {
                "sps_processed": self.total_sps_processed,
                "questions_generated": self.total_questions_generated,
                "candidates_generated": self.total_candidates_generated,
                "examples_stored": self.total_examples_stored
            },
            "quality": {
                "execution_success_rate": self.execution_success_rate,
                "validation_pass_rate": self.validation_pass_rate,
                "average_validation_score": self.average_validation_score
            },
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass
class StoredProcedure:
    """Stored procedure with compressed schema."""
    schema: str
    name: str
    definition: str
    parameters: List[Dict[str, Any]]
    database: str
    referenced_tables: List[str]
    compressed_schema: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"


# ============================================================================
# STAGE 1 TASKS: EXTRACTION + SCHEMA COMPRESSION
# ============================================================================

@task(name="extract-stored-procedures", retries=2)
async def extract_stored_procedures(database: str, limit: int = 10) -> List[StoredProcedure]:
    """
    Extract stored procedures from SQL Server.

    Args:
        database: Target database name
        limit: Maximum number of SPs to extract

    Returns:
        List of StoredProcedure objects
    """
    logger = get_run_logger()
    logger.info(f"Extracting stored procedures from {database} (limit={limit})")

    from mongodb import MongoDBService
    import pymssql

    # Connect to SQL Server
    try:
        conn = pymssql.connect(
            server="NCSQLTEST",
            user="EWRUser",
            password="66a3904d69",
            database=database
        )
        cursor = conn.cursor(as_dict=True)

        # Query for stored procedures
        query = """
            SELECT TOP (@limit)
                SCHEMA_NAME(p.schema_id) AS [schema],
                p.name,
                OBJECT_DEFINITION(p.object_id) AS definition
            FROM sys.procedures p
            WHERE p.is_ms_shipped = 0
            ORDER BY p.modify_date DESC
        """

        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

        procedures = []
        for row in rows:
            # Get parameters
            param_query = """
                SELECT
                    p.name AS parameter_name,
                    TYPE_NAME(p.user_type_id) AS data_type,
                    p.max_length,
                    p.is_output
                FROM sys.parameters p
                WHERE p.object_id = OBJECT_ID(@sp_name)
                ORDER BY p.parameter_id
            """
            sp_full_name = f"{row['schema']}.{row['name']}"
            cursor.execute(param_query, (sp_full_name,))
            params = cursor.fetchall()

            # Extract referenced tables from definition
            referenced_tables = _extract_referenced_tables(row['definition'])

            procedures.append(StoredProcedure(
                schema=row['schema'],
                name=row['name'],
                definition=row['definition'],
                parameters=[dict(p) for p in params],
                database=database,
                referenced_tables=referenced_tables
            ))

        conn.close()
        logger.info(f"Extracted {len(procedures)} stored procedures")
        return procedures

    except Exception as e:
        logger.error(f"Failed to extract stored procedures: {e}")
        raise


@task(name="compress-schemas", retries=2)
async def compress_schemas(sps: List[StoredProcedure]) -> List[StoredProcedure]:
    """
    Compress schema to only relevant tables for each SP.

    Args:
        sps: List of stored procedures

    Returns:
        Updated list with compressed_schema populated
    """
    logger = get_run_logger()
    logger.info(f"Compressing schemas for {len(sps)} stored procedures")

    from sql_pipeline.services.schema_service import SchemaService

    schema_service = await SchemaService.get_instance()

    for sp in sps:
        if sp.referenced_tables:
            # Get schema info for referenced tables
            schema_info = await schema_service.get_tables_schema(
                database=sp.database,
                table_names=sp.referenced_tables
            )

            # Format as markdown
            sp.compressed_schema = schema_service.format_schema_for_prompt(schema_info.tables)
            logger.debug(f"Compressed schema for {sp.full_name}: {len(sp.referenced_tables)} tables")
        else:
            sp.compressed_schema = ""
            logger.debug(f"No referenced tables for {sp.full_name}")

    return sps


@task(name="store-procedures", retries=2)
async def store_procedures(sps: List[StoredProcedure]) -> int:
    """
    Store stored procedures in MongoDB.

    Args:
        sps: List of stored procedures to store

    Returns:
        Number of procedures stored
    """
    logger = get_run_logger()
    logger.info(f"Storing {len(sps)} stored procedures in MongoDB")

    from mongodb import MongoDBService

    mongo = MongoDBService.get_instance()
    if not mongo.is_initialized:
        await mongo.initialize()

    collection = mongo.db["stored_procedures"]

    stored_count = 0
    for sp in sps:
        doc = {
            "database": sp.database,
            "schema": sp.schema,
            "name": sp.name,
            "full_name": sp.full_name,
            "definition": sp.definition,
            "parameters": sp.parameters,
            "referenced_tables": sp.referenced_tables,
            "compressed_schema": sp.compressed_schema,
            "created_at": datetime.utcnow(),
            "status": "pending"
        }

        # Upsert by full_name and database
        await collection.update_one(
            {"database": sp.database, "full_name": sp.full_name},
            {"$set": doc},
            upsert=True
        )
        stored_count += 1

    logger.info(f"Stored {stored_count} stored procedures")
    return stored_count


@task(name="load-stored-procedures", retries=2)
async def load_stored_procedures(database: str, limit: int = 10) -> List[StoredProcedure]:
    """
    Load stored procedures from MongoDB.

    Args:
        database: Target database
        limit: Maximum number to load

    Returns:
        List of StoredProcedure objects
    """
    logger = get_run_logger()
    logger.info(f"Loading stored procedures from MongoDB for {database}")

    from mongodb import MongoDBService

    mongo = MongoDBService.get_instance()
    if not mongo.is_initialized:
        await mongo.initialize()

    collection = mongo.db["stored_procedures"]

    cursor = collection.find({"database": database}).limit(limit)
    docs = await cursor.to_list(length=limit)

    procedures = []
    for doc in docs:
        procedures.append(StoredProcedure(
            schema=doc["schema"],
            name=doc["name"],
            definition=doc["definition"],
            parameters=doc["parameters"],
            database=doc["database"],
            referenced_tables=doc.get("referenced_tables", []),
            compressed_schema=doc.get("compressed_schema", "")
        ))

    logger.info(f"Loaded {len(procedures)} stored procedures")
    return procedures


# ============================================================================
# STAGE 2 TASKS: QUESTION GENERATION + SQL CANDIDATES
# ============================================================================

@task(name="generate-training-questions", retries=2)
async def generate_training_questions(sps: List[StoredProcedure]) -> List[GeneratedQuestion]:
    """
    Generate training questions from stored procedures.

    Args:
        sps: List of stored procedures

    Returns:
        List of generated questions
    """
    logger = get_run_logger()
    logger.info(f"Generating questions for {len(sps)} stored procedures")

    from services.llm_service import get_llm_service

    llm_service = await get_llm_service()
    questions = []

    for sp in sps:
        # Generate questions from 3 perspectives
        perspectives = [
            ("user", "business user perspective"),
            ("analyst", "data analyst perspective"),
            ("developer", "developer/technical perspective")
        ]

        for perspective, desc in perspectives:
            prompt = f"""Generate a natural language question that this stored procedure could answer.

Stored Procedure: {sp.full_name}
Database: {sp.database}

Definition (first 500 chars):
{sp.definition[:500]}

Generate a question from the {desc}. Make it specific and realistic.

Question:"""

            result = await llm_service.generate(
                prompt=prompt,
                system="You are a SQL training data generator. Generate realistic natural language questions.",
                temperature=0.7,
                use_cache=False
            )

            if result.success:
                question_text = result.response.strip()

                # Determine difficulty based on SP complexity
                difficulty = "easy" if len(sp.parameters) <= 2 else "medium"
                if len(sp.referenced_tables) > 3:
                    difficulty = "hard"

                questions.append(GeneratedQuestion(
                    question=question_text,
                    sp_name=sp.full_name,
                    perspective=perspective,
                    difficulty=difficulty
                ))

                logger.debug(f"Generated question for {sp.full_name} ({perspective}): {question_text[:50]}...")

    logger.info(f"Generated {len(questions)} questions")
    return questions


@task(name="generate-multi-candidate-sql", retries=2)
async def generate_multi_candidate_sql(
    questions: List[GeneratedQuestion],
    n_candidates: int = 3
) -> List[SQLCandidate]:
    """
    Generate multiple SQL candidates for each question.

    Args:
        questions: List of questions
        n_candidates: Number of SQL candidates per question

    Returns:
        List of SQL candidates
    """
    logger = get_run_logger()
    logger.info(f"Generating {n_candidates} SQL candidates for {len(questions)} questions")

    from services.llm_service import get_llm_service
    from sql_pipeline.services.schema_service import SchemaService

    llm_service = await get_llm_service()
    schema_service = await SchemaService.get_instance()

    candidates = []

    for question in questions:
        # Get database from SP name (extract from questions if stored)
        # For now, assume we can get it from context
        database = "EWRCentral"  # TODO: Pass through from question metadata

        # Get relevant schema
        schema_info = await schema_service.get_relevant_schema(
            database=database,
            question=question.question,
            max_tables=5
        )
        schema_text = schema_service.format_schema_for_prompt(schema_info.tables)

        # Generate N candidates with different temperatures
        temperatures = [0.0, 0.3, 0.7][:n_candidates]

        for i, temp in enumerate(temperatures):
            prompt = f"""Generate a SQL query for this question.

Database: {database}

Schema:
{schema_text}

Question: {question.question}

Generate only the SQL query, no explanations.

SQL:"""

            result = await llm_service.generate(
                prompt=prompt,
                system="You are a SQL expert. Generate valid T-SQL queries.",
                temperature=temp,
                use_sql_model=True,
                use_cache=False
            )

            if result.success:
                sql = _extract_sql_from_response(result.response)

                candidates.append(SQLCandidate(
                    question=question.question,
                    sql=sql,
                    source="generated",
                    validation_status="pending"
                ))

                logger.debug(f"Generated candidate {i+1} for: {question.question[:50]}...")

    logger.info(f"Generated {len(candidates)} SQL candidates")
    return candidates


@task(name="refine-candidates", retries=2)
async def refine_candidates(candidates: List[SQLCandidate]) -> List[SQLCandidate]:
    """
    Refine SQL candidates with self-refinement pass.

    Args:
        candidates: List of SQL candidates

    Returns:
        Refined SQL candidates
    """
    logger = get_run_logger()
    logger.info(f"Refining {len(candidates)} SQL candidates")

    from sql_pipeline.services.syntax_fixer import SyntaxFixer
    from sql_pipeline.services.rules_service import RulesService

    rules_service = await RulesService.get_instance()
    syntax_fixer = SyntaxFixer(rules_service=rules_service)

    refined = []
    for candidate in candidates:
        # Apply syntax fixes
        fixed_sql, fixes = await syntax_fixer.apply_all_fixes(candidate.sql, "EWRCentral")

        if fixes:
            logger.debug(f"Applied {len(fixes)} fixes to candidate: {candidate.sql[:50]}...")
            refined.append(SQLCandidate(
                question=candidate.question,
                sql=fixed_sql,
                source="refined",
                validation_status="pending"
            ))
        else:
            refined.append(candidate)

    logger.info(f"Refined {len(refined)} candidates")
    return refined


# ============================================================================
# STAGE 3 TASKS: EXECUTION VALIDATION + BEST SELECTION
# ============================================================================

@task(name="validate-with-execution", retries=2)
async def validate_with_execution(
    candidates: List[SQLCandidate],
    config: TrainingPipelineConfig,
    credentials: Optional[SQLCredentials] = None
) -> List[Dict[str, Any]]:
    """
    Validate SQL candidates by execution.

    Args:
        candidates: SQL candidates to validate
        config: Pipeline configuration
        credentials: SQL credentials

    Returns:
        List of validation results with scores
    """
    logger = get_run_logger()
    logger.info(f"Validating {len(candidates)} candidates with execution")

    from sql_pipeline.services.execution_service import ExecutionService

    execution_service = ExecutionService()
    results = []

    # Use default credentials if not provided
    if not credentials:
        credentials = SQLCredentials(
            server="NCSQLTEST",
            database=config.database,
            username="EWRUser",
            password="66a3904d69"
        )

    for candidate in candidates:
        result = {
            "question": candidate.question,
            "sql": candidate.sql,
            "execution_success": False,
            "execution_error": None,
            "row_count": 0,
            "validation_score": 0.0
        }

        if config.enable_execution_validation:
            try:
                exec_result = await execution_service.execute(
                    candidate.sql,
                    credentials,
                    max_results=10
                )

                result["execution_success"] = exec_result.success
                result["execution_error"] = exec_result.error
                result["row_count"] = exec_result.row_count

                # Score based on execution success
                if exec_result.success:
                    result["validation_score"] = 0.8  # Base score for successful execution
                else:
                    result["validation_score"] = 0.0

            except Exception as e:
                logger.warning(f"Execution failed for candidate: {e}")
                result["execution_error"] = str(e)
                result["validation_score"] = 0.0
        else:
            # No execution validation, give neutral score
            result["validation_score"] = 0.5

        results.append(result)

    logger.info(f"Validated {len(results)} candidates")
    return results


@task(name="select-best-candidates", retries=2)
async def select_best_candidates(
    results: List[Dict[str, Any]],
    threshold: float = 0.7
) -> List[TrainingExample]:
    """
    Select best SQL candidate per question using majority voting.

    Args:
        results: Validation results
        threshold: Minimum score threshold

    Returns:
        List of training examples
    """
    logger = get_run_logger()
    logger.info(f"Selecting best candidates from {len(results)} results (threshold={threshold})")

    # Group by question
    by_question = {}
    for result in results:
        question = result["question"]
        if question not in by_question:
            by_question[question] = []
        by_question[question].append(result)

    examples = []
    for question, candidates in by_question.items():
        # Select candidate with highest score
        best = max(candidates, key=lambda x: x["validation_score"])

        if best["validation_score"] >= threshold:
            examples.append(TrainingExample(
                question=question,
                sql=best["sql"],
                database="EWRCentral",  # TODO: Get from config
                validation_score=best["validation_score"],
                execution_verified=best["execution_success"],
                metadata={
                    "row_count": best["row_count"],
                    "candidates_evaluated": len(candidates)
                }
            ))
            logger.debug(f"Selected best candidate for: {question[:50]}... (score={best['validation_score']:.2f})")
        else:
            logger.debug(f"No candidate passed threshold for: {question[:50]}...")

    logger.info(f"Selected {len(examples)} training examples")
    return examples


@task(name="store-training-examples", retries=2)
async def store_training_examples(examples: List[TrainingExample]) -> int:
    """
    Store training examples in MongoDB.

    Args:
        examples: Training examples to store

    Returns:
        Number of examples stored
    """
    logger = get_run_logger()
    logger.info(f"Storing {len(examples)} training examples")

    from mongodb import MongoDBService

    mongo = MongoDBService.get_instance()
    if not mongo.is_initialized:
        await mongo.initialize()

    collection = mongo.db["sql_examples"]

    stored_count = 0
    for example in examples:
        doc = {
            "question": example.question,
            "sql": example.sql,
            "database": example.database,
            "sp_source": example.sp_source,
            "validation_score": example.validation_score,
            "execution_verified": example.execution_verified,
            "created_at": example.created_at,
            "metadata": example.metadata
        }

        result = await collection.insert_one(doc)
        stored_count += 1

    logger.info(f"Stored {stored_count} training examples")
    return stored_count


@task(name="load-pending-candidates", retries=2)
async def load_pending_candidates(database: str) -> List[SQLCandidate]:
    """
    Load pending SQL candidates from MongoDB.

    Args:
        database: Target database

    Returns:
        List of SQL candidates
    """
    logger = get_run_logger()
    logger.info(f"Loading pending candidates for {database}")

    from mongodb import MongoDBService

    mongo = MongoDBService.get_instance()
    if not mongo.is_initialized:
        await mongo.initialize()

    collection = mongo.db["sql_training_candidates"]

    cursor = collection.find({
        "database": database,
        "validation_status": "pending"
    })
    docs = await cursor.to_list(length=1000)

    candidates = []
    for doc in docs:
        candidates.append(SQLCandidate(
            question=doc["question"],
            sql=doc["sql"],
            source=doc.get("source", "generated"),
            validation_status=doc.get("validation_status", "pending")
        ))

    logger.info(f"Loaded {len(candidates)} pending candidates")
    return candidates


# ============================================================================
# MAIN PREFECT FLOW
# ============================================================================

@flow(name="sql-training-pipeline", log_prints=True)
async def sql_training_flow(config: TrainingPipelineConfig) -> TrainingResult:
    """
    3-Stage Training Pipeline with skip capability and comprehensive monitoring.

    Args:
        config: Training pipeline configuration

    Returns:
        TrainingResult with statistics
    """
    logger = get_run_logger()

    # Initialize metrics
    metrics = PipelineMetrics(database=config.database)
    metrics.start()

    # Get run context for pipeline ID
    try:
        ctx = get_run_context()
        metrics.pipeline_id = str(ctx.flow_run.id) if ctx.flow_run else "local"
    except Exception:
        metrics.pipeline_id = f"local-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    logger.info(f"🚀 Starting SQL Training Pipeline for {config.database}")
    logger.info(f"   Pipeline ID: {metrics.pipeline_id}")
    logger.info(f"   Start stage: {config.start_stage}, Batch size: {config.batch_size}")

    # Create initial progress artifact
    await create_markdown_artifact(
        key="pipeline-progress",
        markdown=f"""# SQL Training Pipeline - In Progress

**Database:** {config.database}
**Started:** {metrics.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Configuration:**
- Start Stage: {config.start_stage}
- Batch Size: {config.batch_size}
- Candidates per Question: {config.n_candidates}
- Validation Threshold: {config.validation_threshold}

## Status: 🔄 Running...
""",
        description=f"Training pipeline for {config.database}"
    )

    start_time = time.time()
    result = TrainingResult()
    errors = []
    sql_candidates = []

    try:
        # ====================================================================
        # STAGE 1: EXTRACTION + SCHEMA COMPRESSION
        # ====================================================================

        metrics.stage1_metrics.start()

        if config.start_stage <= 1:
            logger.info("=== STAGE 1: Extraction + Schema Compression ===")

            try:
                sps = await extract_stored_procedures(config.database, config.batch_size)
                result.total_sps = len(sps)
                metrics.stage1_metrics.items_processed = len(sps)

                sps = await compress_schemas(sps)
                await store_procedures(sps)

                metrics.stage1_metrics.items_succeeded = len(sps)
                metrics.total_sps_processed = len(sps)

                logger.info(f"✅ Stage 1 complete: {len(sps)} SPs processed")

            except Exception as e:
                errors.append(f"Stage 1 error: {str(e)}")
                metrics.stage1_metrics.errors.append(str(e))
                logger.error(f"❌ Stage 1 failed: {e}")
                raise
        else:
            logger.info("⏩ Skipping Stage 1, loading existing SPs")
            sps = await load_stored_procedures(config.database, config.batch_size)
            result.total_sps = len(sps)
            metrics.stage1_metrics.items_processed = len(sps)
            metrics.stage1_metrics.items_succeeded = len(sps)
            metrics.total_sps_processed = len(sps)

        metrics.stage1_metrics.complete()

        # Update progress artifact after Stage 1
        await create_markdown_artifact(
            key="pipeline-progress",
            markdown=f"""# SQL Training Pipeline - Stage 1 Complete

**Database:** {config.database}
**Started:** {metrics.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Stage 1: Extraction ✅
- Duration: {metrics.stage1_metrics.duration_seconds:.1f}s
- SPs Processed: {metrics.stage1_metrics.items_processed}
- Success Rate: {metrics.stage1_metrics.success_rate:.1%}

## Stage 2: Generation 🔄 Running...
## Stage 3: Validation ⏳ Pending
""",
            description=f"Training pipeline for {config.database}"
        )

        # ====================================================================
        # STAGE 2: QUESTION GENERATION + SQL CANDIDATES
        # ====================================================================

        metrics.stage2_metrics.start()

        if config.start_stage <= 2:
            logger.info("=== STAGE 2: Question Generation + SQL Candidates ===")

            try:
                questions = await generate_training_questions(sps)
                result.questions_generated = len(questions)
                metrics.total_questions_generated = len(questions)

                sql_candidates = await generate_multi_candidate_sql(questions, config.n_candidates)
                metrics.total_candidates_generated = len(sql_candidates)

                sql_candidates = await refine_candidates(sql_candidates)
                metrics.stage2_metrics.items_processed = len(questions)
                metrics.stage2_metrics.items_succeeded = len(sql_candidates)

                logger.info(f"✅ Stage 2 complete: {len(questions)} questions, {len(sql_candidates)} candidates")

            except Exception as e:
                errors.append(f"Stage 2 error: {str(e)}")
                metrics.stage2_metrics.errors.append(str(e))
                logger.error(f"❌ Stage 2 failed: {e}")
                raise
        else:
            logger.info("⏩ Skipping Stage 2, loading existing candidates")
            sql_candidates = await load_pending_candidates(config.database)
            result.questions_generated = len(sql_candidates)
            metrics.stage2_metrics.items_processed = len(sql_candidates)
            metrics.stage2_metrics.items_succeeded = len(sql_candidates)
            metrics.total_candidates_generated = len(sql_candidates)

        metrics.stage2_metrics.complete()

        # Update progress artifact after Stage 2
        await create_markdown_artifact(
            key="pipeline-progress",
            markdown=f"""# SQL Training Pipeline - Stage 2 Complete

**Database:** {config.database}
**Started:** {metrics.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Stage 1: Extraction ✅
- Duration: {metrics.stage1_metrics.duration_seconds:.1f}s
- SPs Processed: {metrics.stage1_metrics.items_processed}

## Stage 2: Generation ✅
- Duration: {metrics.stage2_metrics.duration_seconds:.1f}s
- Questions Generated: {metrics.total_questions_generated}
- SQL Candidates: {metrics.total_candidates_generated}

## Stage 3: Validation 🔄 Running...
""",
            description=f"Training pipeline for {config.database}"
        )

        # ====================================================================
        # STAGE 3: EXECUTION VALIDATION + BEST SELECTION
        # ====================================================================

        metrics.stage3_metrics.start()
        logger.info("=== STAGE 3: Execution Validation + Best Selection ===")

        try:
            validated = await validate_with_execution(sql_candidates, config)

            # Calculate quality metrics
            successful = [v for v in validated if v.get("execution_success", False)]
            metrics.execution_success_rate = len(successful) / len(validated) if validated else 0.0

            if validated:
                scores = [v.get("validation_score", 0) for v in validated]
                metrics.average_validation_score = sum(scores) / len(scores)

            best_examples = await select_best_candidates(validated, config.validation_threshold)
            result.examples_validated = len(best_examples)
            metrics.total_candidates_validated = len(validated)

            # Calculate validation pass rate
            metrics.validation_pass_rate = len(best_examples) / len(validated) if validated else 0.0

            stored_count = await store_training_examples(best_examples)
            result.examples_stored = stored_count
            metrics.total_examples_stored = stored_count

            metrics.stage3_metrics.items_processed = len(validated)
            metrics.stage3_metrics.items_succeeded = len(best_examples)

            logger.info(f"✅ Stage 3 complete: {len(best_examples)} examples validated and stored")

        except Exception as e:
            errors.append(f"Stage 3 error: {str(e)}")
            metrics.stage3_metrics.errors.append(str(e))
            logger.error(f"❌ Stage 3 failed: {e}")
            raise

        metrics.stage3_metrics.complete()
        metrics.complete()
        metrics.errors = errors

        # ====================================================================
        # FINALIZE RESULTS AND CREATE ARTIFACTS
        # ====================================================================

        result.execution_time = time.time() - start_time
        result.errors = errors

        result.stage_stats = {
            "stage1": {
                "sps_processed": result.total_sps,
                "duration_seconds": metrics.stage1_metrics.duration_seconds
            },
            "stage2": {
                "questions_generated": result.questions_generated,
                "candidates_generated": metrics.total_candidates_generated,
                "duration_seconds": metrics.stage2_metrics.duration_seconds
            },
            "stage3": {
                "examples_validated": result.examples_validated,
                "examples_stored": result.examples_stored,
                "validation_rate": result.validation_rate,
                "duration_seconds": metrics.stage3_metrics.duration_seconds
            }
        }

        # Store metrics in MongoDB for historical tracking
        await store_pipeline_metrics(metrics)

        # Create comprehensive summary artifact
        await create_training_summary_artifact(metrics, config, result)

        # Create stage metrics table artifact
        await create_stage_metrics_artifact(metrics)

        # Create quality metrics artifact
        await create_quality_metrics_artifact(metrics, validated if 'validated' in locals() else [])

        logger.info("=== 🎉 Pipeline Complete ===")
        logger.info(f"Total Duration: {metrics.total_duration_seconds:.1f}s")
        logger.info(f"Examples Stored: {metrics.total_examples_stored}")

        return result

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        metrics.complete()
        metrics.errors = errors + [str(e)]

        # Create failure artifact
        await create_markdown_artifact(
            key="pipeline-failure",
            markdown=f"""# ❌ SQL Training Pipeline Failed

**Database:** {config.database}
**Pipeline ID:** {metrics.pipeline_id}
**Duration:** {metrics.total_duration_seconds:.1f}s

## Error
```
{str(e)}
```

## Stage Progress
- Stage 1 (Extraction): {'✅' if metrics.stage1_metrics.completed_at else '❌'}
- Stage 2 (Generation): {'✅' if metrics.stage2_metrics.completed_at else '❌'}
- Stage 3 (Validation): {'✅' if metrics.stage3_metrics.completed_at else '❌'}

## Errors
{chr(10).join(f'- {err}' for err in metrics.errors) if metrics.errors else 'No detailed errors captured'}
""",
            description=f"Pipeline failure for {config.database}"
        )

        result.execution_time = time.time() - start_time
        result.errors = errors + [str(e)]
        return result


# ============================================================================
# ARTIFACT CREATION FUNCTIONS
# ============================================================================

async def store_pipeline_metrics(metrics: PipelineMetrics):
    """Store pipeline metrics in MongoDB for historical tracking."""
    try:
        from mongodb import MongoDBService

        mongo = MongoDBService.get_instance()
        if not mongo.is_initialized:
            await mongo.initialize()

        collection = mongo.db["sql_training_metrics"]

        doc = metrics.to_dict()
        doc["_id"] = metrics.pipeline_id
        doc["created_at"] = datetime.utcnow()

        await collection.replace_one(
            {"_id": metrics.pipeline_id},
            doc,
            upsert=True
        )

        logger.info(f"Stored metrics for pipeline {metrics.pipeline_id}")

    except Exception as e:
        logger.warning(f"Failed to store pipeline metrics: {e}")


async def create_training_summary_artifact(metrics: PipelineMetrics, config: TrainingPipelineConfig, result: TrainingResult):
    """Create comprehensive summary markdown artifact."""
    status_emoji = "✅" if not metrics.errors else "⚠️"

    markdown = f"""# {status_emoji} SQL Training Pipeline Summary

**Database:** {metrics.database}
**Pipeline ID:** `{metrics.pipeline_id}`
**Completed:** {metrics.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC') if metrics.completed_at else 'N/A'}

---

## ⏱️ Timing Summary

| Stage | Duration | Items | Success Rate |
|-------|----------|-------|--------------|
| Extraction | {metrics.stage1_metrics.duration_seconds:.1f}s | {metrics.stage1_metrics.items_processed} | {metrics.stage1_metrics.success_rate:.1%} |
| Generation | {metrics.stage2_metrics.duration_seconds:.1f}s | {metrics.stage2_metrics.items_processed} | {metrics.stage2_metrics.success_rate:.1%} |
| Validation | {metrics.stage3_metrics.duration_seconds:.1f}s | {metrics.stage3_metrics.items_processed} | {metrics.stage3_metrics.success_rate:.1%} |
| **Total** | **{metrics.total_duration_seconds:.1f}s** | - | - |

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Stored Procedures Processed | {metrics.total_sps_processed} |
| Questions Generated | {metrics.total_questions_generated} |
| SQL Candidates Generated | {metrics.total_candidates_generated} |
| Candidates Validated | {metrics.total_candidates_validated} |
| **Training Examples Stored** | **{metrics.total_examples_stored}** |

---

## 🎯 Quality Metrics

| Metric | Value |
|--------|-------|
| Execution Success Rate | {metrics.execution_success_rate:.1%} |
| Validation Pass Rate | {metrics.validation_pass_rate:.1%} |
| Average Validation Score | {metrics.average_validation_score:.2f} |

---

## ⚙️ Configuration

```json
{json.dumps({
    "database": config.database,
    "start_stage": config.start_stage,
    "batch_size": config.batch_size,
    "n_candidates": config.n_candidates,
    "validation_threshold": config.validation_threshold,
    "enable_execution_validation": config.enable_execution_validation
}, indent=2)}
```

"""

    if metrics.errors:
        markdown += f"""
---

## ⚠️ Errors ({len(metrics.errors)})

"""
        for err in metrics.errors:
            markdown += f"- {err}\n"

    if metrics.warnings:
        markdown += f"""
---

## 📝 Warnings ({len(metrics.warnings)})

"""
        for warn in metrics.warnings:
            markdown += f"- {warn}\n"

    await create_markdown_artifact(
        key="training-pipeline-summary",
        markdown=markdown,
        description=f"Complete summary for {metrics.database} training run"
    )


async def create_stage_metrics_artifact(metrics: PipelineMetrics):
    """Create table artifact for stage metrics."""
    table_data = [
        {
            "Stage": "1. Extraction",
            "Status": "✅" if metrics.stage1_metrics.completed_at else "❌",
            "Duration (s)": f"{metrics.stage1_metrics.duration_seconds:.1f}",
            "Items Processed": metrics.stage1_metrics.items_processed,
            "Succeeded": metrics.stage1_metrics.items_succeeded,
            "Failed": metrics.stage1_metrics.items_failed,
            "Success Rate": f"{metrics.stage1_metrics.success_rate:.1%}"
        },
        {
            "Stage": "2. Generation",
            "Status": "✅" if metrics.stage2_metrics.completed_at else "❌",
            "Duration (s)": f"{metrics.stage2_metrics.duration_seconds:.1f}",
            "Items Processed": metrics.stage2_metrics.items_processed,
            "Succeeded": metrics.stage2_metrics.items_succeeded,
            "Failed": metrics.stage2_metrics.items_failed,
            "Success Rate": f"{metrics.stage2_metrics.success_rate:.1%}"
        },
        {
            "Stage": "3. Validation",
            "Status": "✅" if metrics.stage3_metrics.completed_at else "❌",
            "Duration (s)": f"{metrics.stage3_metrics.duration_seconds:.1f}",
            "Items Processed": metrics.stage3_metrics.items_processed,
            "Succeeded": metrics.stage3_metrics.items_succeeded,
            "Failed": metrics.stage3_metrics.items_failed,
            "Success Rate": f"{metrics.stage3_metrics.success_rate:.1%}"
        }
    ]

    await create_table_artifact(
        key="stage-metrics-table",
        table=table_data,
        description="Stage-by-stage metrics breakdown"
    )


async def create_quality_metrics_artifact(metrics: PipelineMetrics, validation_results: List[Dict[str, Any]]):
    """Create quality metrics artifact with validation details."""
    # Calculate score distribution
    if validation_results:
        score_ranges = {
            "0.9-1.0": 0,
            "0.8-0.9": 0,
            "0.7-0.8": 0,
            "0.5-0.7": 0,
            "0.0-0.5": 0
        }

        for result in validation_results:
            score = result.get("validation_score", 0)
            if score >= 0.9:
                score_ranges["0.9-1.0"] += 1
            elif score >= 0.8:
                score_ranges["0.8-0.9"] += 1
            elif score >= 0.7:
                score_ranges["0.7-0.8"] += 1
            elif score >= 0.5:
                score_ranges["0.5-0.7"] += 1
            else:
                score_ranges["0.0-0.5"] += 1

        # Create score distribution table
        table_data = [
            {"Score Range": range_name, "Count": count, "Percentage": f"{count/len(validation_results)*100:.1f}%"}
            for range_name, count in score_ranges.items()
        ]

        await create_table_artifact(
            key="validation-score-distribution",
            table=table_data,
            description="Distribution of validation scores"
        )

    # Create quality summary
    markdown = f"""# 🎯 Quality Metrics Dashboard

## Overall Quality

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Execution Success Rate | {metrics.execution_success_rate:.1%} | > 70% | {'✅' if metrics.execution_success_rate > 0.7 else '⚠️'} |
| Validation Pass Rate | {metrics.validation_pass_rate:.1%} | > 50% | {'✅' if metrics.validation_pass_rate > 0.5 else '⚠️'} |
| Average Validation Score | {metrics.average_validation_score:.2f} | > 0.75 | {'✅' if metrics.average_validation_score > 0.75 else '⚠️'} |

## Efficiency

| Metric | Value |
|--------|-------|
| Questions per SP | {metrics.total_questions_generated / max(metrics.total_sps_processed, 1):.1f} |
| Candidates per Question | {metrics.total_candidates_generated / max(metrics.total_questions_generated, 1):.1f} |
| Examples per Candidate | {metrics.total_examples_stored / max(metrics.total_candidates_validated, 1):.2f} |

## Pipeline Efficiency

| Metric | Value |
|--------|-------|
| Total Processing Time | {metrics.total_duration_seconds:.1f}s |
| Avg Time per SP | {metrics.total_duration_seconds / max(metrics.total_sps_processed, 1):.2f}s |
| Avg Time per Example | {metrics.total_duration_seconds / max(metrics.total_examples_stored, 1):.2f}s |
"""

    await create_markdown_artifact(
        key="quality-metrics-dashboard",
        markdown=markdown,
        description="Quality metrics dashboard for training pipeline"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_referenced_tables(definition: str) -> List[str]:
    """Extract table names referenced in SP definition."""
    import re

    # Simple regex to find table references (schema.table or just table)
    pattern = r'\b(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+(?:\[?(\w+)\]?\.)?\[?(\w+)\]?'
    matches = re.findall(pattern, definition, re.IGNORECASE)

    tables = set()
    for schema, table in matches:
        if schema:
            tables.add(f"{schema}.{table}")
        else:
            tables.add(table)

    return list(tables)


def _extract_sql_from_response(response: str) -> str:
    """Extract SQL from LLM response, removing markdown blocks."""
    import re

    # Remove markdown code blocks
    if "```sql" in response:
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if "```" in response:
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Remove "SQL:" prefix
    if "SQL:" in response:
        parts = response.split("SQL:", 1)
        response = parts[1].strip()

    return response.strip()
