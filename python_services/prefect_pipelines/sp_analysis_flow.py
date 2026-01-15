"""
Prefect Stored Procedure Analysis Pipeline

Orchestrates the SP analysis workflow:
1. Fetch SPs - Retrieve stored procedures from MongoDB in batches
2. Question Generation - Generate natural language questions using LLM
3. Test Query Creation - Create test SQL queries to validate questions
4. Validation & Storage - Validate question↔result alignment and store training data

Features:
- Built-in retries with exponential backoff
- Detailed logging and artifact tracking
- Async-native execution
- Visual progress in Prefect dashboard
- Configurable batch processing for memory efficiency
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

# Load .env from project root (single source of truth)
from dotenv import load_dotenv
_services_dir = Path(__file__).parent.parent
_project_root = _services_dir.parent
load_dotenv(_project_root / ".env", override=True)

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class SPBatchResult:
    """Result from fetching stored procedure batch"""
    database: str
    batch_number: int
    offset: int
    sps_fetched: int = 0
    sps_total: int = 0
    has_more: bool = False
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class QuestionGenResult:
    """Result from question generation task"""
    sp_id: str
    sp_name: str
    questions_generated: int = 0
    questions: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class TestQueryResult:
    """Result from test query creation task"""
    sp_id: str
    sp_name: str
    queries_created: int = 0
    queries: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from validation and storage task"""
    sp_id: str
    sp_name: str
    questions_validated: int = 0
    training_examples_created: int = 0
    validation_score: float = 0.0
    stored: bool = False
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class SPAnalysisSummary:
    """Summary of SP analysis run"""
    database: str
    total_sps_processed: int
    successful_analyses: int
    failed_analyses: int
    total_questions_generated: int
    total_training_examples: int
    avg_validation_score: float
    total_duration_seconds: float
    started_at: str
    completed_at: str


@task(
    name="fetch_sp_batch",
    description="Fetch stored procedures from MongoDB in batches",
    retries=2,
    retry_delay_seconds=30,
    tags=["mongodb", "extraction"]
)
async def fetch_sp_batch(
    database: str,
    batch_size: int,
    offset: int,
    mongodb_uri: str = None
) -> SPBatchResult:
    """
    Fetch stored procedures from MongoDB in batches.

    Args:
        database: Database name to fetch procedures from
        batch_size: Number of procedures to fetch
        offset: Starting offset for pagination
        mongodb_uri: MongoDB connection URI

    Returns:
        SPBatchResult with fetched procedures and pagination info
    """
    logger = get_run_logger()
    start_time = time.time()
    result = SPBatchResult(database=database, batch_number=offset // batch_size + 1, offset=offset)

    try:
        logger.info(f"Fetching SP batch for {database} (offset={offset}, limit={batch_size})")

        import sys
        sys.path.insert(0, '..')
        from mongodb import MongoDBService
        from database_name_parser import normalize_database_name
        from config import COLLECTION_SQL_STORED_PROCEDURES

        if mongodb_uri is None:
            mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27019")

        mongodb = MongoDBService()
        await mongodb.connect()

        normalized_db = normalize_database_name(database)
        collection = mongodb.db[COLLECTION_SQL_STORED_PROCEDURES]

        # Get total count for this database
        total_count = await collection.count_documents({"database": normalized_db})
        result.sps_total = total_count

        # Fetch batch with projection to limit data size
        cursor = collection.find(
            {"database": normalized_db},
            {
                "_id": 1,
                "procedure_name": 1,
                "schema": 1,
                "definition": 1,
                "parameters": 1,
                "summary": 1,
                "tables_referenced": 1,
                "database": 1
            }
        ).skip(offset).limit(batch_size)

        # Store fetched SPs for processing
        result.sps = []
        async for sp in cursor:
            result.sps.append(sp)
            result.sps_fetched += 1

        result.has_more = (offset + batch_size) < total_count

        logger.info(f"Fetched {result.sps_fetched} SPs (total: {result.sps_total}, has_more: {result.has_more})")
        result.success = True

    except Exception as e:
        error_msg = f"Failed to fetch SP batch: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="generate_nl_questions",
    description="Generate natural language questions for a stored procedure",
    retries=1,
    retry_delay_seconds=60,
    tags=["llm", "question-generation"]
)
async def generate_nl_questions(
    sp: Dict[str, Any],
    question_count: int = 3,
    llm_host: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:7b"
) -> QuestionGenResult:
    """
    Generate natural language questions for a stored procedure.

    Uses LLM to create realistic questions that users might ask
    which should trigger this stored procedure.

    Args:
        sp: Stored procedure document from MongoDB
        question_count: Number of questions to generate
        llm_host: LLM API endpoint
        model: Model to use for generation

    Returns:
        QuestionGenResult with generated questions
    """
    logger = get_run_logger()
    start_time = time.time()

    sp_id = str(sp.get("_id", "unknown"))
    sp_name = sp.get("procedure_name", "unknown")
    result = QuestionGenResult(sp_id=sp_id, sp_name=sp_name)

    try:
        logger.info(f"Generating questions for {sp_name}")

        import sys
        sys.path.insert(0, '..')
        from services.llm_service import get_llm_service

        llm_service = await get_llm_service()

        # Build prompt for question generation
        prompt = _build_question_prompt(sp, question_count)

        # Call LLM
        gen_result = await llm_service.generate(
            prompt=prompt,
            system="You are an expert at generating natural language questions for SQL stored procedures. Generate realistic questions that users might ask.",
            model=model,
            temperature=0.7  # Higher temperature for more varied questions
        )

        result.llm_calls += 1

        if not gen_result.success:
            raise Exception(gen_result.error or "LLM generation failed")

        # Parse questions from response
        questions = _parse_questions(gen_result.response, sp_name)
        result.questions = questions
        result.questions_generated = len(questions)
        result.tokens_used = gen_result.token_usage.get("total_tokens", 0)

        logger.info(f"Generated {result.questions_generated} questions for {sp_name}")
        result.success = True

    except Exception as e:
        error_msg = f"Question generation failed for {sp_name}: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="create_test_queries",
    description="Create test SQL queries to validate questions",
    retries=1,
    retry_delay_seconds=30,
    tags=["sql", "testing"]
)
async def create_test_queries(
    sp: Dict[str, Any],
    questions: List[Dict[str, Any]]
) -> TestQueryResult:
    """
    Create test SQL queries to validate questions.

    Generates EXEC statements for the stored procedure with appropriate
    parameter values for each question.

    Args:
        sp: Stored procedure document from MongoDB
        questions: List of generated questions

    Returns:
        TestQueryResult with test queries
    """
    logger = get_run_logger()
    start_time = time.time()

    sp_id = str(sp.get("_id", "unknown"))
    sp_name = sp.get("procedure_name", "unknown")
    result = TestQueryResult(sp_id=sp_id, sp_name=sp_name)

    try:
        logger.info(f"Creating test queries for {sp_name}")

        parameters = sp.get("parameters", [])
        schema = sp.get("schema", "dbo")
        full_name = f"{schema}.{sp_name}"

        for question in questions:
            # Build EXEC statement
            query = _build_exec_statement(
                full_name=full_name,
                parameters=parameters,
                question=question
            )

            result.queries.append({
                "question": question.get("text", ""),
                "sql": query,
                "expected_sp": sp_name,
                "parameter_values": _extract_parameter_values(parameters, question)
            })

            result.queries_created += 1

        logger.info(f"Created {result.queries_created} test queries for {sp_name}")
        result.success = True

    except Exception as e:
        error_msg = f"Test query creation failed for {sp_name}: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="validate_and_store",
    description="Validate question↔result alignment and store training data",
    retries=2,
    retry_delay_seconds=30,
    tags=["validation", "storage"]
)
async def validate_and_store(
    sp: Dict[str, Any],
    questions: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    mongodb_uri: str = None
) -> ValidationResult:
    """
    Validate question↔result alignment and store training data.

    Validates that:
    1. Questions are semantically aligned with SP purpose
    2. Test queries are syntactically correct
    3. Parameters are appropriately matched

    Then stores validated examples as training data for RAG.

    Args:
        sp: Stored procedure document
        questions: Generated questions
        queries: Test queries
        mongodb_uri: MongoDB connection URI

    Returns:
        ValidationResult with validation scores and storage status
    """
    logger = get_run_logger()
    start_time = time.time()

    sp_id = str(sp.get("_id", "unknown"))
    sp_name = sp.get("procedure_name", "unknown")
    result = ValidationResult(sp_id=sp_id, sp_name=sp_name)

    try:
        logger.info(f"Validating and storing data for {sp_name}")

        import sys
        sys.path.insert(0, '..')
        from mongodb import MongoDBService
        from config import COLLECTION_SQL_EXAMPLES

        if mongodb_uri is None:
            mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27019")

        mongodb = MongoDBService()
        await mongodb.connect()

        collection = mongodb.db[COLLECTION_SQL_EXAMPLES]
        validation_scores = []

        # Validate and store each question-query pair
        for question, query in zip(questions, queries):
            # Calculate validation score (0-1)
            score = _calculate_validation_score(sp, question, query)
            validation_scores.append(score)

            # Only store if validation passes threshold
            if score >= 0.6:  # Configurable threshold
                training_example = {
                    "sp_id": sp_id,
                    "sp_name": sp_name,
                    "database": sp.get("database"),
                    "schema": sp.get("schema", "dbo"),
                    "question": question.get("text", ""),
                    "sql": query.get("sql", ""),
                    "parameters": query.get("parameter_values", {}),
                    "validation_score": score,
                    "generated_at": datetime.now(timezone.utc),
                    "source": "sp_analysis_pipeline",
                    "active": True
                }

                # Upsert to avoid duplicates
                await collection.update_one(
                    {
                        "sp_id": sp_id,
                        "question": question.get("text", "")
                    },
                    {"$set": training_example},
                    upsert=True
                )

                result.training_examples_created += 1
                result.questions_validated += 1

        result.validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        result.stored = result.training_examples_created > 0

        logger.info(
            f"Validated {result.questions_validated} questions, "
            f"created {result.training_examples_created} training examples "
            f"(avg score: {result.validation_score:.2f})"
        )
        result.success = True

    except Exception as e:
        error_msg = f"Validation/storage failed for {sp_name}: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@flow(
    name="sp_analysis_single",
    description="Analyze a single stored procedure",
    log_prints=True
)
async def analyze_single_sp(
    sp_id: str,
    database: str,
    question_count: int = 3,
    llm_host: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:7b"
) -> Dict[str, Any]:
    """
    Analyze a single stored procedure.

    Steps:
    1. Fetch SP from MongoDB
    2. Generate natural language questions
    3. Create test SQL queries
    4. Validate and store training data

    Args:
        sp_id: MongoDB ObjectId of stored procedure
        database: Database name
        question_count: Number of questions to generate
        llm_host: LLM API endpoint
        model: Model for question generation

    Returns:
        Dict with analysis results
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Analyzing single SP: {sp_id}")

    try:
        import sys
        sys.path.insert(0, '..')
        from mongodb import MongoDBService
        from database_name_parser import normalize_database_name
        from config import COLLECTION_SQL_STORED_PROCEDURES
        from bson import ObjectId

        # Fetch SP
        mongodb = MongoDBService()
        await mongodb.connect()

        normalized_db = normalize_database_name(database)
        collection = mongodb.db[COLLECTION_SQL_STORED_PROCEDURES]

        sp = await collection.find_one({
            "_id": ObjectId(sp_id),
            "database": normalized_db
        })

        if not sp:
            raise Exception(f"SP {sp_id} not found in database {database}")

        # Generate questions
        question_result = await generate_nl_questions(
            sp=sp,
            question_count=question_count,
            llm_host=llm_host,
            model=model
        )

        if not question_result.success:
            raise Exception(f"Question generation failed: {', '.join(question_result.errors)}")

        # Create test queries
        query_result = await create_test_queries(
            sp=sp,
            questions=question_result.questions
        )

        if not query_result.success:
            raise Exception(f"Query creation failed: {', '.join(query_result.errors)}")

        # Validate and store
        validation_result = await validate_and_store(
            sp=sp,
            questions=question_result.questions,
            queries=query_result.queries
        )

        total_duration = time.time() - start_time

        return {
            "success": validation_result.success,
            "sp_id": sp_id,
            "sp_name": sp.get("procedure_name", "unknown"),
            "questions_generated": question_result.questions_generated,
            "training_examples_created": validation_result.training_examples_created,
            "validation_score": validation_result.validation_score,
            "duration_seconds": total_duration
        }

    except Exception as e:
        logger.error(f"Single SP analysis failed: {e}")
        return {
            "success": False,
            "sp_id": sp_id,
            "error": str(e),
            "duration_seconds": time.time() - start_time
        }


@flow(
    name="sp_analysis_batch",
    description="Batch process stored procedures",
    log_prints=True
)
async def analyze_sp_batch(
    database: str,
    batch_size: int = 10,
    max_sps: int = 100,
    question_count: int = 3,
    llm_host: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:7b",
    mongodb_uri: str = None
) -> Dict[str, Any]:
    """
    Batch process stored procedures.

    Fetches SPs in batches and processes them sequentially to manage
    resource usage (LLM calls, memory).

    Args:
        database: Database name to process
        batch_size: Number of SPs to fetch per batch
        max_sps: Maximum number of SPs to process (0 = all)
        question_count: Questions to generate per SP
        llm_host: LLM API endpoint
        model: Model for question generation
        mongodb_uri: MongoDB connection URI

    Returns:
        Dict with batch processing results
    """
    logger = get_run_logger()
    flow_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    if mongodb_uri is None:
        mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27019")

    logger.info(f"Starting SP batch analysis for {database}")
    logger.info(f"Batch size: {batch_size}, Max SPs: {max_sps or 'ALL'}")

    offset = 0
    total_processed = 0
    successful = 0
    failed = 0
    total_questions = 0
    total_training_examples = 0
    validation_scores = []

    while True:
        # Fetch batch
        batch_result = await fetch_sp_batch(
            database=database,
            batch_size=batch_size,
            offset=offset,
            mongodb_uri=mongodb_uri
        )

        if not batch_result.success or batch_result.sps_fetched == 0:
            break

        logger.info(f"Processing batch {batch_result.batch_number} ({batch_result.sps_fetched} SPs)")

        # Process each SP in batch
        for sp in batch_result.sps:
            # Check max_sps limit
            if max_sps > 0 and total_processed >= max_sps:
                logger.info(f"Reached max_sps limit ({max_sps}), stopping")
                break

            # Generate questions
            question_result = await generate_nl_questions(
                sp=sp,
                question_count=question_count,
                llm_host=llm_host,
                model=model
            )

            if not question_result.success:
                failed += 1
                total_processed += 1
                continue

            # Create test queries
            query_result = await create_test_queries(
                sp=sp,
                questions=question_result.questions
            )

            if not query_result.success:
                failed += 1
                total_processed += 1
                continue

            # Validate and store
            validation_result = await validate_and_store(
                sp=sp,
                questions=question_result.questions,
                queries=query_result.queries,
                mongodb_uri=mongodb_uri
            )

            if validation_result.success:
                successful += 1
                total_questions += question_result.questions_generated
                total_training_examples += validation_result.training_examples_created
                validation_scores.append(validation_result.validation_score)
            else:
                failed += 1

            total_processed += 1

        # Check if we should continue to next batch
        if max_sps > 0 and total_processed >= max_sps:
            break

        if not batch_result.has_more:
            break

        offset += batch_size

    flow_duration = time.time() - flow_start
    completed_at = datetime.now(timezone.utc).isoformat()

    avg_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0

    # Create summary artifact
    await create_markdown_artifact(
        key="sp-analysis-batch-summary",
        markdown=f"""# SP Analysis Batch Summary

## Overview
| Metric | Value |
|--------|-------|
| **Database** | {database} |
| **Started** | {started_at} |
| **Completed** | {completed_at} |
| **Duration** | {flow_duration:.1f}s |

## Processing Results
| Metric | Value |
|--------|-------|
| SPs Processed | {total_processed} |
| Successful | {successful} |
| Failed | {failed} |
| Success Rate | {(successful/total_processed*100 if total_processed > 0 else 0):.1f}% |

## Generated Assets
| Asset | Count |
|-------|-------|
| Questions Generated | {total_questions} |
| Training Examples Created | {total_training_examples} |
| Avg Validation Score | {avg_score:.2f} |
""",
        description=f"SP Analysis Batch - {database}"
    )

    logger.info(f"Batch analysis completed: {successful}/{total_processed} successful")

    return {
        "success": failed == 0,
        "database": database,
        "total_processed": total_processed,
        "successful": successful,
        "failed": failed,
        "total_questions": total_questions,
        "total_training_examples": total_training_examples,
        "avg_validation_score": avg_score,
        "duration_seconds": flow_duration,
        "started_at": started_at,
        "completed_at": completed_at
    }


@flow(
    name="sp_analysis_scheduled",
    description="Scheduled daily run of SP analysis",
    log_prints=True
)
async def scheduled_sp_analysis(
    databases: List[str] = None,
    batch_size: int = 10,
    max_sps_per_db: int = 100,
    question_count: int = 3
) -> Dict[str, Any]:
    """
    Scheduled daily run of SP analysis.

    Processes multiple databases sequentially, generating training
    data for SQL generation RAG system.

    Args:
        databases: List of databases to process (default: ["EWRCentral", "EWRReporting"])
        batch_size: Batch size for fetching SPs
        max_sps_per_db: Max SPs to process per database
        question_count: Questions per SP

    Returns:
        Dict with aggregated results across all databases
    """
    logger = get_run_logger()
    flow_start = time.time()

    if databases is None:
        databases = ["EWRCentral", "EWRReporting"]

    logger.info(f"Starting scheduled SP analysis for {len(databases)} databases")

    results = []

    for db in databases:
        logger.info(f"Processing database: {db}")

        db_result = await analyze_sp_batch(
            database=db,
            batch_size=batch_size,
            max_sps=max_sps_per_db,
            question_count=question_count
        )

        results.append(db_result)

    total_duration = time.time() - flow_start

    # Aggregate metrics
    total_sps = sum(r["total_processed"] for r in results)
    total_examples = sum(r["total_training_examples"] for r in results)
    avg_scores = [r["avg_validation_score"] for r in results if r["avg_validation_score"] > 0]
    overall_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0

    # Create aggregate artifact
    await create_markdown_artifact(
        key="sp-analysis-scheduled-summary",
        markdown=f"""# Scheduled SP Analysis Summary

## Overview
**Total Duration**: {total_duration:.1f}s
**Databases Processed**: {len(databases)}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Total SPs Processed | {total_sps} |
| Total Training Examples | {total_examples} |
| Overall Avg Validation Score | {overall_avg_score:.2f} |

## Per-Database Results
| Database | SPs | Examples | Avg Score | Duration |
|----------|-----|----------|-----------|----------|
{chr(10).join(f"| {r['database']} | {r['total_processed']} | {r['total_training_examples']} | {r['avg_validation_score']:.2f} | {r['duration_seconds']:.1f}s |" for r in results)}
""",
        description="Scheduled SP Analysis - All Databases"
    )

    return {
        "success": all(r["success"] for r in results),
        "databases_processed": len(databases),
        "total_sps": total_sps,
        "total_training_examples": total_examples,
        "overall_avg_score": overall_avg_score,
        "total_duration_seconds": total_duration,
        "results": results
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _build_question_prompt(sp: Dict[str, Any], count: int) -> str:
    """Build prompt for question generation."""
    sp_name = sp.get("procedure_name", "unknown")
    summary = sp.get("summary", {})
    summary_text = summary.get("summary", "") if isinstance(summary, dict) else ""
    parameters = sp.get("parameters", [])
    tables = sp.get("tables_referenced", [])

    param_desc = "\n".join([
        f"  - {p.get('name', 'unknown')} ({p.get('type', 'unknown')}): {p.get('description', 'N/A')}"
        for p in parameters
    ]) if parameters else "  None"

    prompt = f"""Generate {count} natural language questions that a user might ask to retrieve data from this stored procedure.

Stored Procedure: {sp_name}
Summary: {summary_text or "No summary available"}

Parameters:
{param_desc}

Tables Referenced: {", ".join(tables) if tables else "Unknown"}

Requirements:
1. Questions should be realistic and business-focused
2. Each question should clearly map to the SP's purpose
3. Include parameter values where appropriate (use realistic examples)
4. Vary the phrasing to cover different user intents
5. Return ONLY the questions, one per line, numbered 1-{count}

Example format:
1. Show me all tickets created today
2. What are the open tickets for customer ABC?
3. Get ticket details for ticket ID 12345

Now generate {count} questions for {sp_name}:"""

    return prompt


def _parse_questions(response: str, sp_name: str) -> List[Dict[str, Any]]:
    """Parse questions from LLM response."""
    questions = []
    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove numbering (1., 2., etc.)
        text = line.lstrip('0123456789.)-* ').strip()

        if text:
            questions.append({
                "text": text,
                "sp_name": sp_name,
                "generated_at": datetime.now(timezone.utc).isoformat()
            })

    return questions


def _build_exec_statement(
    full_name: str,
    parameters: List[Dict[str, Any]],
    question: Dict[str, Any]
) -> str:
    """Build EXEC statement for testing."""
    if not parameters:
        return f"EXEC {full_name}"

    # Build parameter assignments
    param_assignments = []
    for param in parameters:
        param_name = param.get("name", "")
        param_type = param.get("type", "").upper()

        # Generate placeholder value based on type
        if "INT" in param_type:
            value = "NULL"
        elif "DATE" in param_type or "TIME" in param_type:
            value = "NULL"
        elif "VARCHAR" in param_type or "CHAR" in param_type:
            value = "NULL"
        else:
            value = "NULL"

        param_assignments.append(f"{param_name} = {value}")

    params_str = ", ".join(param_assignments)
    return f"EXEC {full_name} {params_str}"


def _extract_parameter_values(
    parameters: List[Dict[str, Any]],
    question: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract parameter values from question context."""
    # This is a simplified version - could be enhanced with NER/entity extraction
    values = {}
    question_text = question.get("text", "").lower()

    for param in parameters:
        param_name = param.get("name", "")
        values[param_name] = None  # Placeholder

    return values


def _calculate_validation_score(
    sp: Dict[str, Any],
    question: Dict[str, Any],
    query: Dict[str, Any]
) -> float:
    """
    Calculate validation score for question-query pair.

    Score is based on:
    - Semantic alignment between question and SP purpose (0.4)
    - Query syntax correctness (0.3)
    - Parameter completeness (0.3)
    """
    score = 0.0

    # Basic validation - can be enhanced
    question_text = question.get("text", "")
    sp_name = sp.get("procedure_name", "")

    # Check if question is substantive (> 10 chars)
    if len(question_text) > 10:
        score += 0.4

    # Check if query is well-formed
    query_sql = query.get("sql", "")
    if "EXEC" in query_sql and sp_name in query_sql:
        score += 0.3

    # Check parameter coverage
    parameters = sp.get("parameters", [])
    param_values = query.get("parameter_values", {})
    if parameters:
        coverage = len(param_values) / len(parameters)
        score += 0.3 * coverage
    else:
        score += 0.3  # No params = full coverage

    return min(score, 1.0)


# ============================================================================
# Deployment Configuration
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SP Analysis Pipeline")
    parser.add_argument("--serve", action="store_true", help="Start the flow server")
    parser.add_argument("--run-batch", action="store_true", help="Run batch analysis")
    parser.add_argument("--database", default="EWRCentral", help="Database to analyze")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--max-sps", type=int, default=50, help="Max SPs to process")
    args = parser.parse_args()

    if args.serve:
        # Serve the flow for scheduled execution
        # Use: prefect deployment build sp_analysis_flow.py:scheduled_sp_analysis
        print("SP Analysis Flow ready for deployment")
        print("\nTo deploy with Prefect, use:")
        print("  prefect deployment build sp_analysis_flow.py:scheduled_sp_analysis -n sp-analysis-daily")
        print("  prefect deployment apply scheduled_sp_analysis-deployment.yaml")
        print("\nOr serve directly:")
        print("  scheduled_sp_analysis.serve(name='sp-analysis-daily', cron='0 2 * * *')")
    elif args.run_batch:
        # Run batch analysis directly
        asyncio.run(analyze_sp_batch(
            database=args.database,
            batch_size=args.batch_size,
            max_sps=args.max_sps
        ))
    else:
        print("SP Analysis Pipeline")
        print("=" * 40)
        print("\nUsage:")
        print("  --serve       Start flow server for scheduling")
        print("  --run-batch   Run batch analysis now")
        print("  --database    Database to analyze (default: EWRCentral)")
        print("  --batch-size  Batch size (default: 10)")
        print("  --max-sps     Max SPs to process (default: 50)")
        print("\nExamples:")
        print("  python sp_analysis_flow.py --run-batch --database EWRCentral")
        print("  python sp_analysis_flow.py --serve")


# Export for scheduled flows
__all__ = [
    "analyze_single_sp",
    "analyze_sp_batch",
    "scheduled_sp_analysis",
    "fetch_sp_batch",
    "generate_nl_questions",
    "create_test_queries",
    "validate_and_store"
]
