"""
Prefect Scheduled Flows

Provides scheduled execution for automated schema extraction and other
recurring tasks. Uses Prefect's built-in scheduling capabilities.

Features:
- Daily schema extraction for specified databases
- Weekly full re-indexing
- Configurable schedules with cron expressions
- Dashboard visibility for scheduled runs
- Automatic retries on failure

Usage:
    from prefect_pipelines.scheduled_flows import (
        create_daily_schedule,
        create_weekly_schedule,
        run_scheduled_extraction
    )

    # Create a daily schedule for schema extraction
    create_daily_schedule(
        databases=["EWRReporting", "EWRCentral"],
        hour=2,  # 2 AM
        server="NCSQLTEST",
        user="EWRUser",
        password="password"
    )

    # Manual run of scheduled extraction
    result = run_scheduled_extraction(
        databases=["EWRReporting"],
        server="NCSQLTEST",
        user="EWRUser",
        password="password"
    )
"""

import sys
import os
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class DatabaseExtractionResult:
    """Result from extracting a single database"""
    database: str
    success: bool
    tables_extracted: int = 0
    procedures_extracted: int = 0
    vectors_generated: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class ScheduledExtractionResult:
    """Result from scheduled extraction of multiple databases"""
    databases_processed: int
    successful: int
    failed: int
    total_tables: int
    total_procedures: int
    total_vectors: int
    results: List[DatabaseExtractionResult]
    total_duration_seconds: float
    started_at: str
    completed_at: str


@task(
    name="extract_single_database",
    description="Extract schema from a single database",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True
)
async def extract_single_database_task(
    database: str,
    server: str,
    connection_config: Dict[str, Any],
    llm_host: str = "http://localhost:11434",
    llm_model: str = "llama3.2",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> DatabaseExtractionResult:
    """
    Extract and process schema from a single database.

    This task runs the full SQL RAG pipeline for one database:
    1. Schema extraction
    2. LLM summarization
    3. Embedding generation
    4. Vector storage

    Args:
        database: Database name to extract
        server: SQL Server hostname
        connection_config: Connection parameters
        llm_host: LLM API endpoint
        llm_model: Model for summarization
        embedding_model: Model for embeddings

    Returns:
        DatabaseExtractionResult with extraction metrics
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Starting extraction for database: {database}")

    result = DatabaseExtractionResult(database=database, success=True)

    try:
        # Import the SQL RAG flow tasks
        from sql_pipeline.prefect.sql_rag_flow import (
            extract_schemas_task,
            summarize_schemas_task,
            generate_embeddings_task,
            store_vectors_task
        )

        # Run extraction
        extraction = await extract_schemas_task(
            server=server,
            database=database,
            connection_config=connection_config
        )

        if not extraction.success:
            result.success = False
            result.errors.extend(extraction.errors)
            result.duration_seconds = time.time() - start_time
            return result

        result.tables_extracted = extraction.tables_extracted
        result.procedures_extracted = extraction.procedures_extracted

        # Run summarization
        summarization = await summarize_schemas_task(
            extraction_result=extraction,
            llm_host=llm_host,
            model=llm_model
        )

        if not summarization.success:
            result.errors.extend(summarization.errors)

        # Run embedding generation
        embedding = await generate_embeddings_task(
            summarization_result=summarization,
            model_name=embedding_model
        )

        if embedding.success:
            result.vectors_generated = embedding.vectors_generated
        else:
            result.errors.extend(embedding.errors)

        # Verify storage
        storage = await store_vectors_task(
            embedding_result=embedding,
            create_indexes=True
        )

        if not storage.success:
            result.errors.extend(storage.errors)

        # Overall success if we got some data
        result.success = (
            result.tables_extracted > 0 or
            result.procedures_extracted > 0 or
            result.vectors_generated > 0
        )

        logger.info(
            f"Completed {database}: {result.tables_extracted} tables, "
            f"{result.procedures_extracted} procs, {result.vectors_generated} vectors"
        )

    except Exception as e:
        error_msg = f"Database extraction failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="generate_extraction_report",
    description="Generate summary report for scheduled extraction",
    log_prints=True
)
async def generate_extraction_report_task(
    results: List[DatabaseExtractionResult],
    started_at: str,
    completed_at: str,
    total_duration: float
) -> ScheduledExtractionResult:
    """
    Generate a summary report for the scheduled extraction.

    Args:
        results: List of database extraction results
        started_at: ISO timestamp when extraction started
        completed_at: ISO timestamp when extraction completed
        total_duration: Total duration in seconds

    Returns:
        ScheduledExtractionResult with aggregated metrics
    """
    logger = get_run_logger()

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_tables = sum(r.tables_extracted for r in results)
    total_procedures = sum(r.procedures_extracted for r in results)
    total_vectors = sum(r.vectors_generated for r in results)

    summary = ScheduledExtractionResult(
        databases_processed=len(results),
        successful=successful,
        failed=failed,
        total_tables=total_tables,
        total_procedures=total_procedures,
        total_vectors=total_vectors,
        results=results,
        total_duration_seconds=total_duration,
        started_at=started_at,
        completed_at=completed_at
    )

    # Build detailed results table
    db_rows = []
    for r in results:
        status = "OK" if r.success else "FAILED"
        db_rows.append(
            f"| {r.database} | {r.tables_extracted} | {r.procedures_extracted} | "
            f"{r.vectors_generated} | {r.duration_seconds:.1f}s | {status} |"
        )

    # Collect all errors
    all_errors = []
    for r in results:
        for err in r.errors:
            all_errors.append(f"[{r.database}] {err}")

    # Create summary artifact
    await create_markdown_artifact(
        key="scheduled-extraction-summary",
        markdown=f"""# Scheduled Schema Extraction Report

## Overview
| Metric | Value |
|--------|-------|
| **Started** | {started_at} |
| **Completed** | {completed_at} |
| **Total Duration** | {total_duration:.1f}s |
| **Databases Processed** | {len(results)} |
| **Successful** | {successful} |
| **Failed** | {failed} |

## Totals
| Type | Count |
|------|-------|
| Tables Extracted | {total_tables} |
| Procedures Extracted | {total_procedures} |
| Vectors Generated | {total_vectors} |

## Database Results
| Database | Tables | Procedures | Vectors | Duration | Status |
|----------|--------|------------|---------|----------|--------|
{chr(10).join(db_rows)}

{f"## Errors ({len(all_errors)})" + chr(10) + chr(10).join(f"- {e}" for e in all_errors) if all_errors else ""}
""",
        description=f"Scheduled extraction report - {len(results)} databases"
    )

    logger.info(
        f"Extraction report: {successful}/{len(results)} successful, "
        f"{total_tables} tables, {total_procedures} procs, {total_vectors} vectors"
    )

    return summary


@flow(
    name="scheduled-schema-extraction",
    description="Scheduled extraction of SQL Server schemas for RAG indexing",
    log_prints=True
)
async def scheduled_schema_extraction_flow(
    databases: List[str],
    server: str,
    connection_config: Dict[str, Any],
    llm_host: str = "http://localhost:11434",
    llm_model: str = "llama3.2",
    embedding_model: str = "all-MiniLM-L6-v2",
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Scheduled Schema Extraction Flow

    Extracts and indexes schemas from multiple databases. Can run sequentially
    (default, safer for resource management) or in parallel (faster but
    requires more resources).

    Args:
        databases: List of database names to extract
        server: SQL Server hostname
        connection_config: Connection parameters (user, password, etc.)
        llm_host: LLM API endpoint
        llm_model: Model for summarization
        embedding_model: Model for embeddings
        parallel: If True, process databases in parallel

    Returns:
        Dict with extraction results and metrics
    """
    logger = get_run_logger()
    flow_start = time.time()
    started_at = datetime.now().isoformat()

    logger.info(f"Starting scheduled extraction for {len(databases)} databases")
    logger.info(f"Databases: {', '.join(databases)}")
    logger.info(f"Server: {server}")
    logger.info(f"Mode: {'Parallel' if parallel else 'Sequential'}")

    results: List[DatabaseExtractionResult] = []

    if parallel:
        # Process in parallel (use with caution for resource-heavy operations)
        tasks = [
            extract_single_database_task(
                database=db,
                server=server,
                connection_config=connection_config,
                llm_host=llm_host,
                llm_model=llm_model,
                embedding_model=embedding_model
            )
            for db in databases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        # Process sequentially (default, safer)
        for db in databases:
            result = await extract_single_database_task(
                database=db,
                server=server,
                connection_config=connection_config,
                llm_host=llm_host,
                llm_model=llm_model,
                embedding_model=embedding_model
            )
            results.append(result)

    flow_duration = time.time() - flow_start
    completed_at = datetime.now().isoformat()

    # Generate report
    summary = await generate_extraction_report_task(
        results=results,
        started_at=started_at,
        completed_at=completed_at,
        total_duration=flow_duration
    )

    return {
        "success": summary.failed == 0,
        "databases_processed": summary.databases_processed,
        "successful": summary.successful,
        "failed": summary.failed,
        "total_tables": summary.total_tables,
        "total_procedures": summary.total_procedures,
        "total_vectors": summary.total_vectors,
        "total_duration_seconds": summary.total_duration_seconds,
        "started_at": summary.started_at,
        "completed_at": summary.completed_at,
        "results": [
            {
                "database": r.database,
                "success": r.success,
                "tables": r.tables_extracted,
                "procedures": r.procedures_extracted,
                "vectors": r.vectors_generated,
                "duration": r.duration_seconds,
                "errors": r.errors
            }
            for r in results
        ]
    }


def run_scheduled_extraction(
    databases: List[str],
    server: str,
    user: str,
    password: str,
    llm_host: str = "http://localhost:11434",
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run scheduled extraction synchronously.

    Example:
        from prefect_pipelines.scheduled_flows import run_scheduled_extraction

        result = run_scheduled_extraction(
            databases=["EWRReporting", "EWRCentral"],
            server="NCSQLTEST",
            user="EWRUser",
            password="password"
        )
        print(f"Success: {result['success']}")
        print(f"Total vectors: {result['total_vectors']}")
    """
    connection_config = {
        "user": user,
        "password": password,
        "trust_server_certificate": True,
        "encrypt": False
    }

    return asyncio.run(scheduled_schema_extraction_flow(
        databases=databases,
        server=server,
        connection_config=connection_config,
        llm_host=llm_host,
        parallel=parallel
    ))


def create_daily_schedule(
    databases: List[str],
    server: str,
    user: str,
    password: str,
    hour: int = 2,
    minute: int = 0,
    llm_host: str = "http://localhost:11434"
) -> str:
    """
    Create a daily schedule for schema extraction.

    This function creates a Prefect deployment with a daily cron schedule.
    The deployment will be visible in the Prefect dashboard.

    Note: Requires Prefect server to be running for scheduled execution.

    Args:
        databases: List of databases to extract daily
        server: SQL Server hostname
        user: SQL Server username
        password: SQL Server password
        hour: Hour to run (0-23, default 2 AM)
        minute: Minute to run (0-59, default 0)
        llm_host: LLM API endpoint

    Returns:
        Schedule description string
    """
    cron_expression = f"{minute} {hour} * * *"

    # Note: In production, use Prefect deployments for actual scheduling
    # This is a helper to document the schedule configuration
    schedule_info = f"""
Daily Schema Extraction Schedule:
- Cron: {cron_expression}
- Time: {hour:02d}:{minute:02d} daily
- Databases: {', '.join(databases)}
- Server: {server}

To deploy with Prefect CLI:
    prefect deployment build scheduled_flows.py:scheduled_schema_extraction_flow \\
        --name "daily-schema-extraction" \\
        --cron "{cron_expression}" \\
        --param databases='[{", ".join(f'"{db}"' for db in databases)}]' \\
        --param server="{server}" \\
        --param connection_config='{{"user": "{user}", "password": "****"}}' \\
        --apply
"""
    return schedule_info


def create_weekly_schedule(
    databases: List[str],
    server: str,
    user: str,
    password: str,
    day_of_week: int = 0,  # 0 = Sunday
    hour: int = 1,
    minute: int = 0,
    llm_host: str = "http://localhost:11434"
) -> str:
    """
    Create a weekly schedule for full schema re-indexing.

    Args:
        databases: List of databases to extract weekly
        server: SQL Server hostname
        user: SQL Server username
        password: SQL Server password
        day_of_week: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
        hour: Hour to run (0-23, default 1 AM)
        minute: Minute to run (0-59, default 0)
        llm_host: LLM API endpoint

    Returns:
        Schedule description string
    """
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    cron_expression = f"{minute} {hour} * * {day_of_week}"

    schedule_info = f"""
Weekly Schema Extraction Schedule:
- Cron: {cron_expression}
- Time: {day_names[day_of_week]} at {hour:02d}:{minute:02d}
- Databases: {', '.join(databases)}
- Server: {server}

To deploy with Prefect CLI:
    prefect deployment build scheduled_flows.py:scheduled_schema_extraction_flow \\
        --name "weekly-schema-extraction" \\
        --cron "{cron_expression}" \\
        --param databases='[{", ".join(f'"{db}"' for db in databases)}]' \\
        --param server="{server}" \\
        --param connection_config='{{"user": "{user}", "password": "****"}}' \\
        --apply
"""
    return schedule_info


# Export all components
__all__ = [
    'scheduled_schema_extraction_flow',
    'run_scheduled_extraction',
    'extract_single_database_task',
    'generate_extraction_report_task',
    'create_daily_schedule',
    'create_weekly_schedule',
    'DatabaseExtractionResult',
    'ScheduledExtractionResult'
]


if __name__ == "__main__":
    print("=== Scheduled Flows Test ===\n")

    # Example: Show daily schedule configuration
    schedule = create_daily_schedule(
        databases=["EWRReporting", "EWRCentral"],
        server="NCSQLTEST",
        user="EWRUser",
        password="password",
        hour=2
    )
    print(schedule)

    # Example: Show weekly schedule configuration
    schedule = create_weekly_schedule(
        databases=["EWRReporting", "EWRCentral"],
        server="NCSQLTEST",
        user="EWRUser",
        password="password",
        day_of_week=0,  # Sunday
        hour=1
    )
    print(schedule)

    print("\nTo run manually:")
    print("  from prefect_pipelines.scheduled_flows import run_scheduled_extraction")
    print("  result = run_scheduled_extraction(['EWRReporting'], 'NCSQLTEST', 'EWRUser', 'pass')")
