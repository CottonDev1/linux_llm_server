"""
Prefect SQL Query Pipeline

A thin wrapper around the actual QueryPipeline that provides Prefect tracking,
artifacts, and observability without duplicating any pipeline logic.

IMPORTANT: This flow uses the ACTUAL QueryPipeline class from sql_pipeline.
All SQL generation logic lives in query_pipeline.py - this file only provides:
1. Prefect flow/task decorators for tracking
2. Artifact creation for the Prefect UI
3. Timing metrics collection

This ensures that testing the Prefect flow tests the real production code.
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.variables import Variable


# =============================================================================
# Prefect Variables Support
# =============================================================================

async def get_prefect_variable(name: str, default: Any = None) -> Any:
    """Get a Prefect Variable value with fallback to default."""
    try:
        value = await Variable.get(name, default=default)
        return value
    except Exception:
        return default


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class PipelineResult:
    """Result from the actual QueryPipeline."""
    success: bool = False
    sql: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None
    execution_result: Optional[Dict] = None
    matched_rules: List[str] = field(default_factory=list)
    syntax_fixes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    is_exact_match: bool = False
    rule_id: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timing: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_artifact_key(text: str) -> str:
    """Sanitize text for use in artifact keys."""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '-', text.lower())[:50]


# =============================================================================
# Main Pipeline Task - Calls the ACTUAL QueryPipeline
# =============================================================================

@task(
    name="execute_sql_pipeline",
    description="Execute the actual SQL query pipeline (not a copy)",
    retries=1,
    retry_delay_seconds=30,
    tags=["sql", "pipeline", "production"]
)
async def execute_sql_pipeline_task(
    question: str,
    database: str,
    server: str,
    user_id: str,
    credentials: Optional[Dict[str, Any]] = None,
    execute_sql: bool = False,
    include_schema: bool = True,
    use_cache: bool = True,
    max_results: int = 100,
    max_tokens: int = 512
) -> PipelineResult:
    """
    Execute the actual SQL pipeline - NOT a reimplementation.

    This task is a thin wrapper that calls QueryPipeline.generate().
    All SQL generation logic lives in sql_pipeline/query_pipeline.py.
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Calling actual QueryPipeline.generate()")

    result = PipelineResult()

    try:
        # Import and use the ACTUAL pipeline
        from sql_pipeline.query_pipeline import QueryPipeline
        from sql_pipeline.models import SQLCredentials, QueryOptions

        # Create pipeline instance
        pipeline = QueryPipeline()

        # Build credentials if provided
        sql_credentials = None
        if credentials:
            sql_credentials = SQLCredentials(
                server=credentials.get('server', server),
                database=credentials.get('database', database),
                username=credentials.get('username', ''),
                password=credentials.get('password', ''),
                use_windows_auth=credentials.get('use_windows_auth', False),
                domain=credentials.get('domain'),
                trust_cert=credentials.get('trust_cert', True),
            )

        # Build options
        options = QueryOptions(
            execute_sql=execute_sql,
            include_schema=include_schema,
            use_cache=use_cache,
            max_results=max_results,
            max_tokens=max_tokens,
        )

        # Call the ACTUAL pipeline
        pipeline_result = await pipeline.generate(
            question=question,
            database=database,
            server=server,
            credentials=sql_credentials,
            options=options,
        )

        # Map result to our dataclass
        result.success = pipeline_result.success if hasattr(pipeline_result, 'success') else True
        result.sql = pipeline_result.sql
        result.explanation = pipeline_result.explanation
        result.error = pipeline_result.error if hasattr(pipeline_result, 'error') else None
        result.matched_rules = pipeline_result.matched_rules if hasattr(pipeline_result, 'matched_rules') else []
        result.confidence = pipeline_result.confidence if hasattr(pipeline_result, 'confidence') else 0.8
        result.is_exact_match = pipeline_result.is_exact_match if hasattr(pipeline_result, 'is_exact_match') else False
        result.rule_id = pipeline_result.rule_id if hasattr(pipeline_result, 'rule_id') else None
        result.token_usage = pipeline_result.token_usage if hasattr(pipeline_result, 'token_usage') else {}

        # Get syntax fixes if available
        if hasattr(pipeline_result, 'syntax_fixes'):
            result.syntax_fixes = pipeline_result.syntax_fixes

        # Get execution result if available
        if hasattr(pipeline_result, 'execution_result') and pipeline_result.execution_result:
            result.execution_result = pipeline_result.execution_result

        logger.info(f"[{user_id}] Pipeline completed: success={result.success}, sql={result.sql[:50] if result.sql else 'None'}...")

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"[{user_id}] Pipeline error: {e}")

    result.processing_time_ms = (time.time() - start_time) * 1000

    return result


# =============================================================================
# Main Flow
# =============================================================================

@flow(
    name="sql-query-pipeline",
    description="SQL Query Pipeline - Wrapper around actual QueryPipeline for Prefect tracking",
    retries=1,
    retry_delay_seconds=60
)
async def sql_query_flow(
    question: str,
    database: str,
    server: str,
    user_id: str,
    credentials: Optional[Dict[str, Any]] = None,
    execute_sql: bool = False,
    include_schema: bool = True,
    use_cache: bool = True,
    max_results: int = 100,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    SQL Query Pipeline Flow - Tests the ACTUAL production pipeline.

    This flow is a thin wrapper that:
    1. Calls the real QueryPipeline.generate() method
    2. Creates Prefect artifacts for observability
    3. Returns results in a consistent format

    All SQL generation logic lives in sql_pipeline/query_pipeline.py.
    This ensures that Prefect tests verify the actual production code.

    Args:
        question: Natural language question
        database: Target database name
        server: SQL Server hostname
        user_id: User ID for tracking
        credentials: Database credentials (optional)
        execute_sql: Whether to execute the generated SQL
        include_schema: Whether to include schema context
        use_cache: Whether to use cache
        max_results: Maximum rows to return
        max_tokens: Maximum LLM tokens

    Returns:
        Dict with SQL, execution results, and metadata
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"[{user_id}] Starting SQL Query Pipeline (using actual QueryPipeline)")
    logger.info(f"[{user_id}] Question: {question}")
    logger.info(f"[{user_id}] Database: {database}, Server: {server}")

    # Execute the actual pipeline
    result = await execute_sql_pipeline_task(
        question=question,
        database=database,
        server=server,
        user_id=user_id,
        credentials=credentials,
        execute_sql=execute_sql,
        include_schema=include_schema,
        use_cache=use_cache,
        max_results=max_results,
        max_tokens=max_tokens
    )

    total_time_ms = (time.time() - flow_start) * 1000

    # Create artifact for Prefect UI
    artifact_content = f"""
## SQL Query Pipeline Result

**User**: {user_id}
**Database**: {database}
**Server**: {server}
**Success**: {result.success}
**Processing Time**: {total_time_ms:.1f}ms

### Question
{question}

### Generated SQL
```sql
{result.sql or 'No SQL generated'}
```

### Details
- **Confidence**: {result.confidence}
- **Exact Match**: {result.is_exact_match}
- **Rule ID**: {result.rule_id or 'None'}
- **Syntax Fixes**: {result.syntax_fixes or 'None'}
- **Token Usage**: {result.token_usage}

### Explanation
{result.explanation or 'No explanation'}

{f"### Error\\n{result.error}" if result.error else ""}
"""

    await create_markdown_artifact(
        key=f"sql-pipeline-{sanitize_artifact_key(user_id)[:8]}-{int(time.time())}",
        markdown=artifact_content,
        description=f"SQL Pipeline result for: {question[:50]}..."
    )

    # Return result dict
    return {
        "success": result.success,
        "sql": result.sql,
        "explanation": result.explanation,
        "error": result.error,
        "database": database,
        "server": server,
        "user_id": user_id,
        "confidence": result.confidence,
        "is_exact_match": result.is_exact_match,
        "rule_id": result.rule_id,
        "matched_rules": result.matched_rules,
        "syntax_fixes": result.syntax_fixes,
        "token_usage": result.token_usage,
        "execution_result": result.execution_result,
        "timing": {
            "total_ms": total_time_ms,
            "pipeline_ms": result.processing_time_ms,
        },
        "processing_time_ms": total_time_ms,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SQL Query Pipeline")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--database", default="EWRCentral", help="Database name")
    parser.add_argument("--server", default="EWRSQLPROD", help="Server name")
    parser.add_argument("--user-id", default="cli-user", help="User ID")
    parser.add_argument("--execute", action="store_true", help="Execute the SQL")
    parser.add_argument("--no-schema", action="store_true", help="Skip schema loading")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache check")

    args = parser.parse_args()

    result = asyncio.run(sql_query_flow(
        question=args.question,
        database=args.database,
        server=args.server,
        user_id=args.user_id,
        execute_sql=args.execute,
        include_schema=not args.no_schema,
        use_cache=not args.no_cache,
    ))

    print(f"\nResult: {result}")
