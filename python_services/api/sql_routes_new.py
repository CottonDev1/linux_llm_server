"""
SQL Routes - Enhanced FastAPI router for SQL generation and execution.

This module provides the main RAG endpoints for text-to-SQL:
- POST /sql/query-stream - Main RAG endpoint with SSE streaming
- POST /sql/generate - Generate SQL without execution
- POST /sql/validate - Validate SQL against schema
- POST /sql/validate-and-fix - Validate with LLM feedback

Architecture Notes:
------------------
These endpoints migrate the core functionality from Node.js sqlRoutes.js
to Python, enabling better integration with the Python ML/NLP ecosystem.

The query-stream endpoint provides:
1. Progress updates via SSE (analyzing, searching, loading, generating, etc.)
2. SQL generation with self-correction
3. Query execution on SQL Server
4. AI-generated response summaries
5. Error explanations for failed queries

SSE Event Types:
- progress: Stage updates during pipeline execution
- result: Final result (success or error)

Security:
- All endpoints validate user input for injection attacks
- Generated SQL is validated before execution
- Only SELECT/WITH queries are allowed
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sql", tags=["SQL"])


# ==============================================================================
# Request/Response Models
# ==============================================================================

class SQLQueryStreamRequest(BaseModel):
    """Request for streaming SQL query generation and execution."""
    naturalLanguage: str = Field(..., description="Natural language question")
    server: str = Field(..., description="SQL Server hostname")
    database: str = Field(..., description="Target database name")
    user: Optional[str] = Field(None, description="SQL Server username")
    password: Optional[str] = Field(None, description="SQL Server password")
    trustServerCertificate: bool = Field(True, description="Trust server certificate")
    encrypt: bool = Field(False, description="Use encryption")
    conversationHistory: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation")
    maxTokens: int = Field(512, ge=1, le=4096, description="Maximum tokens for LLM response")


class SQLGenerateRequest(BaseModel):
    """Request for SQL generation without execution."""
    naturalLanguage: str = Field(..., description="Natural language question")
    database: str = Field(..., description="Target database name")
    conversationHistory: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation")
    maxTokens: int = Field(512, ge=1, le=4096, description="Maximum tokens for LLM response")


class SQLValidateRequest(BaseModel):
    """Request for SQL validation."""
    sql: str = Field(..., description="SQL query to validate")
    database: str = Field(..., description="Target database name")


class SQLGenerateResponse(BaseModel):
    """Response for SQL generation."""
    success: bool
    sql: str = ""
    error: Optional[str] = None
    is_exact_match: bool = False
    rule_id: Optional[str] = None
    validation_attempts: int = 0
    schema_errors: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    generation_time_ms: int = 0
    model_used: str = ""
    security_blocked: bool = False


class SQLValidateResponse(BaseModel):
    """Response for SQL validation."""
    valid: bool
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: Dict[str, List[str]] = Field(default_factory=dict)
    feedback_for_llm: str = ""


# ==============================================================================
# Helper Functions
# ==============================================================================

def execute_sql_query(
    sql: str,
    server: str,
    database: str,
    user: Optional[str],
    password: Optional[str],
    trust_cert: bool = True,
    encrypt: bool = False,
    timeout: int = 60000,
) -> Dict[str, Any]:
    """
    Execute SQL query against SQL Server.

    Uses pymssql for database connectivity.

    Returns:
        Dict with columns, rows, rowCount, or error
    """
    try:
        import pymssql

        # Build connection kwargs
        conn_kwargs = {
            "server": server,
            "database": database,
            "timeout": timeout // 1000,  # pymssql uses seconds
            "login_timeout": 30,
            "user": user,
            "password": password,
        }

        # Connect and execute
        with pymssql.connect(**conn_kwargs) as conn:
            with conn.cursor(as_dict=True) as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()

                columns = []
                if rows:
                    columns = list(rows[0].keys())

                return {
                    "success": True,
                    "columns": columns,
                    "rows": rows,
                    "rowCount": len(rows)
                }

    except pymssql.Error as e:
        logger.error(f"SQL execution error: {e}")
        return {
            "success": False,
            "error": str(e),
            "columns": [],
            "rows": [],
            "rowCount": 0
        }
    except Exception as e:
        logger.error(f"Unexpected SQL error: {e}")
        return {
            "success": False,
            "error": str(e),
            "columns": [],
            "rows": [],
            "rowCount": 0
        }


def add_safety_limit(sql: str) -> str:
    """
    Add TOP 1000 safety limit to raw SELECT queries.

    This prevents accidentally returning millions of rows.
    Does NOT add limit to:
    - Queries with existing TOP
    - Aggregate queries (COUNT, SUM, AVG, etc.)
    - Queries with GROUP BY
    """
    sql_upper = sql.upper()

    has_top = "TOP " in sql_upper
    has_aggregate = any(agg in sql_upper for agg in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("])
    has_group_by = "GROUP BY" in sql_upper

    if has_top or has_aggregate or has_group_by:
        return sql

    # Add TOP 1000 to raw SELECT
    import re
    if re.search(r"SELECT\s+(?:\*|[\w\[\]\.]+(?:\s*,\s*[\w\[\]\.]+)*)\s+FROM", sql, re.IGNORECASE):
        return re.sub(r"SELECT\s+", "SELECT TOP 1000 ", sql, count=1, flags=re.IGNORECASE)

    return sql


def build_result_summary(columns: List[str], rows: List[Dict], row_count: int) -> str:
    """Build a summary of query results for AI response generation."""
    if row_count == 0:
        return "The query returned no results."

    if row_count == 1 and len(columns) <= 3:
        # Single row with few columns - show actual values
        row = rows[0]
        values = ", ".join(f"{col}: {row.get(col)}" for col in columns)
        return f"The query returned 1 row: {values}"

    # Multiple rows - summarize
    summary = f"The query returned {row_count} row{'s' if row_count != 1 else ''} with columns: {', '.join(columns)}."

    if row_count > 0 and rows:
        first_row = rows[0]
        sample_values = ", ".join(
            f"{col}={str(first_row.get(col, 'NULL'))[:30]}"
            for col in columns[:3]
        )
        summary += f" First row sample: {sample_values}"

    return summary


# ==============================================================================
# Endpoints
# ==============================================================================

@router.post("/query-stream")
async def query_stream(request: SQLQueryStreamRequest, http_request: Request):
    """
    Main RAG endpoint for text-to-SQL with streaming progress.

    This endpoint provides the full RAG pipeline:
    1. Analyzes the natural language question
    2. Searches for similar examples
    3. Loads relevant schema context
    4. Generates SQL using the LLM
    5. Validates and self-corrects if needed
    6. Executes the query on SQL Server
    7. Generates an AI response summary

    Progress is streamed via SSE with events like:
    - {"type": "progress", "stage": "analyzing", "message": "..."}
    - {"type": "result", "success": true, "generatedSql": "...", "rows": [...]}

    This is a migration of the Node.js /api/sql/query-stream endpoint.
    """
    from sql_pipeline import get_query_pipeline
    from sql_pipeline.services.security_service import SecurityService

    async def event_generator():
        """Generate SSE events for the query pipeline."""
        current_stage = "initializing"
        generated_sql = None
        final_sql = None

        def send_progress(stage: str, message: str, **data):
            """Send a progress event."""
            nonlocal current_stage
            current_stage = stage
            event = {"type": "progress", "stage": stage, "message": message, **data}
            return f"data: {json.dumps(event)}\n\n"

        def send_result(success: bool, **data):
            """Send the final result event."""
            event = {"type": "result", "success": success, **data}
            return f"data: {json.dumps(event)}\n\n"

        try:
            # Validate authentication
            if not request.user or not request.password:
                yield send_result(False, error="User and password required for SQL authentication")
                return

            # Step 1: Security validation
            yield send_progress("analyzing", "Analyzing your query...")

            security_service = SecurityService()
            security_check = security_service.check_query(request.naturalLanguage)

            if security_check.blocked:
                logger.warning(f"Security blocked: {security_check.issues}")
                yield send_result(
                    False,
                    error="; ".join(security_check.issues) if security_check.issues else "Security check failed",
                    securityViolation=True,
                    reason="; ".join(security_check.issues) if security_check.issues else "Blocked by security policy"
                )
                return

            # Step 2: Generate SQL
            yield send_progress("searching", "Searching for similar examples...")

            pipeline = await get_query_pipeline()

            yield send_progress("loading", "Loading database schema...")
            yield send_progress("building", "Building query prompt...")
            yield send_progress("generating", "Generating SQL with AI...")

            start_time = time.time()
            gen_result = await pipeline.generate(
                question=request.naturalLanguage,
                database=request.database,
                server=request.server,
            )

            if not gen_result.success:
                if gen_result.security_blocked:
                    yield send_result(
                        False,
                        error=gen_result.error,
                        securityViolation=True,
                        violations=gen_result.security_violations
                    )
                else:
                    yield send_result(False, error=gen_result.error)
                return

            generated_sql = gen_result.sql

            # Log generation info
            if gen_result.is_exact_match:
                logger.info(f"Exact rule match: {gen_result.rule_id}")
            else:
                logger.info(f"Generated SQL in {int(gen_result.processing_time * 1000)}ms")

            # Step 3: Validate generated SQL (already done in generator, double-check security)
            yield send_progress("validating", "Validating SQL query...")

            sql_validation = security_service.check_query(generated_sql)
            if sql_validation.blocked:
                logger.warning(f"Security blocked SQL: {sql_validation.issues}")
                yield send_result(
                    False,
                    error="; ".join(sql_validation.issues) if sql_validation.issues else "SQL blocked by security",
                    generatedSql=generated_sql,
                    securityViolation=True,
                    violations=sql_validation.issues
                )
                return

            # Step 4: Add safety limit if needed
            final_sql = add_safety_limit(generated_sql)
            if final_sql != generated_sql:
                logger.info("Added TOP 1000 safety limit")

            # Step 5: Execute query
            yield send_progress("executing", "Executing query on database...")

            # Run synchronous pymssql in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            exec_result = await loop.run_in_executor(
                None,
                lambda: execute_sql_query(
                    final_sql,
                    request.server,
                    request.database,
                    request.user,
                    request.password,
                    request.trustServerCertificate,
                    request.encrypt,
                )
            )

            if not exec_result.get("success"):
                # Query execution failed
                error_msg = exec_result.get("error", "Query execution failed")

                # Try to generate AI explanation
                ai_explanation = None
                try:
                    yield send_progress("explaining", "Analyzing what went wrong...")
                    ai_explanation = await pipeline.generate_error_explanation(
                        request.naturalLanguage,
                        final_sql,
                        error_msg
                    )
                except Exception as e:
                    logger.warning(f"Could not generate error explanation: {e}")

                yield send_result(
                    False,
                    error="Query failed",
                    details=error_msg,
                    generatedSql=final_sql,
                    aiExplanation=ai_explanation
                )
                return

            # Step 6: Generate AI response
            columns = exec_result.get("columns", [])
            rows = exec_result.get("rows", [])
            row_count = exec_result.get("rowCount", 0)

            ai_response = None
            try:
                yield send_progress("summarizing", "Generating response...")
                result_summary = build_result_summary(columns, rows, row_count)
                ai_response = await pipeline.generate_response_summary(
                    request.naturalLanguage,
                    final_sql,
                    result_summary
                )
            except Exception as e:
                logger.warning(f"Could not generate AI response: {e}")

            # Success!
            yield send_result(
                True,
                generatedSql=final_sql,
                columns=columns,
                rows=rows,
                rowCount=row_count,
                tokenUsage=gen_result.token_usage,
                aiResponse=ai_response,
                isExactMatch=gen_result.is_exact_match,
                ruleId=gen_result.rule_id,
            )

        except Exception as e:
            logger.error(f"Query stream error: {e}", exc_info=True)

            error_msg = str(e)
            is_timeout = "timeout" in error_msg.lower()

            yield send_result(
                False,
                error="Timeout" if is_timeout else "Query failed",
                stage=current_stage,
                isTimeout=is_timeout,
                details=error_msg,
                generatedSql=final_sql or generated_sql
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/generate", response_model=SQLGenerateResponse)
async def generate_sql(request: SQLGenerateRequest) -> SQLGenerateResponse:
    """
    Generate SQL from natural language without execution.

    This endpoint is useful for:
    - Previewing generated SQL before execution
    - Testing SQL generation quality
    - Debugging the RAG pipeline
    """
    from sql_pipeline import get_query_pipeline

    try:
        pipeline = await get_query_pipeline()

        result = await pipeline.generate(
            question=request.naturalLanguage,
            database=request.database,
            server="NCSQLTEST",  # Default server for generation-only
        )

        return SQLGenerateResponse(
            success=result.success,
            sql=result.sql,
            error=result.error,
            is_exact_match=result.is_exact_match,
            rule_id=result.rule_id,
            validation_attempts=0,  # Not tracked in new pipeline
            schema_errors=[],  # Not tracked in new pipeline
            token_usage=result.token_usage,
            generation_time_ms=int(result.processing_time * 1000),
            model_used="qwen2.5-coder",  # Default model
            security_blocked=result.security_blocked,
        )

    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        return SQLGenerateResponse(
            success=False,
            error=str(e)
        )


@router.post("/validate", response_model=SQLValidateResponse)
async def validate_sql(request: SQLValidateRequest) -> SQLValidateResponse:
    """
    Validate SQL against schema.

    Returns validation errors if the SQL references invalid tables or columns,
    along with suggestions for corrections.
    """
    from schema_validator import get_schema_validator
    from mongodb import get_mongodb_service
    from database_name_parser import normalize_database_name

    try:
        mongodb = get_mongodb_service()
        validator = await get_schema_validator(mongodb)

        master_database = normalize_database_name(request.database)
        result = validator.validate_sql(request.sql, master_database)

        return SQLValidateResponse(
            valid=result.get("valid", False),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            suggestions=result.get("suggestions", {}),
            feedback_for_llm=""
        )

    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        return SQLValidateResponse(
            valid=False,  # Fail closed - don't allow potentially invalid SQL
            errors=[{"type": "validation_error", "message": f"Validation failed: {str(e)}"}],
            warnings=[]
        )


@router.post("/validate-and-fix", response_model=SQLValidateResponse)
async def validate_and_fix_sql(request: SQLValidateRequest) -> SQLValidateResponse:
    """
    Validate SQL and generate LLM-friendly feedback for self-correction.

    This endpoint is used by the self-correction loop to get detailed
    feedback that can be added to the LLM prompt for regeneration.
    """
    from schema_validator import get_schema_validator
    from mongodb import get_mongodb_service
    from database_name_parser import normalize_database_name

    try:
        mongodb = get_mongodb_service()
        validator = await get_schema_validator(mongodb)

        master_database = normalize_database_name(request.database)
        result = validator.validate_sql(request.sql, master_database)

        feedback = ""
        if not result.get("valid"):
            feedback = validator.format_validation_feedback(result, master_database)

        return SQLValidateResponse(
            valid=result.get("valid", False),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            suggestions=result.get("suggestions", {}),
            feedback_for_llm=feedback
        )

    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        return SQLValidateResponse(
            valid=False,  # Fail closed - don't allow potentially invalid SQL
            errors=[{"type": "validation_error", "message": f"Validation failed: {str(e)}"}],
            feedback_for_llm=f"VALIDATION ERROR: {str(e)}. Please verify your SQL syntax and schema references."
        )
