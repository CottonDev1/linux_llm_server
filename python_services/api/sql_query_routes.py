"""
SQL Query API Routes
====================

FastAPI endpoints for SQL query generation using the SQL Pipeline.
Provides both streaming and non-streaming endpoints for natural language to SQL conversion.
"""

import logging
import time
import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query as QueryParam, Path, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from sql_pipeline.query_pipeline import QueryPipeline
from sql_pipeline.models.query_models import (
    SQLQueryRequest,
    SQLQueryResult,
    SQLCredentials,
    QueryOptions,
    SSEEvent,
)
from sql_pipeline.models.validation_models import ExecutionResult
from sql_pipeline.services.rules_service import RulesService
from mongodb import MongoDBService
from database_name_parser import normalize_database_name

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sql", tags=["SQL Query"])

# Singleton pipeline instance
_pipeline_instance: Optional[QueryPipeline] = None


async def get_pipeline() -> QueryPipeline:
    """Get or create singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = QueryPipeline()
        await _pipeline_instance._get_services()
    return _pipeline_instance


# ============================================================================
# Request/Response Models
# ============================================================================

class ExecuteRequest(BaseModel):
    """Request model for raw SQL execution."""
    sql: str = Field(..., description="SQL query to execute")
    database: str = Field(..., description="Target database name")
    server: str = Field(default="NCSQLTEST", description="SQL Server hostname")
    credentials: SQLCredentials = Field(..., description="Database credentials")
    max_results: int = Field(default=100, ge=1, le=10000, description="Maximum rows to return")


class TestConnectionRequest(BaseModel):
    """Request model for connection testing."""
    server: str = Field(..., description="SQL Server hostname")
    database: Optional[str] = Field(default="master", description="Target database name (defaults to 'master' for connection test)")
    username: Optional[str] = Field(default=None, description="SQL Server username")
    user: Optional[str] = Field(default=None, description="SQL Server username (alias for username)")
    password: Optional[str] = Field(default=None, description="SQL Server password")
    domain: Optional[str] = Field(default=None, description="Domain for Windows authentication (e.g., 'EWR')")
    authType: Optional[str] = Field(default="sql", description="Authentication type: 'sql' or 'windows'")
    trustServerCertificate: bool = Field(default=True, description="Trust server certificate")
    encrypt: bool = Field(default=False, description="Use encrypted connection")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data):
        """Normalize frontend fields to backend format."""
        if isinstance(data, dict):
            # Handle 'user' alias for 'username' (check key exists, not just truthy value)
            if "user" in data and data["user"] and not data.get("username"):
                data["username"] = data["user"]
        return data

    def get_qualified_username(self) -> str:
        """
        Get username with domain prefix if specified.
        For Windows auth: returns DOMAIN\\username
        For SQL auth: returns username as-is
        """
        if self.domain and self.authType == "windows":
            return f"{self.domain}\\{self.username}"
        return self.username


class TestConnectionResponse(BaseModel):
    """Response model for connection testing."""
    success: bool = Field(..., description="Whether connection was successful")
    message: str = Field(..., description="Connection result message")
    latency_ms: Optional[float] = Field(default=None, description="Connection latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if connection failed")
    details: Optional[str] = Field(default=None, description="Detailed error information")
    connection_info: Optional[Dict[str, Any]] = Field(default=None, description="Connection parameters used")
    databases: Optional[List[str]] = Field(default=None, description="List of available databases (returned on successful connection)")


class DatabaseInfo(BaseModel):
    """Database information model."""
    database: str = Field(..., description="Database name")
    table_count: int = Field(default=0, description="Number of tables analyzed")
    last_updated: Optional[str] = Field(default=None, description="Last analysis timestamp")


class FeedbackRequest(BaseModel):
    """Request model for query feedback."""
    query_id: Optional[str] = Field(default=None, description="Query ID (optional)")
    question: Optional[str] = Field(default=None, description="Original natural language question")
    query: Optional[str] = Field(default=None, description="Original natural language question (alias)")
    sql: Optional[str] = Field(default=None, description="Generated SQL query")
    generatedSql: Optional[str] = Field(default=None, description="Generated SQL query (alias)")
    database: str = Field(..., description="Target database")
    feedback: Optional[str] = Field(default=None, description="Feedback type: 'positive' or 'negative'")
    isPositive: Optional[bool] = Field(default=None, description="Whether feedback is positive (alias)")
    comment: Optional[str] = Field(default=None, description="Optional feedback comment")
    reason: Optional[str] = Field(default=None, description="Feedback reason (alias for comment)")

    @model_validator(mode="after")
    def normalize_fields(self):
        """Normalize frontend field names to backend format."""
        # Handle question/query alias
        if not self.question and self.query:
            self.question = self.query
        # Handle sql/generatedSql alias
        if not self.sql and self.generatedSql:
            self.sql = self.generatedSql
        # Handle feedback/isPositive alias
        if not self.feedback and self.isPositive is not None:
            self.feedback = "positive" if self.isPositive else "negative"
        # Handle comment/reason alias
        if not self.comment and self.reason:
            self.comment = self.reason
        # Validate required fields
        if not self.question:
            raise ValueError("question (or query) is required")
        if not self.sql:
            raise ValueError("sql (or generatedSql) is required")
        if not self.feedback:
            raise ValueError("feedback (or isPositive) is required")
        return self


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool = Field(..., description="Whether feedback was stored")
    message: str = Field(..., description="Result message")
    feedback_id: Optional[str] = Field(default=None, description="Feedback record ID")


class SaveExampleRequest(BaseModel):
    """Request model for saving training examples."""
    question: str = Field(..., description="Natural language question")
    sql: str = Field(..., description="Correct SQL query")
    database: str = Field(..., description="Target database")
    explanation: Optional[str] = Field(default=None, description="Optional explanation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class SaveExampleResponse(BaseModel):
    """Response model for saving examples."""
    success: bool = Field(..., description="Whether example was saved")
    message: str = Field(..., description="Result message")
    example_id: Optional[str] = Field(default=None, description="Example record ID")


class UserSettings(BaseModel):
    """User settings model."""
    default_server: Optional[str] = Field(default=None, description="Default SQL Server")
    default_database: Optional[str] = Field(default=None, description="Default database")
    default_max_results: int = Field(default=100, description="Default result limit")
    enable_streaming: bool = Field(default=True, description="Enable streaming by default")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    use_cache: bool = Field(default=True, description="Use cache by default")


class UserSettingsResponse(BaseModel):
    """Response model for user settings."""
    success: bool = Field(..., description="Whether operation was successful")
    settings: Optional[UserSettings] = Field(default=None, description="User settings")


class RuleInfo(BaseModel):
    """Rule information model."""
    rule_id: str = Field(..., description="Unique rule ID")
    description: str = Field(..., description="Rule description")
    priority: str = Field(..., description="Rule priority: critical, high, normal")
    enabled: bool = Field(..., description="Whether rule is enabled")
    trigger_keywords: List[str] = Field(default_factory=list, description="Trigger keywords")
    trigger_tables: List[str] = Field(default_factory=list, description="Trigger tables")
    trigger_columns: List[str] = Field(default_factory=list, description="Trigger columns")
    rule_text: Optional[str] = Field(default=None, description="Rule guidance text")
    database: Optional[str] = Field(default=None, description="Target database")
    type: Optional[str] = Field(default=None, description="Rule type: assistance or constraint")


class CreateRuleRequest(BaseModel):
    """Request model for creating a SQL rule."""
    database: str = Field(default="_global", description="Database name or '_global' for all")
    rule_id: str = Field(..., description="Unique rule identifier")
    description: str = Field(..., description="Human-readable description")
    type: str = Field(default="assistance", description="Rule type: 'assistance' or 'constraint'")
    priority: str = Field(default="normal", description="Priority: 'critical', 'high', or 'normal'")
    enabled: bool = Field(default=True, description="Whether rule is active")
    trigger_keywords: List[str] = Field(default_factory=list, description="Keywords that trigger this rule")
    trigger_tables: List[str] = Field(default_factory=list, description="Tables that trigger this rule")
    trigger_columns: List[str] = Field(default_factory=list, description="Columns that trigger this rule")
    rule_text: str = Field(..., description="Guidance text for LLM")
    auto_fix_pattern: Optional[str] = Field(default=None, description="Regex pattern for auto-fix")
    auto_fix_replacement: Optional[str] = Field(default=None, description="Replacement for auto-fix")
    example_question: Optional[str] = Field(default=None, description="Example question")
    example_sql: Optional[str] = Field(default=None, description="Example SQL")


class CreateRuleResponse(BaseModel):
    """Response model for rule creation."""
    success: bool = Field(..., description="Whether rule was created")
    message: str = Field(..., description="Result message")
    rule_id: Optional[str] = Field(default=None, description="Created rule ID")


class RulesResponse(BaseModel):
    """Response model for rules retrieval."""
    success: bool = Field(..., description="Whether rules were retrieved")
    database: str = Field(..., description="Target database")
    rule_count: int = Field(default=0, description="Number of rules")
    rules: List[RuleInfo] = Field(default_factory=list, description="List of rules")


# ============================================================================
# Query Endpoints
# ============================================================================

@router.post("/query", response_model=SQLQueryResult)
async def query_non_streaming(request: SQLQueryRequest):
    """
    Generate SQL from natural language question (non-streaming).

    This endpoint processes the question, generates SQL using the pipeline,
    and optionally executes the query if credentials are provided.

    **Request Body:**
    - `natural_language`: The question to convert to SQL
    - `database`: Target database name
    - `server`: SQL Server hostname (default: NCSQLTEST)
    - `credentials`: Database credentials (required if execute_sql is True)
    - `options`: Query processing options

    **Returns:**
    - Generated SQL query
    - Explanation of the SQL
    - Execution results (if execute_sql was True)
    - Matched rules
    - Confidence score
    - Processing time
    """
    try:
        pipeline = await get_pipeline()
        result = await pipeline.process_query(request)
        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/query-stream")
async def query_streaming(request: SQLQueryRequest):
    """
    Generate SQL from natural language question with Server-Sent Events (SSE) streaming.

    This endpoint streams progress updates as the pipeline processes the query.

    **Event Types:**
    - `status`: Pipeline stage updates
    - `rules`: Matched rules information
    - `schema`: Loaded schema information
    - `sql`: Generated SQL query
    - `explanation`: SQL explanation
    - `validation`: Validation results
    - `execution`: Execution results (if execute_sql was True)
    - `error`: Error information
    - `done`: Processing complete

    **Returns:**
    - StreamingResponse with text/event-stream content type
    """
    try:
        pipeline = await get_pipeline()

        async def event_generator():
            """Generate SSE events from pipeline."""
            try:
                async for event in pipeline.process_query_streaming(request):
                    yield event.to_sse()

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                error_event = SSEEvent(
                    event="error",
                    data={"error": str(e)}
                )
                yield error_event.to_sse()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Failed to initialize streaming: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Streaming initialization failed: {str(e)}")


# ============================================================================
# Execution Endpoints
# ============================================================================

@router.post("/execute", response_model=ExecutionResult)
async def execute_raw_sql(request: ExecuteRequest):
    """
    Execute raw SQL query directly without generation.

    This endpoint executes a provided SQL query against the specified database.
    Use with caution as it bypasses the SQL generation pipeline.

    **Request Body:**
    - `sql`: SQL query to execute
    - `database`: Target database name
    - `server`: SQL Server hostname
    - `credentials`: Database credentials
    - `max_results`: Maximum rows to return

    **Returns:**
    - Execution result with data, row count, and timing
    """
    try:
        pipeline = await get_pipeline()

        result = await pipeline.execute_sql(
            sql=request.sql,
            credentials=request.credentials,
            max_results=request.max_results
        )

        return result

    except Exception as e:
        logger.error(f"SQL execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")


@router.post("/test-connection", response_model=TestConnectionResponse)
async def test_database_connection(request: TestConnectionRequest):
    """
    Test database connection and return available databases.

    This endpoint establishes a connection to the specified server, tests it,
    and returns the list of available databases. The connection is pooled
    in the singleton pipeline for reuse by subsequent queries.

    **Request Body:**
    - `server`: SQL Server hostname
    - `database`: Target database name (optional, defaults to 'master')
    - `username`: SQL Server username
    - `password`: SQL Server password

    **Returns:**
    - Connection success status
    - Message describing the result
    - Connection latency in milliseconds
    - List of available databases (on success)
    - Error details if connection failed
    - Connection parameters used
    """
    logger.info(f"Test connection request received:")
    logger.info(f"  server={request.server}")
    logger.info(f"  database={request.database}")
    logger.info(f"  authType={request.authType}")
    logger.info(f"  domain={request.domain}")
    logger.info(f"  username={request.username}")
    logger.info(f"  password={request.password}")

    try:
        # Use singleton pipeline's execution service for connection pooling
        pipeline = await get_pipeline()
        await pipeline._get_services()
        execution_service = pipeline._execution_service

        # Use 'master' as default database for connection testing
        database = request.database if request.database else "master"

        # Create credentials object with domain for qualified username
        credentials = SQLCredentials(
            server=request.server,
            database=database,
            username=request.username,
            password=request.password,
            domain=request.domain
        )

        # Build connection info for response
        connection_info = credentials.get_connection_params()

        # Test connection using singleton's execution service
        start_time = time.time()
        success, error_msg = await execution_service.test_connection(credentials)
        latency_ms = (time.time() - start_time) * 1000

        if success:
            # Connection successful - now get database list using same pooled connection
            databases = []
            try:
                # Create master credentials for listing databases
                master_credentials = SQLCredentials(
                    server=request.server,
                    database="master",
                    username=request.username,
                    password=request.password,
                    domain=request.domain
                )

                list_db_sql = """
                    SELECT name
                    FROM sys.databases
                    WHERE state_desc = 'ONLINE'
                    AND name NOT IN ('master', 'tempdb', 'model', 'msdb')
                    ORDER BY name
                """

                result = await execution_service.execute(list_db_sql, master_credentials, max_results=1000)
                if result.success and result.data:
                    databases = [row.get("name") for row in result.data if row.get("name")]
                    logger.info(f"Found {len(databases)} databases on {request.server}")
            except Exception as db_error:
                logger.warning(f"Failed to list databases after successful connection: {db_error}")
                # Don't fail the whole request, just return empty database list

            return TestConnectionResponse(
                success=True,
                message=f"Connected to {request.server} successfully",
                latency_ms=round(latency_ms, 2),
                connection_info=connection_info,
                databases=databases
            )
        else:
            return TestConnectionResponse(
                success=False,
                message=f"Failed to connect to {request.server}",
                error=error_msg,
                details=f"Connection string: {credentials.get_connection_string()}",
                connection_info=connection_info,
                latency_ms=None
            )

    except ValueError as e:
        # Validation errors from SQLCredentials
        logger.error(f"Connection validation error: {e}", exc_info=True)
        return TestConnectionResponse(
            success=False,
            message="Invalid connection parameters",
            error=str(e),
            details="Please check your credentials",
            latency_ms=None
        )

    except Exception as e:
        logger.error(f"Connection test error: {e}", exc_info=True)
        return TestConnectionResponse(
            success=False,
            message=f"Connection test failed",
            error=str(e),
            details=f"Server: {request.server}, Database: {request.database or 'master'}",
            latency_ms=None
        )


@router.post("/disconnect")
async def disconnect():
    """
    Close all database connections in the connection pool.

    This endpoint should be called when the user clicks "Clear" to release
    database connections and clean up resources.

    **Returns:**
    - Success status and message
    """
    try:
        pipeline = await get_pipeline()
        if pipeline._execution_service:
            await pipeline._execution_service.close()
            logger.info("Connection pool closed via disconnect endpoint")

        return {
            "success": True,
            "message": "All database connections closed"
        }

    except Exception as e:
        logger.error(f"Disconnect error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error closing connections: {str(e)}"
        }


# ============================================================================
# Database & Schema Endpoints
# ============================================================================

@router.get("/databases", response_model=List[DatabaseInfo])
async def list_analyzed_databases():
    """
    List all databases that have been analyzed for schema.

    This endpoint queries the sql_schema_context collection to find all
    databases with analyzed schemas and returns metadata about each.

    **Returns:**
    - List of database information including table count and last update time
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_schema_context"]

        # Aggregate to get database stats
        pipeline_agg = [
            {
                "$group": {
                    "_id": "$database",
                    "table_count": {"$sum": 1},
                    "last_updated": {"$max": "$updated_at"}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]

        cursor = collection.aggregate(pipeline_agg)
        results = await cursor.to_list(length=None)

        databases = []
        for result in results:
            last_updated = result.get("last_updated")
            # Convert datetime to ISO string if needed
            if hasattr(last_updated, 'isoformat'):
                last_updated = last_updated.isoformat()
            databases.append(DatabaseInfo(
                database=result["_id"],
                table_count=result.get("table_count", 0),
                last_updated=last_updated
            ))

        return databases

    except Exception as e:
        logger.error(f"Failed to list databases: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list databases: {str(e)}")


class ListDatabasesRequest(BaseModel):
    """Request model for listing databases from a SQL server."""
    server: str = Field(..., description="SQL Server hostname")
    username: Optional[str] = Field(default=None, description="SQL Server username")
    user: Optional[str] = Field(default=None, description="SQL Server username (alias for username)")
    password: Optional[str] = Field(default=None, description="SQL Server password")
    domain: Optional[str] = Field(default=None, description="Domain for Windows authentication (e.g., 'EWR')")

    @model_validator(mode="before")
    @classmethod
    def normalize_username(cls, data):
        """Accept either 'user' or 'username' field."""
        if isinstance(data, dict):
            if data.get("user") and not data.get("username"):
                data["username"] = data["user"]
        return data


@router.post("/databases")
async def list_server_databases(request: ListDatabasesRequest):
    """
    List available databases from a SQL server.

    This endpoint connects to the specified SQL server and retrieves
    a list of online databases.

    **Request Body:**
    - `server`: SQL Server hostname
    - `username`: SQL Server username
    - `password`: SQL Server password

    **Returns:**
    - List of database names available on the server
    """
    try:
        from sql_pipeline.services.execution_service import ExecutionService

        # Create credentials for 'master' database to list all databases
        credentials = SQLCredentials(
            server=request.server,
            database="master",
            username=request.username,
            password=request.password,
            domain=request.domain
        )

        execution_service = ExecutionService()

        # Query to list online databases
        list_db_sql = """
            SELECT name
            FROM sys.databases
            WHERE state_desc = 'ONLINE'
            AND name NOT IN ('master', 'tempdb', 'model', 'msdb')
            ORDER BY name
        """

        result = await execution_service.execute_sql(list_db_sql, credentials, max_results=1000)

        if result.get("success") and result.get("data"):
            databases = [row.get("name") for row in result["data"] if row.get("name")]
            return {
                "success": True,
                "databases": databases
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Failed to retrieve database list"),
                "databases": []
            }

    except Exception as e:
        logger.error(f"Failed to list server databases: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "databases": []
        }


# ============================================================================
# Feedback & Learning Endpoints
# ============================================================================

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a generated SQL query.

    This endpoint stores user feedback in the agent_learning collection
    to improve future query generation.

    **Request Body:**
    - `query_id`: Optional query ID for tracking
    - `question`: Original natural language question
    - `sql`: Generated SQL query
    - `database`: Target database
    - `feedback`: "positive" or "negative"
    - `comment`: Optional feedback comment

    **Returns:**
    - Success status and feedback ID
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        # Create feedback document
        feedback_doc = {
            "query_id": request.query_id,
            "question": request.question,
            "question_normalized": request.question.lower().strip(),
            "sql": request.sql,
            "database": request.database,
            "feedback": request.feedback,
            "comment": request.comment,
            "success": request.feedback == "positive",
            "created_at": datetime.utcnow().isoformat(),
            "timestamp": time.time()
        }

        result = await collection.insert_one(feedback_doc)

        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=str(result.inserted_id)
        )

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


class FeedbackItem(BaseModel):
    """Model for feedback item in list response."""
    id: str = Field(..., description="Feedback ID")
    question: str = Field(..., description="Original question")
    sql: str = Field(..., description="Generated SQL")
    database: str = Field(..., description="Target database")
    feedback: str = Field(..., description="Feedback type: positive or negative")
    comment: Optional[str] = Field(default=None, description="User comment")
    created_at: str = Field(..., description="Creation timestamp")
    processed: bool = Field(default=False, description="Whether feedback has been processed")
    processed_at: Optional[str] = Field(default=None, description="Processing timestamp")
    rule_created: Optional[str] = Field(default=None, description="Rule ID if rule was created from this feedback")


class FeedbackListResponse(BaseModel):
    """Response model for feedback list."""
    success: bool = Field(..., description="Whether retrieval was successful")
    total: int = Field(default=0, description="Total feedback count")
    unprocessed: int = Field(default=0, description="Unprocessed feedback count")
    feedback: List[FeedbackItem] = Field(default_factory=list, description="List of feedback items")


class UpdateFeedbackRequest(BaseModel):
    """Request model for updating feedback."""
    processed: bool = Field(default=True, description="Mark as processed")
    rule_created: Optional[str] = Field(default=None, description="Rule ID if rule was created")
    notes: Optional[str] = Field(default=None, description="Admin notes")


@router.get("/feedback", response_model=FeedbackListResponse)
async def get_feedback(
    feedback_type: Optional[str] = QueryParam(default=None, description="Filter by type: 'positive' or 'negative'"),
    database: Optional[str] = QueryParam(default=None, description="Filter by database"),
    processed: Optional[bool] = QueryParam(default=None, description="Filter by processed status"),
    limit: int = QueryParam(default=50, ge=1, le=200, description="Maximum items to return"),
    skip: int = QueryParam(default=0, ge=0, description="Items to skip for pagination")
):
    """
    Get feedback submissions for review.

    This endpoint retrieves feedback from the agent_learning collection
    with optional filtering by type, database, and processed status.

    **Query Parameters:**
    - `feedback_type`: Filter by 'positive' or 'negative'
    - `database`: Filter by database name
    - `processed`: Filter by processed status (true/false)
    - `limit`: Maximum items to return (default 50)
    - `skip`: Items to skip for pagination

    **Returns:**
    - List of feedback items with counts
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        # Build query filter
        query: Dict[str, Any] = {}

        # Only get feedback entries (not corrections)
        query["feedback"] = {"$exists": True}

        if feedback_type:
            query["feedback"] = feedback_type
        if database:
            query["database"] = database
        if processed is not None:
            query["processed"] = processed

        # Get total count
        total = await collection.count_documents({"feedback": {"$exists": True}})

        # Get unprocessed count
        unprocessed = await collection.count_documents({
            "feedback": {"$exists": True},
            "$or": [{"processed": False}, {"processed": {"$exists": False}}]
        })

        # Get feedback with pagination, newest first
        cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        items = await cursor.to_list(length=limit)

        feedback_list = []
        for item in items:
            feedback_list.append(FeedbackItem(
                id=str(item.get("_id")),
                question=item.get("question", ""),
                sql=item.get("sql", ""),
                database=item.get("database", ""),
                feedback=item.get("feedback", ""),
                comment=item.get("comment"),
                created_at=item.get("created_at", ""),
                processed=item.get("processed", False),
                processed_at=item.get("processed_at"),
                rule_created=item.get("rule_created")
            ))

        return FeedbackListResponse(
            success=True,
            total=total,
            unprocessed=unprocessed,
            feedback=feedback_list
        )

    except Exception as e:
        logger.error(f"Failed to get feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")


@router.patch("/feedback/{feedback_id}")
async def update_feedback(
    feedback_id: str = Path(..., description="Feedback ID to update"),
    request: UpdateFeedbackRequest = Body(...)
):
    """
    Update feedback status (mark as processed, link to created rule).

    **Path Parameters:**
    - `feedback_id`: The feedback ID to update

    **Request Body:**
    - `processed`: Mark as processed (default true)
    - `rule_created`: Rule ID if a rule was created from this feedback
    - `notes`: Optional admin notes

    **Returns:**
    - Success status
    """
    try:
        from bson import ObjectId

        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        # Build update document
        update_doc: Dict[str, Any] = {
            "processed": request.processed,
            "processed_at": datetime.utcnow().isoformat()
        }

        if request.rule_created:
            update_doc["rule_created"] = request.rule_created
        if request.notes:
            update_doc["admin_notes"] = request.notes

        result = await collection.update_one(
            {"_id": ObjectId(feedback_id)},
            {"$set": update_doc}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Feedback '{feedback_id}' not found")

        logger.info(f"Updated feedback {feedback_id}: processed={request.processed}")

        return {
            "success": True,
            "message": "Feedback updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {str(e)}")


@router.post("/feedback/{feedback_id}/generate-rule")
async def generate_rule_from_feedback(
    feedback_id: str = Path(..., description="Feedback ID to generate rule from")
):
    """
    Generate a rule suggestion from negative feedback using AI.

    **Path Parameters:**
    - `feedback_id`: The feedback ID to generate a rule from

    **Returns:**
    - Generated rule fields to populate the Add Rule form
    """
    try:
        from bson import ObjectId

        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        # Get the feedback
        feedback = await collection.find_one({"_id": ObjectId(feedback_id)})

        if not feedback:
            raise HTTPException(status_code=404, detail=f"Feedback '{feedback_id}' not found")

        # Build context for AI
        question = feedback.get("question", "")
        sql = feedback.get("sql", "")
        comment = feedback.get("comment", "")
        database = feedback.get("database", "")

        # Create prompt for rule generation
        prompt = f"""Based on this negative feedback about a SQL query, generate a rule to prevent similar issues.

Database: {database}
User Question: {question}
Generated SQL (that failed or was wrong): {sql}
User's Feedback Comment: {comment}

Please analyze what went wrong and generate a SQL rule in this exact JSON format:
{{
    "description": "Brief description of what this rule fixes",
    "type": "assistance or constraint",
    "trigger_keywords": ["keyword1", "keyword2"],
    "rule_text": "Detailed guidance for the LLM on how to handle this type of query correctly",
    "auto_fix_pattern": "optional regex pattern to fix common mistakes",
    "auto_fix_replacement": "optional replacement text",
    "example_question": "The question that demonstrates when this rule applies",
    "example_sql": "The correct SQL that should be generated"
}}

Focus on:
1. What pattern should trigger this rule (keywords users might use)
2. What the LLM should do instead
3. If possible, provide an exact match example"""

        pipeline = await get_pipeline()
        llm_service = pipeline.llm_service

        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available")

        response = await llm_service.generate(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.3
        )

        if not response or not response.get("response"):
            raise HTTPException(status_code=500, detail="LLM did not return a response")

        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.get("response", ""))
        if not json_match:
            raise HTTPException(status_code=500, detail="Could not parse AI response as JSON")

        try:
            rule_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON from AI: {e}")

        # Add database context
        rule_data["database"] = database
        rule_data["source_feedback_id"] = feedback_id

        logger.info(f"Generated rule suggestion from feedback {feedback_id}")

        return {
            "success": True,
            "rule": rule_data,
            "feedback": {
                "question": question,
                "sql": sql,
                "comment": comment
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate rule from feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate rule: {str(e)}")


@router.post("/save-example", response_model=SaveExampleResponse)
async def save_training_example(request: SaveExampleRequest):
    """
    Save a query as a training example.

    This endpoint stores a validated question/SQL pair in the sql_examples
    collection for use in training and few-shot learning.

    **Request Body:**
    - `question`: Natural language question
    - `sql`: Correct SQL query
    - `database`: Target database
    - `explanation`: Optional explanation of the SQL
    - `tags`: Tags for categorization

    **Returns:**
    - Success status and example ID
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_examples"]

        # Create example document with unique ID
        example_doc = {
            "id": str(uuid.uuid4()),
            "question": request.question,
            "question_normalized": request.question.lower().strip(),
            "sql": request.sql,
            "database": request.database,
            "explanation": request.explanation,
            "tags": request.tags,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "verified": True,
            "usage_count": 0
        }

        result = await collection.insert_one(example_doc)

        return SaveExampleResponse(
            success=True,
            message="Training example saved successfully",
            example_id=str(result.inserted_id)
        )

    except Exception as e:
        logger.error(f"Failed to save example: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save example: {str(e)}")


# ============================================================================
# Settings Endpoints
# ============================================================================

@router.get("/settings/{user_id}", response_model=UserSettingsResponse)
async def get_user_settings(user_id: str = Path(..., description="User ID")):
    """
    Get user settings for SQL query generation.

    **Path Parameters:**
    - `user_id`: User identifier

    **Returns:**
    - User settings or defaults if not found
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_user_settings"]

        # Find user settings
        settings_doc = await collection.find_one({"user_id": user_id})

        if settings_doc:
            settings = UserSettings(
                default_server=settings_doc.get("default_server"),
                default_database=settings_doc.get("default_database"),
                default_max_results=settings_doc.get("default_max_results", 100),
                enable_streaming=settings_doc.get("enable_streaming", True),
                temperature=settings_doc.get("temperature", 0.0),
                use_cache=settings_doc.get("use_cache", True)
            )
        else:
            # Return defaults
            settings = UserSettings()

        return UserSettingsResponse(success=True, settings=settings)

    except Exception as e:
        logger.error(f"Failed to get user settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get user settings: {str(e)}")


@router.post("/settings/{user_id}", response_model=UserSettingsResponse)
async def save_user_settings(
    user_id: str = Path(..., description="User ID"),
    settings: UserSettings = Body(..., description="User settings to save")
):
    """
    Save user settings for SQL query generation.

    **Path Parameters:**
    - `user_id`: User identifier

    **Request Body:**
    - User settings to save

    **Returns:**
    - Success status and saved settings
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_user_settings"]

        # Prepare settings document
        settings_doc = {
            "user_id": user_id,
            "default_server": settings.default_server,
            "default_database": settings.default_database,
            "default_max_results": settings.default_max_results,
            "enable_streaming": settings.enable_streaming,
            "temperature": settings.temperature,
            "use_cache": settings.use_cache,
            "updated_at": datetime.utcnow().isoformat()
        }

        # Upsert settings
        await collection.update_one(
            {"user_id": user_id},
            {"$set": settings_doc},
            upsert=True
        )

        return UserSettingsResponse(success=True, settings=settings)

    except Exception as e:
        logger.error(f"Failed to save user settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save user settings: {str(e)}")


# ============================================================================
# Rules Endpoints
# ============================================================================

@router.get("/rules/{database}", response_model=RulesResponse)
async def get_database_rules(
    database: str = Path(..., description="Database name"),
    include_global: bool = QueryParam(default=True, description="Include global rules")
):
    """
    Get SQL rules for a specific database.

    This endpoint retrieves all enabled rules for the specified database,
    optionally including global rules that apply to all databases.

    **Path Parameters:**
    - `database`: Target database name

    **Query Parameters:**
    - `include_global`: Whether to include global rules (default: true)

    **Returns:**
    - List of rules with metadata
    """
    try:
        rules_service = await RulesService.get_instance()
        rules = await rules_service.get_rules(database, include_global=include_global)

        rule_infos = []
        for rule in rules:
            rule_infos.append(RuleInfo(
                rule_id=rule.rule_id,
                description=rule.description,
                priority=rule.priority,
                enabled=rule.enabled,
                trigger_keywords=rule.trigger_keywords or [],
                trigger_tables=rule.trigger_tables or []
            ))

        return RulesResponse(
            success=True,
            database=database,
            rule_count=len(rule_infos),
            rules=rule_infos
        )

    except Exception as e:
        logger.error(f"Failed to get rules: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")


@router.get("/rules")
async def get_all_rules():
    """
    Get all SQL rules across all databases.

    **Returns:**
    - List of all rules with full details
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_rules"]
        cursor = collection.find({"enabled": True})
        rules = await cursor.to_list(length=None)

        rule_list = []
        for rule in rules:
            rule_list.append({
                "rule_id": rule.get("rule_id"),
                "database": rule.get("database"),
                "description": rule.get("description"),
                "type": rule.get("type"),
                "priority": rule.get("priority"),
                "enabled": rule.get("enabled", True),
                "trigger_keywords": rule.get("trigger_keywords", []),
                "trigger_tables": rule.get("trigger_tables", []),
                "trigger_columns": rule.get("trigger_columns", []),
                "rule_text": rule.get("rule_text"),
                "auto_fix": rule.get("auto_fix"),
                "example": rule.get("example"),
                "created_at": str(rule.get("created_at", "")),
                "updated_at": str(rule.get("updated_at", ""))
            })

        return {
            "success": True,
            "rule_count": len(rule_list),
            "rules": rule_list
        }

    except Exception as e:
        logger.error(f"Failed to get all rules: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")


@router.post("/rules", response_model=CreateRuleResponse)
async def create_rule(request: CreateRuleRequest):
    """
    Create a new SQL rule.

    **Request Body:**
    - `database`: Database name or '_global' for all databases
    - `rule_id`: Unique identifier for the rule
    - `description`: Human-readable description
    - `type`: 'assistance' or 'constraint'
    - `priority`: 'critical', 'high', or 'normal'
    - `trigger_keywords`: Keywords that activate this rule
    - `trigger_tables`: Tables that activate this rule
    - `trigger_columns`: Columns that activate this rule
    - `rule_text`: Guidance text for LLM
    - `auto_fix_pattern`: Optional regex pattern for auto-fix
    - `auto_fix_replacement`: Optional replacement string
    - `example_question`: Optional example question
    - `example_sql`: Optional example SQL

    **Returns:**
    - Success status and created rule ID
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_rules"]

        # Check if rule_id already exists
        existing = await collection.find_one({"rule_id": request.rule_id})
        if existing:
            raise HTTPException(status_code=400, detail=f"Rule with ID '{request.rule_id}' already exists")

        # Build rule document
        rule_doc = {
            "database": request.database,
            "rule_id": request.rule_id,
            "description": request.description,
            "type": request.type,
            "priority": request.priority,
            "enabled": request.enabled,
            "trigger_keywords": request.trigger_keywords,
            "trigger_tables": request.trigger_tables,
            "trigger_columns": request.trigger_columns,
            "rule_text": request.rule_text,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Add auto_fix if provided
        if request.auto_fix_pattern and request.auto_fix_replacement:
            rule_doc["auto_fix"] = {
                "pattern": request.auto_fix_pattern,
                "replacement": request.auto_fix_replacement
            }

        # Add example if provided
        if request.example_question and request.example_sql:
            rule_doc["example"] = {
                "question": request.example_question,
                "sql": request.example_sql
            }

        result = await collection.insert_one(rule_doc)

        # Auto-invalidate cache so the new rule is immediately available
        from sql_pipeline.services.rules_service import RulesService
        rules_service = await RulesService.get_instance()
        rules_service.invalidate_cache(request.database)

        logger.info(f"Created SQL rule: {request.rule_id} for database {request.database} (cache invalidated)")

        return CreateRuleResponse(
            success=True,
            message=f"Rule '{request.rule_id}' created successfully",
            rule_id=request.rule_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {str(e)}")


@router.post("/rules/invalidate-cache")
async def invalidate_rules_cache(database: Optional[str] = None):
    """
    Invalidate the rules cache to force a refresh from MongoDB.

    **Query Parameters:**
    - `database`: Optional database name to invalidate. If not provided, invalidates all caches.

    **Returns:**
    - Success status and message

    **Example:**
    ```
    POST /api/sql/rules/invalidate-cache
    POST /api/sql/rules/invalidate-cache?database=EWRCentral
    ```
    """
    try:
        from sql_pipeline.services.rules_service import RulesService

        rules_service = await RulesService.get_instance()
        rules_service.invalidate_cache(database)

        if database:
            message = f"Cache invalidated for database '{database}'"
        else:
            message = "All rules cache invalidated"

        logger.info(message)
        return {"success": True, "message": message}

    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str = Path(..., description="Rule ID to delete")):
    """
    Delete a SQL rule by ID.

    **Path Parameters:**
    - `rule_id`: The unique rule identifier to delete

    **Returns:**
    - Success status
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_rules"]

        result = await collection.delete_one({"rule_id": rule_id})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        # Invalidate all caches since we don't know which database the rule was for
        from sql_pipeline.services.rules_service import RulesService
        rules_service = await RulesService.get_instance()
        rules_service.invalidate_cache()

        logger.info(f"Deleted SQL rule: {rule_id} (cache invalidated)")

        return {
            "success": True,
            "message": f"Rule '{rule_id}' deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete rule: {str(e)}")


@router.put("/rules/{rule_id}")
async def update_rule(
    rule_id: str = Path(..., description="Rule ID to update"),
    request: CreateRuleRequest = Body(...)
):
    """
    Update an existing SQL rule.

    **Path Parameters:**
    - `rule_id`: The unique rule identifier to update

    **Request Body:**
    - Same fields as create rule

    **Returns:**
    - Success status
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_rules"]

        # Build update document
        update_doc = {
            "database": request.database,
            "rule_id": request.rule_id,
            "description": request.description,
            "type": request.type,
            "priority": request.priority,
            "enabled": request.enabled,
            "trigger_keywords": request.trigger_keywords,
            "trigger_tables": request.trigger_tables,
            "trigger_columns": request.trigger_columns,
            "rule_text": request.rule_text,
            "updated_at": datetime.utcnow()
        }

        # Add auto_fix if provided
        if request.auto_fix_pattern and request.auto_fix_replacement:
            update_doc["auto_fix"] = {
                "pattern": request.auto_fix_pattern,
                "replacement": request.auto_fix_replacement
            }
        else:
            update_doc["auto_fix"] = None

        # Add example if provided
        if request.example_question and request.example_sql:
            update_doc["example"] = {
                "question": request.example_question,
                "sql": request.example_sql
            }
        else:
            update_doc["example"] = None

        result = await collection.update_one(
            {"rule_id": rule_id},
            {"$set": update_doc}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        # Auto-invalidate cache so the updated rule is immediately available
        from sql_pipeline.services.rules_service import RulesService
        rules_service = await RulesService.get_instance()
        rules_service.invalidate_cache(request.database)

        logger.info(f"Updated SQL rule: {rule_id} (cache invalidated)")

        return {
            "success": True,
            "message": f"Rule '{rule_id}' updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update rule: {str(e)}")


class GenerateRuleRequest(BaseModel):
    """Request model for AI-assisted rule generation."""
    prompt: str = Field(..., description="Problem description from user")
    database: str = Field(..., description="Target database name")


@router.post("/generate-rule")
async def generate_rule_with_ai(request: GenerateRuleRequest):
    """
    Generate SQL rule fields using AI based on problem description.

    **Request Body:**
    - `prompt`: Description of the problem the rule should fix
    - `database`: Target database name

    **Returns:**
    - Generated rule fields to populate the form
    """
    try:
        pipeline = await get_pipeline()

        # Use the LLM service from the pipeline
        llm_service = pipeline.llm_service
        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available")

        # Generate rule using LLM
        response = await llm_service.generate(
            prompt=request.prompt,
            max_tokens=1000,
            temperature=0.3
        )

        if not response or not response.get("response"):
            raise HTTPException(status_code=500, detail="LLM did not return a response")

        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.get("response", ""))
        if not json_match:
            raise HTTPException(status_code=500, detail="Could not parse AI response as JSON")

        try:
            rule_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON from AI: {e}")

        logger.info(f"AI generated rule for database: {request.database}")

        return {
            "success": True,
            "rule": rule_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate rule with AI: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint for SQL query service.

    **Returns:**
    - Service status and component availability
    """
    try:
        mongo_service = MongoDBService.get_instance()
        mongo_initialized = mongo_service.is_initialized

        pipeline = await get_pipeline()
        pipeline_ready = pipeline is not None

        return {
            "status": "healthy",
            "service": "sql_query",
            "mongodb": mongo_initialized,
            "pipeline": pipeline_ready,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "sql_query",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# Schema Endpoints (replaces Node.js load-database and extract-schema-stream)
# ============================================================================

class SchemaCheckRequest(BaseModel):
    """Request to check if database schema exists."""
    database: str
    server: Optional[str] = None


class SchemaCheckResponse(BaseModel):
    """Response for schema check."""
    success: bool
    exists: bool
    database: str
    table_count: int = 0
    last_updated: Optional[str] = None
    message: str = ""


class SchemaExtractRequest(BaseModel):
    """Request to extract database schema."""
    database: str
    server: str
    username: Optional[str] = None
    password: Optional[str] = None
    domain: Optional[str] = None


@router.post("/schema/check", response_model=SchemaCheckResponse)
async def check_schema(request: SchemaCheckRequest):
    """
    Check if database schema exists in MongoDB.

    Replaces Node.js /api/sql/load-database endpoint.

    For EWR databases (e.g., EWR.Gin.CustomerName), normalizes to base product
    (e.g., EWR.Gin) since all customer databases share the same schema.

    **Request Body:**
    - `database`: Database name to check
    - `server`: Optional server name

    **Returns:**
    - Whether schema exists, table count, last update time
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["sql_schema_context"]

        # Normalize database name for lookup (EWR databases use base product name)
        db_lookup = normalize_database_name(request.database)
        logger.info(f"Schema check: '{request.database}' -> lookup as '{db_lookup}'")

        # Count tables for this database
        table_count = await collection.count_documents({"database": db_lookup})

        if table_count > 0:
            # Get last update time
            latest = await collection.find_one(
                {"database": db_lookup},
                sort=[("updated_at", -1)]
            )
            last_updated = latest.get("updated_at") if latest else None

            return SchemaCheckResponse(
                success=True,
                exists=True,
                database=request.database,
                table_count=table_count,
                last_updated=str(last_updated) if last_updated else None,
                message=f"Schema found with {table_count} tables"
            )
        else:
            return SchemaCheckResponse(
                success=True,
                exists=False,
                database=request.database,
                table_count=0,
                message="No schema found. Run schema extraction to analyze this database."
            )

    except Exception as e:
        logger.error(f"Schema check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema check failed: {str(e)}")


@router.post("/schema/extract-stream")
async def extract_schema_stream(request: SchemaExtractRequest):
    """
    Extract database schema with SSE streaming progress.

    Replaces Node.js /api/sql/extract-schema-stream endpoint.

    **Request Body:**
    - `database`: Target database
    - `server`: SQL Server hostname
    - `username`: SQL auth username
    - `password`: SQL auth password

    **Returns:**
    - SSE stream of extraction progress events
    """
    from sql_pipeline.models.query_models import SQLCredentials
    from sql_pipeline.services.execution_service import ExecutionService

    credentials = SQLCredentials(
        server=request.server,
        database=request.database,
        username=request.username or "EWRUser",
        password=request.password or "66a3904d69",
        domain=request.domain
    )

    async def event_generator():
        try:
            # Send start event
            yield f"event: status\ndata: {json.dumps({'stage': 'connecting', 'message': 'Connecting to database...'})}\n\n"

            execution_service = ExecutionService()

            # Test connection
            if not await execution_service.test_connection(credentials):
                yield f"event: error\ndata: {json.dumps({'error': 'Failed to connect to database'})}\n\n"
                return

            yield f"event: status\ndata: {json.dumps({'stage': 'connected', 'message': 'Connected. Extracting schema...'})}\n\n"

            # Extract tables
            tables_sql = """
                SELECT
                    t.TABLE_SCHEMA,
                    t.TABLE_NAME,
                    t.TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES t
                WHERE t.TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
            """

            tables_result = await execution_service.execute(tables_sql, credentials, max_results=1000)

            if not tables_result.success:
                yield f"event: error\ndata: {json.dumps({'error': f'Failed to extract tables: {tables_result.error}'})}\n\n"
                return

            tables = tables_result.data or []
            total_tables = len(tables)

            yield f"event: progress\ndata: {json.dumps({'stage': 'tables', 'message': f'Found {total_tables} tables', 'total': total_tables, 'current': 0})}\n\n"

            # Get MongoDB connection
            mongo_service = MongoDBService.get_instance()
            if not mongo_service.is_initialized:
                await mongo_service.initialize()

            collection = mongo_service.db["sql_schema_context"]
            db_lower = request.database.lower()

            # Process each table
            for idx, table in enumerate(tables):
                table_schema = table.get("TABLE_SCHEMA", "dbo")
                table_name = table.get("TABLE_NAME", "")
                full_name = f"{table_schema}.{table_name}"

                yield f"event: progress\ndata: {json.dumps({'stage': 'extracting', 'message': f'Extracting {full_name}...', 'total': total_tables, 'current': idx + 1})}\n\n"

                # Get columns
                columns_sql = f"""
                    SELECT
                        c.COLUMN_NAME,
                        c.DATA_TYPE,
                        c.IS_NULLABLE,
                        c.CHARACTER_MAXIMUM_LENGTH,
                        c.COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    WHERE c.TABLE_SCHEMA = '{table_schema}'
                    AND c.TABLE_NAME = '{table_name}'
                    ORDER BY c.ORDINAL_POSITION
                """

                columns_result = await execution_service.execute(columns_sql, credentials, max_results=500)
                columns = []

                if columns_result.success and columns_result.data:
                    for col in columns_result.data:
                        col_type = col.get("DATA_TYPE", "unknown")
                        max_len = col.get("CHARACTER_MAXIMUM_LENGTH")
                        if max_len and max_len > 0:
                            col_type = f"{col_type}({max_len})"

                        columns.append({
                            "name": col.get("COLUMN_NAME"),
                            "type": col_type,
                            "nullable": col.get("IS_NULLABLE") == "YES"
                        })

                # Get primary keys
                pk_sql = f"""
                    SELECT c.COLUMN_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE c
                        ON tc.CONSTRAINT_NAME = c.CONSTRAINT_NAME
                    WHERE tc.TABLE_SCHEMA = '{table_schema}'
                    AND tc.TABLE_NAME = '{table_name}'
                    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                """

                pk_result = await execution_service.execute(pk_sql, credentials, max_results=20)
                primary_keys = []
                if pk_result.success and pk_result.data:
                    primary_keys = [pk.get("COLUMN_NAME") for pk in pk_result.data]

                # Store in MongoDB
                table_doc = {
                    "database": db_lower,
                    "table_name": full_name,
                    "schema": table_schema,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": [],
                    "related_tables": [],
                    "updated_at": datetime.utcnow()
                }

                await collection.update_one(
                    {"database": db_lower, "table_name": full_name},
                    {"$set": table_doc},
                    upsert=True
                )

            yield f"event: complete\ndata: {json.dumps({'success': True, 'message': f'Schema extraction complete. Processed {total_tables} tables.', 'table_count': total_tables})}\n\n"

        except Exception as e:
            logger.error(f"Schema extraction error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# Save Correction Endpoint (replaces Node.js save-correction)
# ============================================================================

class SaveCorrectionRequest(BaseModel):
    """Request to save a SQL correction."""
    database: str
    question: str
    failed_sql: Optional[str] = None
    corrected_sql: str
    error_message: Optional[str] = None


class SaveCorrectionResponse(BaseModel):
    """Response for save correction."""
    success: bool
    message: str
    correction_id: Optional[str] = None


@router.post("/save-correction", response_model=SaveCorrectionResponse)
async def save_sql_correction(request: SaveCorrectionRequest):
    """
    Save a corrected SQL query after a failed attempt.

    Replaces Node.js /api/sql/save-correction endpoint.

    This creates a high-quality learning example by linking:
    - The original natural language question
    - The failed SQL (what NOT to do)
    - The corrected SQL (what TO do)

    **Request Body:**
    - `database`: Target database
    - `question`: Original natural language question
    - `failed_sql`: The SQL that failed (optional)
    - `corrected_sql`: The corrected SQL that works
    - `error_message`: Error from the failed SQL (optional)

    **Returns:**
    - Success status and correction ID
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        # Store in sql_examples collection (as verified examples)
        examples_collection = mongo_service.db["sql_examples"]

        # Create the example document with unique ID
        example_doc = {
            "id": str(uuid.uuid4()),
            "question": request.question,
            "question_normalized": request.question.lower().strip(),
            "sql": request.corrected_sql,
            "database": request.database,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "verified": True,
            "source": "user_correction",
            "usage_count": 0
        }

        result = await examples_collection.insert_one(example_doc)
        example_id = str(result.inserted_id)

        # Also store the correction pair in agent_learning for negative examples
        if request.failed_sql:
            learning_collection = mongo_service.db["agent_learning"]

            correction_doc = {
                "question": request.question,
                "question_normalized": request.question.lower().strip(),
                "database": request.database,
                "failed_sql": request.failed_sql,
                "corrected_sql": request.corrected_sql,
                "error_message": request.error_message,
                "type": "correction",
                "example_id": example_id,
                "created_at": datetime.utcnow().isoformat()
            }

            await learning_collection.insert_one(correction_doc)

            logger.info(f"Saved correction for database {request.database}: {request.question[:50]}...")

        return SaveCorrectionResponse(
            success=True,
            message="Correction saved successfully. The AI will learn from this example.",
            correction_id=example_id
        )

    except Exception as e:
        logger.error(f"Failed to save correction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save correction: {str(e)}")


@router.delete("/cache")
async def clear_cache(
    question: Optional[str] = QueryParam(None, description="Partial question text to match (regex)"),
    database: Optional[str] = QueryParam(None, description="Database name to filter by")
):
    """
    Clear cached queries from agent_learning collection.

    **Query Parameters:**
    - `question`: Optional regex pattern to match question text
    - `database`: Optional database name filter

    **Returns:**
    - Number of cache entries deleted

    **Examples:**
    - Clear all cache: DELETE /api/sql/cache
    - Clear specific: DELETE /api/sql/cache?question=tickets.*today
    - Clear by database: DELETE /api/sql/cache?database=EWRCentral
    """
    try:
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        # Build filter - match either question or question_normalized
        # The cache check uses question_normalized, so we need to match both
        filter_query = {}
        if question:
            # Use $or to match either field
            filter_query["$or"] = [
                {"question": {"$regex": question, "$options": "i"}},
                {"question_normalized": {"$regex": question.lower(), "$options": "i"}}
            ]
        if database:
            filter_query["database"] = {"$regex": f"^{database}$", "$options": "i"}

        # Delete matching entries
        result = await collection.delete_many(filter_query)

        message = f"Deleted {result.deleted_count} cache entries"
        if question:
            message += f" matching '{question}'"
        if database:
            message += f" for database '{database}'"

        logger.info(message)

        return {
            "success": True,
            "deleted_count": result.deleted_count,
            "message": message
        }

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
