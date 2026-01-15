"""
SQL RAG routes for SQL query generation, examples, and RAG search.

Provides endpoints for:
- SQL examples storage and retrieval (few-shot learning)
- Failed queries tracking (error learning)
- SQL corrections management
- Schema context for RAG
- Stored procedures search
- Comprehensive SQL context retrieval
"""
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Request, Query, HTTPException, Body
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COLLECTION_SQL_EXAMPLES, COLLECTION_SQL_FAILED_QUERIES,
    COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES,
    COLLECTION_CODE_CLASSES, COLLECTION_CODE_METHODS,
    COLLECTION_CODE_CALLGRAPH, COLLECTION_DOCUMENTS
)
from mongodb import get_mongodb_service
from database_name_parser import normalize_database_name
from log_service import log_pipeline, log_error


# Create router with /sql prefix
router = APIRouter(prefix="/sql", tags=["SQL RAG"])


# ============================================================================
# Pydantic Models
# ============================================================================

class CorrectionCreate(BaseModel):
    """Model for creating SQL corrections."""
    database: str
    original_prompt: str
    original_sql: str = ""
    error_message: str = ""
    corrected_prompt: Optional[str] = None  # Defaults to original_prompt if not provided
    corrected_sql: str
    correction_notes: str = ""
    correction_type: str = "unknown"
    tables_used: Optional[List[str]] = None


# ============================================================================
# SQL Examples (Few-shot Learning)
# ============================================================================

@router.post("/examples")
async def store_sql_example(
    database: str = Body(...),
    prompt: str = Body(..., description="Natural language question"),
    sql: str = Body(..., description="Correct SQL query"),
    response: Optional[str] = Body(default=None, description="Query result description"),
    tables_used: Optional[List[str]] = Body(default=None)
):
    """
    Store a successful SQL query example for few-shot learning.
    Key: Only the question is embedded for semantic search.
    """
    mongodb = get_mongodb_service()
    example_id = await mongodb.store_sql_example(
        database=database,
        prompt=prompt,
        sql=sql,
        response=response,
        tables_used=tables_used
    )
    return {"success": True, "example_id": example_id}


@router.get("/examples/search")
async def search_sql_examples(
    query: str = Query(..., description="Natural language question"),
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(3, ge=1, le=10, description="Number of examples (default 3)")
):
    """
    Search for similar SQL examples for few-shot prompting.
    Returns question-SQL pairs most similar to the query.
    """
    mongodb = get_mongodb_service()
    return await mongodb.search_sql_examples(query, database, limit)


# ============================================================================
# SQL Failed Queries (Error Learning)
# ============================================================================

@router.post("/failed-queries")
async def store_failed_query(
    database: str = Body(...),
    prompt: str = Body(..., description="Natural language question"),
    sql: str = Body(..., description="Failed SQL query"),
    error: str = Body(..., description="Error message"),
    tables_involved: Optional[List[str]] = Body(default=None)
):
    """Store a failed SQL query to avoid repeating mistakes."""
    mongodb = get_mongodb_service()
    failed_id = await mongodb.store_failed_query(
        database=database,
        prompt=prompt,
        sql=sql,
        error=error,
        tables_involved=tables_involved
    )
    return {"success": True, "failed_id": failed_id}


@router.get("/failed-queries/search")
async def search_failed_queries(
    query: str = Query(..., description="Natural language question"),
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(2, ge=1, le=10, description="Number of results (default 2)")
):
    """Search for similar failed queries to avoid repeating mistakes."""
    mongodb = get_mongodb_service()
    return await mongodb.search_failed_queries(query, database, limit)


# ============================================================================
# SQL Corrections (User-provided corrections for RAG improvement)
# ============================================================================

@router.post("/corrections")
async def store_sql_correction(correction: CorrectionCreate):
    """
    Store a user-provided SQL correction for RAG improvement.
    Corrections start as 'pending' and can be validated/promoted later.
    """
    mongodb = get_mongodb_service()
    correction_id = await mongodb.store_sql_correction(
        database=correction.database,
        original_prompt=correction.original_prompt,
        original_sql=correction.original_sql,
        error_message=correction.error_message,
        corrected_prompt=correction.corrected_prompt or correction.original_prompt,
        corrected_sql=correction.corrected_sql,
        correction_notes=correction.correction_notes,
        correction_type=correction.correction_type,
        tables_used=correction.tables_used
    )
    return {"success": True, "correction_id": correction_id}


@router.get("/corrections/search")
async def search_sql_corrections(
    query: str = Query(..., description="Natural language question"),
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(3, ge=1, le=10, description="Number of corrections"),
    status: Optional[str] = Query(None, description="Filter by status: pending, validated, promoted")
):
    """Search for relevant SQL corrections to augment examples."""
    mongodb = get_mongodb_service()
    status_filter = [status] if status else None
    return await mongodb.search_sql_corrections(query, database, limit, status_filter)


@router.get("/corrections/pending")
async def get_pending_corrections(
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """Get corrections awaiting review."""
    mongodb = get_mongodb_service()
    return await mongodb.get_corrections_for_review(database, "pending", limit, offset)


@router.post("/corrections/{correction_id}/validate")
async def validate_correction(
    correction_id: str,
    reviewer_id: Optional[str] = Body(default=None),
    approve: bool = Body(default=True)
):
    """Approve or reject a correction."""
    mongodb = get_mongodb_service()
    status = "validated" if approve else "rejected"
    success = await mongodb.update_correction_status(correction_id, status, reviewer_id)
    return {"success": success, "status": status}


@router.post("/corrections/{correction_id}/promote")
async def promote_correction(
    correction_id: str,
    reviewer_id: Optional[str] = Body(default=None)
):
    """Promote a validated correction to the sql_examples collection."""
    mongodb = get_mongodb_service()
    example_id = await mongodb.promote_correction_to_example(correction_id, reviewer_id)
    if example_id:
        return {"success": True, "example_id": example_id}
    return {"success": False, "error": "Correction not found or not validated"}


@router.get("/corrections/stats")
async def get_correction_stats(
    database: Optional[str] = Query(None, description="Filter by database")
):
    """Get statistics about SQL corrections."""
    mongodb = get_mongodb_service()
    return await mongodb.get_correction_stats(database)


# ============================================================================
# SQL Schema Context
# ============================================================================

@router.post("/schema-context")
async def store_schema_context(
    database: str = Body(...),
    table_name: str = Body(..., description="Full table name (e.g., dbo.Orders)"),
    schema_info: Dict[str, Any] = Body(..., description="Schema with columns, primaryKeys, foreignKeys, relatedTables, sampleValues"),
    summary: Optional[str] = Body(default=None, description="LLM-generated table summary")
):
    """
    Store enhanced schema context with FK relationships.
    Include LLM summary and sample values for best results.
    """
    mongodb = get_mongodb_service()
    schema_id = await mongodb.store_schema_context(
        database=database,
        table_name=table_name,
        schema_info=schema_info,
        summary=summary
    )
    return {"success": True, "schema_id": schema_id}


@router.get("/schema-context/search")
async def search_schema_context(
    query: str = Query(..., description="Natural language question"),
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(5, ge=1, le=50, description="Number of schemas (default 5)")
):
    """
    Search for relevant schema context using semantic similarity.
    Returns table schemas most relevant to the query.
    """
    mongodb = get_mongodb_service()
    return await mongodb.search_schema_context(query, database, limit)


@router.get("/schema-context/{database}/{table_name}")
async def get_schema_by_table(database: str, table_name: str):
    """Get schema context for a specific table."""
    mongodb = get_mongodb_service()
    schema = await mongodb.get_schema_by_table(database, table_name)
    if not schema:
        raise HTTPException(status_code=404, detail="Schema not found")
    return schema


# ============================================================================
# SQL Stored Procedures
# ============================================================================

@router.post("/stored-procedures")
async def store_stored_procedure(
    database: str = Body(...),
    procedure_name: str = Body(...),
    procedure_info: Dict[str, Any] = Body(..., description="Dict with schema, parameters, definition, summary, keywords")
):
    """
    Store stored procedure with semantic search support.
    Include summary for best retrieval results.
    """
    mongodb = get_mongodb_service()
    sp_id = await mongodb.store_stored_procedure(
        database=database,
        procedure_name=procedure_name,
        procedure_info=procedure_info
    )
    return {"success": True, "procedure_id": sp_id}


@router.get("/stored-procedures/search")
async def search_stored_procedures(
    query: str = Query(..., description="Natural language question"),
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(3, ge=1, le=20, description="Number of procedures (default 3)")
):
    """
    Search for stored procedures by semantic similarity.
    Finds procedures that perform operations similar to what's needed.
    """
    mongodb = get_mongodb_service()
    return await mongodb.search_stored_procedures(query, database, limit)


# ============================================================================
# Comprehensive Context (Main RAG Entry Point)
# ============================================================================

@router.get("/comprehensive-context")
async def get_comprehensive_context(
    query: str = Query(..., description="Natural language question"),
    database: str = Query(..., description="Database identifier"),
    schema_limit: int = Query(20, ge=1, le=100),
    example_limit: int = Query(5, ge=1, le=20),
    failed_limit: int = Query(3, ge=1, le=10),
    sp_limit: int = Query(5, ge=1, le=20)
):
    """
    Get comprehensive SQL context for text-to-SQL generation.
    This is the main entry point for the RAG pipeline.

    Returns schema context, examples, failed queries, and stored procedures
    all in one call for building the LLM prompt.
    """
    mongodb = get_mongodb_service()
    return await mongodb.get_comprehensive_sql_context(
        query=query,
        database=database,
        schema_limit=schema_limit,
        example_limit=example_limit,
        failed_limit=failed_limit,
        sp_limit=sp_limit
    )


@router.get("/rag-stats")
async def get_sql_rag_stats():
    """Get statistics for all SQL RAG collections."""
    mongodb = get_mongodb_service()
    return await mongodb.get_sql_rag_stats()


@router.get("/database-stats/{database}")
async def get_database_stats(database: str):
    """
    Get schema and stored procedure counts for a specific database.
    Used for checking if a database has been extracted.
    """
    mongodb = get_mongodb_service()
    normalized_db = normalize_database_name(database)

    schema_coll = mongodb.db[COLLECTION_SQL_SCHEMA_CONTEXT]
    proc_coll = mongodb.db[COLLECTION_SQL_STORED_PROCEDURES]

    schema_count = await schema_coll.count_documents({"database": normalized_db})
    proc_count = await proc_coll.count_documents({"database": normalized_db})

    return {
        "database": normalized_db,
        "original_name": database,
        "schema_count": schema_count,
        "procedure_count": proc_count,
        "has_data": schema_count > 0 or proc_count > 0
    }


@router.get("/pipeline-stats")
async def get_sql_pipeline_stats(request: Request):
    """Get comprehensive SQL pipeline statistics for dashboard."""
    user_ip = request.client.host if request.client else "Unknown"
    mongodb = get_mongodb_service()

    try:
        # Get basic RAG stats
        rag_stats = await mongodb.get_sql_rag_stats()

        # Get failed queries with correction rate
        failed_queries_coll = mongodb.db[COLLECTION_SQL_FAILED_QUERIES]
        total_failed = await failed_queries_coll.count_documents({})
        with_corrections = await failed_queries_coll.count_documents({"corrected_sql": {"$exists": True, "$ne": None}})
        correction_rate = (with_corrections / total_failed * 100) if total_failed > 0 else 0

        # Get examples by database
        examples_coll = mongodb.db[COLLECTION_SQL_EXAMPLES]
        db_pipeline = [
            {"$group": {"_id": "$database", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        examples_by_db = {}
        async for doc in examples_coll.aggregate(db_pipeline):
            examples_by_db[doc["_id"] or "Unknown"] = doc["count"]

        # Get schema coverage by database
        schema_coll = mongodb.db[COLLECTION_SQL_SCHEMA_CONTEXT]
        schema_by_db = {}
        async for doc in schema_coll.aggregate(db_pipeline):
            schema_by_db[doc["_id"] or "Unknown"] = doc["count"]

        # Get stored procedures by database
        proc_coll = mongodb.db[COLLECTION_SQL_STORED_PROCEDURES]
        procs_by_db = {}
        async for doc in proc_coll.aggregate(db_pipeline):
            procs_by_db[doc["_id"] or "Unknown"] = doc["count"]

        # Get recent failed queries
        recent_failed = []
        async for doc in failed_queries_coll.find().sort("timestamp", -1).limit(10):
            recent_failed.append({
                "id": doc.get("id", str(doc.get("_id", ""))),
                "query": doc.get("natural_language_query", "")[:100],
                "database": doc.get("database", "Unknown"),
                "error": doc.get("error_message", "")[:100],
                "has_correction": bool(doc.get("corrected_sql")),
                "timestamp": doc.get("timestamp", "").isoformat() if hasattr(doc.get("timestamp", ""), "isoformat") else str(doc.get("timestamp", ""))
            })

        # Get collection sizes in MB for chart
        collection_sizes_mb = {}
        collections_to_check = [
            (COLLECTION_SQL_SCHEMA_CONTEXT, "SQL Schemas"),
            (COLLECTION_SQL_STORED_PROCEDURES, "SQL Procedures"),
            (COLLECTION_SQL_EXAMPLES, "SQL Examples"),
            (COLLECTION_SQL_FAILED_QUERIES, "Failed Queries"),
            (COLLECTION_CODE_CLASSES, "C# Classes"),
            (COLLECTION_CODE_METHODS, "C# Methods"),
            (COLLECTION_CODE_CALLGRAPH, "Call Graph"),
            (COLLECTION_DOCUMENTS, "Documents")
        ]
        for coll_name, display_name in collections_to_check:
            try:
                coll_stats = await mongodb.db.command("collStats", coll_name)
                size_mb = coll_stats.get("size", 0) / (1024 * 1024)
                collection_sizes_mb[display_name] = round(size_mb, 2)
            except Exception:
                collection_sizes_mb[display_name] = 0

        log_pipeline("SQLPipeline", user_ip, "Pipeline stats retrieved",
                    details={"total_examples": rag_stats.get("examples", 0)})

        return {
            "collection_counts": rag_stats,
            "failed_queries": {
                "total": total_failed,
                "with_corrections": with_corrections,
                "correction_rate": round(correction_rate, 1)
            },
            "examples_by_database": examples_by_db,
            "schema_by_database": schema_by_db,
            "procedures_by_database": procs_by_db,
            "recent_failed_queries": recent_failed,
            "collection_sizes_mb": collection_sizes_mb
        }

    except Exception as e:
        log_error("SQLPipeline", user_ip, "Failed to get pipeline stats", str(e))
        raise HTTPException(status_code=500, detail=str(e))
