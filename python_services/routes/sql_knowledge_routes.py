"""SQL Knowledge routes for SQL schema and stored procedure context."""
from fastapi import APIRouter, Query, Body
from typing import Optional, List
from mongodb import get_mongodb_service

router = APIRouter(prefix="/sql-knowledge", tags=["SQL Knowledge"])


@router.post("")
async def store_sql_knowledge(
    knowledge_id: str = Body(...),
    content: str = Body(...),
    knowledge_type: str = Body(...),
    database_name: Optional[str] = Body(default=None),
    table_name: Optional[str] = Body(default=None),
    procedure_name: Optional[str] = Body(default=None),
    description: Optional[str] = Body(default=None),
    tags: Optional[List[str]] = Body(default=None)
):
    """Store SQL knowledge entry"""
    mongodb = get_mongodb_service()
    result_id = await mongodb.store_sql_knowledge(
        knowledge_id=knowledge_id,
        content=content,
        knowledge_type=knowledge_type,
        database_name=database_name,
        table_name=table_name,
        procedure_name=procedure_name,
        description=description,
        tags=tags
    )
    return {"success": True, "knowledge_id": result_id}


@router.get("/search")
async def search_sql_knowledge(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    knowledge_type: Optional[str] = Query(None, description="Filter by knowledge type"),
    database_name: Optional[str] = Query(None, description="Filter by database name")
):
    """Search SQL knowledge using semantic similarity"""
    mongodb = get_mongodb_service()
    return await mongodb.search_sql_knowledge(
        query=query,
        limit=limit,
        knowledge_type=knowledge_type,
        database_name=database_name
    )


@router.get("/stats/summary")
async def get_sql_knowledge_stats():
    """Get SQL knowledge statistics"""
    mongodb = get_mongodb_service()
    return await mongodb.get_sql_knowledge_stats()
