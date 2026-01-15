"""Code context routes for C# code analysis and search."""
from fastapi import APIRouter, Query, Body, HTTPException
from typing import Optional, List, Dict, Any

from mongodb import get_mongodb_service
from config import (
    COLLECTION_CODE_METHODS, COLLECTION_CODE_CLASSES, COLLECTION_CODE_CALLGRAPH
)

router = APIRouter(prefix="/code-context", tags=["Code Context"])


@router.post("")
async def store_code_context(
    document_id: str = Body(...),
    content: str = Body(...),
    metadata: Optional[Dict[str, Any]] = Body(default=None)
):
    """Store code context document"""
    mongodb = get_mongodb_service()
    result_id = await mongodb.store_code_context(
        document_id=document_id,
        content=content,
        metadata=metadata
    )
    return {"success": True, "document_id": result_id}


@router.post("/bulk")
async def store_code_context_bulk(
    documents: List[Dict[str, Any]] = Body(...)
):
    """Store multiple code context documents"""
    mongodb = get_mongodb_service()
    results = []
    for doc in documents:
        doc_id = await mongodb.store_code_context(
            document_id=doc.get("id"),
            content=doc.get("content"),
            metadata=doc.get("metadata")
        )
        results.append(doc_id)
    return {"success": True, "document_ids": results, "count": len(results)}


@router.get("/search")
async def search_code_context(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    project: Optional[str] = Query(None, description="Filter by project/database"),
    threshold: float = Query(0.4, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """Search code context using semantic similarity"""
    mongodb = get_mongodb_service()
    return await mongodb.search_code_context(
        query=query,
        limit=limit,
        project=project
    )


@router.get("/stats/summary")
async def get_code_context_stats():
    """Get code context statistics"""
    mongodb = get_mongodb_service()
    return await mongodb.get_code_context_stats()


@router.get("/{document_id}")
async def get_code_context(document_id: str):
    """Get code context by ID"""
    mongodb = get_mongodb_service()
    doc = await mongodb.get_code_context_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Code context not found")
    return doc


@router.delete("/{document_id}")
async def delete_code_context(document_id: str):
    """Delete code context by ID"""
    mongodb = get_mongodb_service()
    success = await mongodb.delete_code_context(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Code context not found")
    return {"success": True, "message": "Code context deleted"}


@router.delete("/project/{project_name}")
async def delete_project_code_data(project_name: str):
    """
    Delete all code analysis data for a project from code_methods, code_classes, and code_callgraph collections.
    This is used when removing a git repository from monitoring.
    """
    mongodb = get_mongodb_service()

    results = {
        "project": project_name,
        "deleted": {}
    }

    # Delete from code_methods
    try:
        methods_result = await mongodb.db[COLLECTION_CODE_METHODS].delete_many({"project": project_name})
        results["deleted"]["code_methods"] = methods_result.deleted_count
    except Exception as e:
        results["deleted"]["code_methods"] = f"Error: {str(e)}"

    # Delete from code_classes
    try:
        classes_result = await mongodb.db[COLLECTION_CODE_CLASSES].delete_many({"project": project_name})
        results["deleted"]["code_classes"] = classes_result.deleted_count
    except Exception as e:
        results["deleted"]["code_classes"] = f"Error: {str(e)}"

    # Delete from code_callgraph (uses caller_project)
    try:
        callgraph_result = await mongodb.db[COLLECTION_CODE_CALLGRAPH].delete_many({"caller_project": project_name})
        results["deleted"]["code_callgraph"] = callgraph_result.deleted_count
    except Exception as e:
        results["deleted"]["code_callgraph"] = f"Error: {str(e)}"

    # Also delete from code_context if exists
    try:
        context_result = await mongodb.db["code_context"].delete_many({"project": project_name})
        results["deleted"]["code_context"] = context_result.deleted_count
    except Exception as e:
        results["deleted"]["code_context"] = f"Error: {str(e)}"

    total_deleted = sum(v for v in results["deleted"].values() if isinstance(v, int))
    results["total_deleted"] = total_deleted
    results["success"] = True

    return results
