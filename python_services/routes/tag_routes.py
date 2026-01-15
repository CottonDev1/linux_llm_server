"""
Tag management API routes.

Provides endpoints for:
- Tag CRUD operations
- Document tagging
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    TagCreate, TagUpdate, TagResponse, TagListResponse,
    DocumentTagCreate, DocumentTagsResponse,
    SuccessResponse
)
from services.sqlite_service import get_sqlite_service, SQLiteService
from routes.auth_routes import get_current_user, require_admin

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/tags", tags=["Tags"])


def get_db() -> SQLiteService:
    """Dependency to get database service."""
    return get_sqlite_service()


# ============================================================================
# Tag Routes
# ============================================================================

@router.get("", response_model=TagListResponse)
async def list_tags(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    List all tags.
    """
    tags = db.get_all_tags()
    return TagListResponse(
        tags=[TagResponse.model_validate(t) for t in tags],
        total=len(tags)
    )


@router.post("", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
async def create_tag(
    tag_data: TagCreate,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Create a new tag.
    """
    # Check if name already exists
    existing = db.get_tag_by_name(tag_data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tag with this name already exists"
        )
    
    tag = db.create_tag(tag_data)
    return TagResponse.model_validate(tag)


@router.get("/{tag_id}", response_model=TagResponse)
async def get_tag(
    tag_id: int,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get tag by ID.
    """
    tag = db.get_tag_by_id(tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return TagResponse.model_validate(tag)


@router.patch("/{tag_id}", response_model=TagResponse)
async def update_tag(
    tag_id: int,
    updates: TagUpdate,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Update a tag.
    """
    # Check for name conflict if updating name
    if updates.name:
        existing = db.get_tag_by_name(updates.name)
        if existing and existing.id != tag_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Tag with this name already exists"
            )
    
    tag = db.update_tag(tag_id, updates)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return TagResponse.model_validate(tag)


@router.delete("/{tag_id}", response_model=SuccessResponse)
async def delete_tag(
    tag_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Delete a tag (admin only).
    
    Also removes all document-tag associations.
    """
    success = db.delete_tag(tag_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    return SuccessResponse(message="Tag deleted successfully")


# ============================================================================
# Document Tagging Routes
# ============================================================================

@router.get("/documents/{document_id}", response_model=DocumentTagsResponse)
async def get_document_tags(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get all tags for a document.
    """
    tags = db.get_document_tags(document_id)
    return DocumentTagsResponse(
        document_id=document_id,
        tags=[TagResponse.model_validate(t) for t in tags]
    )


@router.post("/documents/{document_id}/{tag_id}", response_model=SuccessResponse)
async def tag_document(
    document_id: str,
    tag_id: int,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Add a tag to a document.
    """
    # Verify tag exists
    tag = db.get_tag_by_id(tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    
    db.tag_document(document_id, tag_id)
    return SuccessResponse(message="Document tagged successfully")


@router.delete("/documents/{document_id}/{tag_id}", response_model=SuccessResponse)
async def untag_document(
    document_id: str,
    tag_id: int,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Remove a tag from a document.
    """
    success = db.untag_document(document_id, tag_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag association not found"
        )
    return SuccessResponse(message="Tag removed from document")


@router.get("/{tag_id}/documents")
async def get_documents_by_tag(
    tag_id: int,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get all document IDs with a specific tag.
    """
    # Verify tag exists
    tag = db.get_tag_by_id(tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )
    
    document_ids = db.get_documents_by_tag(tag_id)
    return {
        "tag_id": tag_id,
        "tag_name": tag.name,
        "document_ids": document_ids,
        "total": len(document_ids)
    }
