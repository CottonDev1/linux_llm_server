"""
Git repository management API routes.

Provides endpoints for:
- Git repository CRUD operations
- Repository sync configuration
- Pull status tracking
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    GitRepositoryCreate, GitRepositoryUpdate, GitRepositoryPullUpdate,
    GitRepositoryResponse, GitRepositoryListResponse,
    SuccessResponse
)
from services.sqlite_service import get_sqlite_service, SQLiteService
from routes.auth_routes import get_current_user, require_admin

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/git-repositories", tags=["Git Repositories"])


def get_db() -> SQLiteService:
    """Dependency to get database service."""
    return get_sqlite_service()


# ============================================================================
# Git Repository Routes
# ============================================================================

@router.get("", response_model=GitRepositoryListResponse)
async def list_repositories(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    List all git repositories.
    """
    repos = db.get_all_git_repositories()
    return GitRepositoryListResponse(
        repositories=[GitRepositoryResponse.model_validate(r) for r in repos],
        total=len(repos)
    )


@router.post("", response_model=GitRepositoryResponse, status_code=status.HTTP_201_CREATED)
async def create_repository(
    repo_data: GitRepositoryCreate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Create a new git repository configuration (admin only).
    """
    # Check if name already exists
    existing = db.get_git_repository(repo_data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Repository with this name already exists"
        )
    
    repo = db.create_git_repository(repo_data)
    return GitRepositoryResponse.model_validate(repo)


@router.get("/auto-sync", response_model=GitRepositoryListResponse)
async def list_auto_sync_repositories(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    List repositories with auto-sync enabled.
    """
    repos = db.get_auto_sync_repositories()
    return GitRepositoryListResponse(
        repositories=[GitRepositoryResponse.model_validate(r) for r in repos],
        total=len(repos)
    )


@router.get("/needs-analysis", response_model=GitRepositoryListResponse)
async def list_repositories_needing_analysis(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    List repositories that need re-analysis.
    """
    repos = db.get_repositories_needing_analysis()
    return GitRepositoryListResponse(
        repositories=[GitRepositoryResponse.model_validate(r) for r in repos],
        total=len(repos)
    )


@router.get("/{name}", response_model=GitRepositoryResponse)
async def get_repository(
    name: str,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get git repository by name.
    """
    repo = db.get_git_repository(name)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    return GitRepositoryResponse.model_validate(repo)


@router.patch("/{name}", response_model=GitRepositoryResponse)
async def update_repository(
    name: str,
    updates: GitRepositoryUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Update git repository configuration (admin only).
    """
    repo = db.update_git_repository(name, updates)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    return GitRepositoryResponse.model_validate(repo)


@router.delete("/{name}", response_model=SuccessResponse)
async def delete_repository(
    name: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Delete git repository configuration (admin only).
    """
    success = db.delete_git_repository(name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    return SuccessResponse(message="Repository deleted successfully")


@router.post("/{name}/pull", response_model=GitRepositoryResponse)
async def update_repository_pull(
    name: str,
    pull_data: GitRepositoryPullUpdate,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Update repository after a pull operation.
    
    Records the pull time and commit information.
    """
    repo = db.update_git_repository_pull(name, pull_data)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    return GitRepositoryResponse.model_validate(repo)


@router.post("/{name}/analyzed", response_model=GitRepositoryResponse)
async def mark_repository_analyzed(
    name: str,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Mark repository as analyzed.
    
    Updates the last_analysis_date timestamp.
    """
    repo = db.update_git_repository_analysis_date(name)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    return GitRepositoryResponse.model_validate(repo)


@router.get("/{name}/token")
async def get_repository_token(
    name: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Get decrypted access token for repository (admin only).
    
    Returns the token in a secure response (not logged).
    """
    token = db.get_repository_access_token(name)
    if token is None:
        repo = db.get_git_repository(name)
        if not repo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Repository not found"
            )
        return {"token": None}
    
    return {"token": token}
