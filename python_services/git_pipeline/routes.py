"""
Git Pipeline API Routes
=======================

FastAPI routes for the Git Repository Sync Pipeline.

Endpoints:
- POST /pull - Execute git pull for all repositories
- POST /update-context - Update git context from repositories
- GET /repositories - List all configured repositories
- POST /repositories - Add a new repository
- POST /repositories/{id}/sync - Sync a specific repository
- DELETE /repositories/{id} - Remove a repository

Migrated from: src/routes/gitRoutes.js
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Body

from git_pipeline.models.pipeline_models import (
    SyncRequest,
    SyncResult,
    RepositorySyncStatus,
    RepositoryConfig,
    RepositoryListResponse,
    RepositoryAddRequest,
    RepositoryAddResponse,
    UpdateContextRequest,
)
from git_pipeline.pipeline import GitSyncPipeline, create_pipeline

logger = logging.getLogger(__name__)

_git_pipeline: Optional[GitSyncPipeline] = None


def get_git_pipeline() -> GitSyncPipeline:
    """Get or create the git pipeline instance."""
    global _git_pipeline
    if _git_pipeline is None:
        _git_pipeline = create_pipeline()
    return _git_pipeline


def create_git_routes() -> APIRouter:
    """Create FastAPI router with git pipeline endpoints."""
    router = APIRouter()

    @router.post("/pull", response_model=SyncResult, summary="Pull All Repositories")
    async def pull_repositories(request: SyncRequest = Body(default=SyncRequest())) -> SyncResult:
        """Pull updates from all configured repositories."""
        try:
            logger.info("Starting git pull for all repositories...")
            pipeline = get_git_pipeline()
            if request.repo:
                result = await pipeline.sync_by_name(request.repo, request)
                return SyncResult(
                    success=result.status.value in ["success", "no_changes"],
                    message=f"Synced {request.repo}: {result.status.value}",
                    repositories=[result],
                    total_duration_seconds=result.duration_seconds,
                    repos_synced=1 if result.status.value == "success" else 0,
                    repos_failed=1 if result.status.value == "failed" else 0,
                    repos_no_changes=1 if result.status.value == "no_changes" else 0
                )
            return await pipeline.sync_all(request)
        except Exception as e:
            logger.error(f"Git pull process failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/update-context", summary="Update Git Context")
    async def update_context(request: UpdateContextRequest = Body(default=UpdateContextRequest())):
        """Update context from git repositories."""
        try:
            return {
                "success": True,
                "message": "Context update initiated",
                "repo": request.repo,
                "changed_only": request.changed_only
            }
        except Exception as e:
            logger.error(f"Context update failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/repositories", response_model=RepositoryListResponse, summary="List Repositories")
    async def list_repositories() -> RepositoryListResponse:
        """List all configured repositories."""
        try:
            pipeline = get_git_pipeline()
            return RepositoryListResponse(success=True, repositories=pipeline.list_repositories())
        except Exception as e:
            logger.error(f"Error listing repositories: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/repositories", response_model=RepositoryAddResponse, summary="Add Repository")
    async def add_repository(request: RepositoryAddRequest) -> RepositoryAddResponse:
        """Add a new repository to monitoring."""
        try:
            pipeline = get_git_pipeline()
            repo_config = RepositoryConfig(
                name=request.name,
                path=request.path,
                db_name=request.name.lower(),
                display_name=request.name
            )
            success = pipeline.add_repository(repo_config)
            if not success:
                return RepositoryAddResponse(success=False, error="Repository already exists or path not found")
            return RepositoryAddResponse(success=True, repository=repo_config)
        except Exception as e:
            logger.error(f"Error adding repository: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/repositories/{repo_name}/sync", response_model=RepositorySyncStatus, summary="Sync Repository")
    async def sync_repository(repo_name: str, request: SyncRequest = Body(default=SyncRequest())) -> RepositorySyncStatus:
        """Sync a specific repository."""
        try:
            pipeline = get_git_pipeline()
            request.repo = repo_name
            return await pipeline.sync_by_name(repo_name, request)
        except Exception as e:
            logger.error(f"Error syncing repository: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/repositories/{repo_name}", summary="Remove Repository")
    async def remove_repository(repo_name: str):
        """Remove a repository from monitoring."""
        try:
            pipeline = get_git_pipeline()
            if not pipeline.remove_repository(repo_name):
                raise HTTPException(status_code=404, detail=f"Repository not found: {repo_name}")
            return {"success": True, "message": f"Repository {repo_name} removed"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error removing repository: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/health", summary="Health Check")
    async def health_check():
        """Check git pipeline health."""
        try:
            pipeline = get_git_pipeline()
            return {
                "healthy": True,
                "pipeline": "git_sync",
                "repositories_count": len(pipeline.repositories)
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    return router


def register_git_routes(app, prefix: str = "/api/admin/git"):
    """Register git routes with a FastAPI app."""
    router = create_git_routes()
    app.include_router(router, prefix=prefix, tags=["Git"])
    logger.info(f"Git routes registered at {prefix}")
