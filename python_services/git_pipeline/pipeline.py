"""
Git Sync Pipeline
=================

Main orchestration class for the Git Repository Sync Pipeline.

This pipeline coordinates:
1. Git pull operations to fetch latest changes
2. Roslyn code analysis to extract semantic information
3. Vector database import for semantic search

The pipeline can be run:
- Manually via API endpoints
- Scheduled via Prefect flows
- Triggered by webhooks (future)

Migrated from: src/routes/gitRoutes.js (POST /pull endpoint)
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from git_pipeline.models.pipeline_models import (
    PipelineConfig,
    RepositoryConfig,
    SyncRequest,
    SyncResult,
    RepositorySyncStatus,
    SyncStatus,
    AnalysisResult,
    ImportResult,
    DEFAULT_REPOSITORIES,
)
from git_pipeline.services.roslyn_service import RoslynService
from git_pipeline.services.code_import_service import CodeImportService

logger = logging.getLogger(__name__)


class GitSyncPipeline:
    """
    Main pipeline for synchronizing git repositories with code analysis.

    This class orchestrates the complete sync workflow:
    1. Pull latest changes from git remote
    2. Detect if changes were made
    3. Run Roslyn analysis on the repository
    4. Import analysis results to vector database
    5. Cleanup temporary files

    The pipeline uses the existing git_service for git operations and
    adds Roslyn analysis + vector import capabilities.

    Attributes:
        config: Pipeline configuration
        repositories: List of configured repositories
        roslyn_service: Service for Roslyn analysis
        import_service: Service for vector database import
        git_service: Service for git operations (lazy loaded)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        repositories: Optional[List[RepositoryConfig]] = None,
    ):
        """
        Initialize the Git Sync Pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
            repositories: List of repositories to sync. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.repositories = repositories or DEFAULT_REPOSITORIES

        # Initialize services
        self.roslyn_service = RoslynService(config=self.config)
        self.import_service = CodeImportService(config=self.config)

        # Git service is lazy loaded to avoid circular imports
        self._git_service = None

        logger.info(
            f"GitSyncPipeline initialized with {len(self.repositories)} repositories"
        )

    async def _get_git_service(self):
        """
        Lazy load the GitService.

        Returns:
            GitService instance
        """
        if self._git_service is None:
            from git_service.git_service import GitService
            self._git_service = GitService(git_root=self.config.git_root)
        return self._git_service

    # =========================================================================
    # Main Pipeline Methods
    # =========================================================================

    async def sync_all(
        self,
        request: Optional[SyncRequest] = None
    ) -> SyncResult:
        """
        Sync all configured repositories.

        This is the main entry point for syncing all repositories.

        Args:
            request: Sync request with options. Uses defaults if not provided.

        Returns:
            SyncResult with status for all repositories
        """
        request = request or SyncRequest()
        start_time = time.time()

        logger.info("Starting sync for all repositories")

        results: List[RepositorySyncStatus] = []
        repos_synced = 0
        repos_failed = 0
        repos_no_changes = 0

        for repo_config in self.repositories:
            if not repo_config.enabled:
                logger.info(f"Skipping disabled repository: {repo_config.name}")
                continue

            result = await self.sync_repository(
                repo_config=repo_config,
                analyze_changes=request.analyze_changes,
                import_to_vector_db=request.import_to_vector_db,
                force_analysis=request.force_analysis
            )

            results.append(result)

            if result.status == SyncStatus.SUCCESS:
                repos_synced += 1
            elif result.status == SyncStatus.NO_CHANGES:
                repos_no_changes += 1
            elif result.status in (SyncStatus.FAILED, SyncStatus.PARTIAL):
                repos_failed += 1

        total_duration = time.time() - start_time

        success = repos_failed == 0
        message = (
            f"Synced {repos_synced} repositories successfully, "
            f"{repos_no_changes} with no changes, {repos_failed} failed"
        )

        logger.info(f"Sync complete: {message}")

        return SyncResult(
            success=success,
            message=message,
            repositories=results,
            total_duration_seconds=total_duration,
            repos_synced=repos_synced,
            repos_failed=repos_failed,
            repos_no_changes=repos_no_changes
        )

    async def sync_repository(
        self,
        repo_config: RepositoryConfig,
        analyze_changes: bool = True,
        import_to_vector_db: bool = True,
        force_analysis: bool = False
    ) -> RepositorySyncStatus:
        """
        Sync a single repository.

        Performs git pull, optional Roslyn analysis, and vector import.

        Args:
            repo_config: Repository configuration
            analyze_changes: Whether to run Roslyn analysis
            import_to_vector_db: Whether to import to vector database
            force_analysis: Run analysis even if no changes detected

        Returns:
            RepositorySyncStatus with detailed status
        """
        start_time = time.time()
        repo_name = repo_config.name
        repo_path = repo_config.path

        logger.info(f"Starting sync for repository: {repo_name}")

        # Verify repository exists
        if not os.path.exists(repo_path):
            return RepositorySyncStatus(
                repository=repo_name,
                status=SyncStatus.FAILED,
                error=f"Repository path not found: {repo_path}",
                duration_seconds=time.time() - start_time
            )

        # Initialize result
        result = RepositorySyncStatus(
            repository=repo_name,
            status=SyncStatus.SUCCESS
        )

        try:
            # Step 1: Git pull
            git_service = await self._get_git_service()
            pull_result = await git_service.pull_repository_async(repo_path)

            result.pull_success = pull_result.success
            result.pull_output = pull_result.output
            result.has_changes = not pull_result.is_already_up_to_date

            if not pull_result.success:
                result.status = SyncStatus.FAILED
                result.error = pull_result.error or "Git pull failed"
                result.duration_seconds = time.time() - start_time
                return result

            # Check if we should run analysis
            should_analyze = (
                analyze_changes and
                (result.has_changes or force_analysis)
            )

            if not should_analyze:
                if not result.has_changes:
                    result.status = SyncStatus.NO_CHANGES
                    logger.info(f"{repo_name}: Already up to date, skipping analysis")
                result.duration_seconds = time.time() - start_time
                return result

            # Step 2: Roslyn analysis
            logger.info(f"{repo_name}: Running Roslyn analysis...")
            analysis_result = await self.roslyn_service.analyze_repository(
                repo_path=repo_path,
                repo_name=repo_name
            )
            result.analysis_result = analysis_result

            if not analysis_result.success:
                result.status = SyncStatus.PARTIAL
                result.error = f"Analysis failed: {analysis_result.error}"
                logger.warning(f"{repo_name}: Analysis failed but continuing")
                # Continue to try import if we have some data
                if not analysis_result.analysis_data:
                    result.duration_seconds = time.time() - start_time
                    return result

            # Step 3: Import to vector database
            if import_to_vector_db and analysis_result.analysis_data:
                logger.info(f"{repo_name}: Importing to vector database...")
                import_result = await self.import_service.import_analysis(
                    analysis=analysis_result.analysis_data,
                    db_name=repo_config.db_name
                )
                result.import_result = import_result

                if not import_result.success:
                    result.status = SyncStatus.PARTIAL
                    result.error = f"Import failed: {import_result.error}"

            # Cleanup temporary analysis file
            if analysis_result.output_file:
                self.roslyn_service.cleanup_output_file(analysis_result.output_file)

            result.duration_seconds = time.time() - start_time

            logger.info(
                f"{repo_name}: Sync complete in {result.duration_seconds:.2f}s "
                f"(status: {result.status.value})"
            )

            return result

        except Exception as e:
            logger.error(f"Sync failed for {repo_name}: {e}", exc_info=True)
            return RepositorySyncStatus(
                repository=repo_name,
                status=SyncStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time
            )

    async def sync_by_name(
        self,
        repo_name: str,
        request: Optional[SyncRequest] = None
    ) -> RepositorySyncStatus:
        """
        Sync a repository by name.

        Finds the repository configuration and syncs it.

        Args:
            repo_name: Name of the repository to sync
            request: Optional sync request with options

        Returns:
            RepositorySyncStatus for the repository
        """
        request = request or SyncRequest(repo=repo_name)

        # Find repository config
        repo_config = self.get_repository_config(repo_name)

        if not repo_config:
            return RepositorySyncStatus(
                repository=repo_name,
                status=SyncStatus.FAILED,
                error=f"Repository not found: {repo_name}",
                duration_seconds=0.0
            )

        return await self.sync_repository(
            repo_config=repo_config,
            analyze_changes=request.analyze_changes,
            import_to_vector_db=request.import_to_vector_db,
            force_analysis=request.force_analysis
        )

    # =========================================================================
    # Repository Management
    # =========================================================================

    def get_repository_config(self, repo_name: str) -> Optional[RepositoryConfig]:
        """
        Get configuration for a repository by name.

        Args:
            repo_name: Repository name (case-insensitive)

        Returns:
            RepositoryConfig or None if not found
        """
        repo_name_lower = repo_name.lower()
        for repo in self.repositories:
            if repo.name.lower() == repo_name_lower:
                return repo
        return None

    def list_repositories(self) -> List[RepositoryConfig]:
        """
        Get list of all configured repositories.

        Returns:
            List of RepositoryConfig objects
        """
        return self.repositories.copy()

    def add_repository(self, config: RepositoryConfig) -> bool:
        """
        Add a new repository to the pipeline.

        Args:
            config: Repository configuration

        Returns:
            True if added successfully
        """
        # Check if already exists
        existing = self.get_repository_config(config.name)
        if existing:
            logger.warning(f"Repository already exists: {config.name}")
            return False

        # Verify path exists
        if not os.path.exists(config.path):
            logger.error(f"Repository path not found: {config.path}")
            return False

        self.repositories.append(config)
        logger.info(f"Added repository: {config.name}")
        return True

    def remove_repository(self, repo_name: str) -> bool:
        """
        Remove a repository from the pipeline.

        Args:
            repo_name: Name of repository to remove

        Returns:
            True if removed, False if not found
        """
        repo_name_lower = repo_name.lower()
        for i, repo in enumerate(self.repositories):
            if repo.name.lower() == repo_name_lower:
                self.repositories.pop(i)
                logger.info(f"Removed repository: {repo_name}")
                return True
        return False

    # =========================================================================
    # Analysis-Only Methods
    # =========================================================================

    async def analyze_repository(
        self,
        repo_name: str
    ) -> AnalysisResult:
        """
        Run Roslyn analysis on a repository without git pull.

        Args:
            repo_name: Name of repository to analyze

        Returns:
            AnalysisResult with extracted code entities
        """
        repo_config = self.get_repository_config(repo_name)

        if not repo_config:
            return AnalysisResult(
                success=False,
                repository=repo_name,
                error=f"Repository not found: {repo_name}"
            )

        return await self.roslyn_service.analyze_repository(
            repo_path=repo_config.path,
            repo_name=repo_name
        )

    async def import_repository(
        self,
        repo_name: str,
        analysis: Optional[AnalysisResult] = None
    ) -> ImportResult:
        """
        Import analysis results to vector database.

        If analysis is not provided, runs analysis first.

        Args:
            repo_name: Name of repository
            analysis: Optional pre-existing analysis result

        Returns:
            ImportResult with import statistics
        """
        repo_config = self.get_repository_config(repo_name)

        if not repo_config:
            return ImportResult(
                success=False,
                repository=repo_name,
                error=f"Repository not found: {repo_name}"
            )

        # Run analysis if not provided
        if analysis is None or analysis.analysis_data is None:
            analysis = await self.analyze_repository(repo_name)

        if not analysis.success or not analysis.analysis_data:
            return ImportResult(
                success=False,
                repository=repo_name,
                error=analysis.error or "Analysis failed"
            )

        return await self.import_service.import_analysis(
            analysis=analysis.analysis_data,
            db_name=repo_config.db_name
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def check_roslyn_available(self) -> Dict[str, Any]:
        """
        Check if Roslyn analyzer is available.

        Returns:
            Dict with availability status and path
        """
        available = self.roslyn_service.is_analyzer_available()
        return {
            "available": available,
            "analyzer_path": self.roslyn_service.analyzer_path,
            "message": "Roslyn analyzer ready" if available else "Roslyn analyzer not found"
        }

    async def get_repository_status(
        self,
        repo_name: str
    ) -> Dict[str, Any]:
        """
        Get current status of a repository.

        Args:
            repo_name: Name of repository

        Returns:
            Dict with repository status information
        """
        repo_config = self.get_repository_config(repo_name)

        if not repo_config:
            return {
                "exists": False,
                "error": f"Repository not found: {repo_name}"
            }

        git_service = await self._get_git_service()

        # Get git info
        info = await git_service.get_repository_info(repo_config.path)

        return {
            "exists": True,
            "name": repo_config.name,
            "path": repo_config.path,
            "db_name": repo_config.db_name,
            "enabled": repo_config.enabled,
            "git_status": info.status,
            "last_sync": info.last_sync,
            "recent_commits": [c.model_dump() for c in info.recent_commits[:5]]
        }


# Convenience function for creating pipeline with defaults
def create_pipeline(
    config: Optional[PipelineConfig] = None,
    repositories: Optional[List[RepositoryConfig]] = None
) -> GitSyncPipeline:
    """
    Create a GitSyncPipeline with optional configuration.

    Args:
        config: Pipeline configuration
        repositories: Repository configurations

    Returns:
        Configured GitSyncPipeline instance
    """
    return GitSyncPipeline(config=config, repositories=repositories)


# Synchronous convenience functions
def sync_all_sync(request: Optional[SyncRequest] = None) -> SyncResult:
    """
    Sync all repositories synchronously.

    Args:
        request: Sync request options

    Returns:
        SyncResult with status for all repositories
    """
    pipeline = create_pipeline()
    return asyncio.run(pipeline.sync_all(request))


def sync_repository_sync(
    repo_name: str,
    request: Optional[SyncRequest] = None
) -> RepositorySyncStatus:
    """
    Sync a single repository synchronously.

    Args:
        repo_name: Name of repository to sync
        request: Sync request options

    Returns:
        RepositorySyncStatus for the repository
    """
    pipeline = create_pipeline()
    return asyncio.run(pipeline.sync_by_name(repo_name, request))
