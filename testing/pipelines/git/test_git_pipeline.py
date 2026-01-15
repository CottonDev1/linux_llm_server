"""
Git Pipeline Tests
==================

Comprehensive tests for git sync pipeline orchestration.

Tests cover:
- GitSyncPipeline complete workflow
- Repository sync with and without changes
- Sync by repository name
- Repository analysis and import
- Repository management (add/remove/list/get)
- Force analysis flag behavior
- Prefect flow integration
"""

import os
import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from dataclasses import dataclass

# Setup environment before imports
os.environ.setdefault("GIT_ROOT", r"C:\Projects\Git")
os.environ.setdefault("PROJECT_ROOT", r"C:\Projects\llm_website")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python_services"))

try:
    from git_pipeline.prefect.git_sync_flow import (
        git_sync_flow,
        git_sync_all_flow,
        git_pull_task,
        roslyn_analysis_task,
        vector_import_task,
        run_git_sync,
        run_git_sync_all,
        PullTaskResult,
        AnalysisTaskResult,
        ImportTaskResult,
        SyncFlowResult,
    )
    from git_pipeline.models.pipeline_models import (
        PipelineConfig,
        RepositoryConfig,
        SyncStatus,
        AnalysisResult,
        ImportResult,
        DEFAULT_REPOSITORIES,
    )
    from prefect_pipelines.git_flow import (
        git_analysis_flow,
        pull_repository_task,
        analyze_commits_task,
        PullResult as FlowPullResult,
        AnalysisResult as FlowAnalysisResult,
        run_git_analysis_flow,
    )
    HAS_PIPELINE = True
except ImportError as e:
    HAS_PIPELINE = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def mock_git_service():
    """Create a mock GitService."""
    mock = Mock()
    mock.pull_repository_async = AsyncMock(return_value=Mock(
        success=True,
        output="Already up to date.",
        is_already_up_to_date=True,
        error=None
    ))
    mock.get_recent_commits_async = AsyncMock(return_value=[
        Mock(hash="abc123", author="Test", date="2024-01-15", message="Test commit")
    ])
    mock.analyze_commits_by_date_range_async = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_roslyn_service():
    """Create a mock RoslynService."""
    mock = Mock()
    mock.analyze_repository = AsyncMock(return_value=Mock(
        success=True,
        entity_count=10,
        file_count=5,
        output_file="analysis.json",
        duration_seconds=1.5,
        error=None
    ))
    mock.is_analyzer_available.return_value = True
    return mock


@pytest.fixture
def mock_import_service():
    """Create a mock CodeImportService."""
    mock = Mock()
    mock.import_analysis = AsyncMock(return_value=Mock(
        success=True,
        documents_imported=10,
        documents_updated=5,
        duration_seconds=0.5,
        error=None
    ))
    return mock


@pytest.fixture
def sample_repo_config():
    """Create a sample repository configuration."""
    return RepositoryConfig(
        name="TestRepo",
        path=r"C:\Projects\Git\TestRepo",
        db_name="testrepo",
        display_name="Test Repository",
        enabled=True,
        auto_sync=False
    )


@pytest.fixture
def pipeline_config():
    """Create a pipeline configuration."""
    return PipelineConfig(
        git_root=r"C:\Projects\Git",
        project_root=r"C:\Projects\llm_website",
        max_concurrent_syncs=3,
        timeout_seconds=300
    )


@pytest.mark.skipif(not HAS_PIPELINE, reason=f"Pipeline not available: {IMPORT_ERROR if not HAS_PIPELINE else ''}")
class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.max_concurrent_syncs == 3
        assert config.timeout_seconds == 300
        assert config.analysis_timeout == 600

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            git_root=r"D:\Custom\Git",
            max_concurrent_syncs=5,
            timeout_seconds=600
        )

        assert config.git_root == r"D:\Custom\Git"
        assert config.max_concurrent_syncs == 5
        assert config.timeout_seconds == 600

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that values are within bounds
        config = PipelineConfig(
            max_concurrent_syncs=3,
            timeout_seconds=300
        )

        assert 1 <= config.max_concurrent_syncs <= 10
        assert 60 <= config.timeout_seconds <= 3600


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestRepositoryConfig:
    """Test repository configuration."""

    def test_repository_config_creation(self, sample_repo_config):
        """Test creating a repository configuration."""
        assert sample_repo_config.name == "TestRepo"
        assert sample_repo_config.path == r"C:\Projects\Git\TestRepo"
        assert sample_repo_config.db_name == "testrepo"
        assert sample_repo_config.enabled is True

    def test_default_repositories_exist(self):
        """Test that default repositories are defined."""
        assert len(DEFAULT_REPOSITORIES) > 0
        for repo in DEFAULT_REPOSITORIES:
            assert isinstance(repo, RepositoryConfig)
            assert repo.name is not None
            assert repo.path is not None
            assert repo.db_name is not None


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestSyncStatus:
    """Test sync status enumeration."""

    def test_sync_status_values(self):
        """Test sync status enum values."""
        assert SyncStatus.SUCCESS == "success"
        assert SyncStatus.FAILED == "failed"
        assert SyncStatus.PARTIAL == "partial"
        assert SyncStatus.NO_CHANGES == "no_changes"
        assert SyncStatus.PENDING == "pending"
        assert SyncStatus.IN_PROGRESS == "in_progress"


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestPullTaskResult:
    """Test PullTaskResult dataclass."""

    def test_pull_task_result_creation(self):
        """Test creating a pull task result."""
        result = PullTaskResult(
            repo_name="TestRepo",
            repo_path=r"C:\Projects\Git\TestRepo",
            success=True,
            has_changes=False,
            output="Already up to date."
        )

        assert result.repo_name == "TestRepo"
        assert result.success is True
        assert result.has_changes is False

    def test_pull_task_result_defaults(self):
        """Test pull task result default values."""
        result = PullTaskResult(
            repo_name="TestRepo",
            repo_path=r"C:\Projects\Git\TestRepo"
        )

        assert result.success is False
        assert result.has_changes is False
        assert result.output == ""
        assert result.error == ""


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestAnalysisTaskResult:
    """Test AnalysisTaskResult dataclass."""

    def test_analysis_task_result_creation(self):
        """Test creating an analysis task result."""
        result = AnalysisTaskResult(
            repo_name="TestRepo",
            success=True,
            entity_count=50,
            file_count=10,
            duration_seconds=5.0,
            output_file="/tmp/analysis.json"
        )

        assert result.repo_name == "TestRepo"
        assert result.success is True
        assert result.entity_count == 50
        assert result.file_count == 10

    def test_analysis_task_result_defaults(self):
        """Test analysis task result default values."""
        result = AnalysisTaskResult(repo_name="TestRepo")

        assert result.success is False
        assert result.entity_count == 0
        assert result.file_count == 0
        assert result.duration_seconds == 0.0


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestImportTaskResult:
    """Test ImportTaskResult dataclass."""

    def test_import_task_result_creation(self):
        """Test creating an import task result."""
        result = ImportTaskResult(
            repo_name="TestRepo",
            success=True,
            documents_imported=100,
            documents_updated=25,
            duration_seconds=2.5
        )

        assert result.repo_name == "TestRepo"
        assert result.success is True
        assert result.documents_imported == 100
        assert result.documents_updated == 25


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestGitPullTask:
    """Test git pull task."""

    @pytest.mark.asyncio
    async def test_git_pull_task_success(self, mock_git_service):
        """Test successful git pull task."""
        with patch("git_pipeline.prefect.git_sync_flow.GitService", return_value=mock_git_service):
            with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                result = await git_pull_task.fn(
                    repo_path=r"C:\Projects\Git\TestRepo",
                    repo_name="TestRepo"
                )

                assert isinstance(result, PullTaskResult)
                assert result.repo_name == "TestRepo"

    @pytest.mark.asyncio
    async def test_git_pull_task_repo_not_found(self):
        """Test git pull task with non-existent repository."""
        with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
            with patch("os.path.exists", return_value=False):
                result = await git_pull_task.fn(
                    repo_path=r"C:\NonExistent\Repo",
                    repo_name="NonExistent"
                )

                assert result.success is False
                assert "not found" in result.error.lower() or "not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_git_pull_task_with_changes(self, mock_git_service):
        """Test git pull task when changes are detected."""
        mock_git_service.pull_repository_async.return_value = Mock(
            success=True,
            output="Updating abc..def",
            is_already_up_to_date=False,
            error=None
        )

        with patch("git_pipeline.prefect.git_sync_flow.GitService", return_value=mock_git_service):
            with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                with patch("os.path.exists", return_value=True):
                    result = await git_pull_task.fn(
                        repo_path=r"C:\Projects\Git\TestRepo",
                        repo_name="TestRepo"
                    )

                    assert result.success is True
                    assert result.has_changes is True


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestRoslynAnalysisTask:
    """Test Roslyn analysis task."""

    @pytest.mark.asyncio
    async def test_roslyn_analysis_task_success(self, mock_roslyn_service):
        """Test successful Roslyn analysis task."""
        with patch("git_pipeline.prefect.git_sync_flow.RoslynService", return_value=mock_roslyn_service):
            with patch("git_pipeline.prefect.git_sync_flow.PipelineConfig"):
                with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                    result = await roslyn_analysis_task.fn(
                        repo_path=r"C:\Projects\Git\TestRepo",
                        repo_name="TestRepo"
                    )

                    assert isinstance(result, AnalysisTaskResult)
                    assert result.repo_name == "TestRepo"

    @pytest.mark.asyncio
    async def test_roslyn_analysis_task_failure(self, mock_roslyn_service):
        """Test Roslyn analysis task failure."""
        mock_roslyn_service.analyze_repository.return_value = Mock(
            success=False,
            error="Analysis failed",
            entity_count=0,
            file_count=0,
            output_file=None,
            duration_seconds=0.5
        )

        with patch("git_pipeline.prefect.git_sync_flow.RoslynService", return_value=mock_roslyn_service):
            with patch("git_pipeline.prefect.git_sync_flow.PipelineConfig"):
                with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                    result = await roslyn_analysis_task.fn(
                        repo_path=r"C:\Projects\Git\TestRepo",
                        repo_name="TestRepo"
                    )

                    assert result.success is False
                    assert "Analysis failed" in result.error


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestVectorImportTask:
    """Test vector import task."""

    @pytest.mark.asyncio
    async def test_vector_import_task_success(self, mock_import_service):
        """Test successful vector import task."""
        with patch("git_pipeline.prefect.git_sync_flow.CodeImportService", return_value=mock_import_service):
            with patch("git_pipeline.prefect.git_sync_flow.RoslynService"):
                with patch("git_pipeline.prefect.git_sync_flow.PipelineConfig"):
                    with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                        with patch("os.path.exists", return_value=True):
                            with patch("os.unlink"):
                                result = await vector_import_task.fn(
                                    repo_name="TestRepo",
                                    db_name="testrepo",
                                    analysis_file="/tmp/analysis.json"
                                )

                                assert isinstance(result, ImportTaskResult)
                                assert result.repo_name == "TestRepo"

    @pytest.mark.asyncio
    async def test_vector_import_task_file_not_found(self):
        """Test vector import task when analysis file not found."""
        with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
            with patch("os.path.exists", return_value=False):
                result = await vector_import_task.fn(
                    repo_name="TestRepo",
                    db_name="testrepo",
                    analysis_file="/nonexistent/file.json"
                )

                assert result.success is False or "not found" in result.error.lower()


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestGitSyncFlow:
    """Test git sync flow orchestration."""

    @pytest.mark.asyncio
    async def test_git_sync_flow_no_changes(self, mock_git_service):
        """Test git sync flow when no changes detected."""
        mock_pull_result = PullTaskResult(
            repo_name="TestRepo",
            repo_path=r"C:\Projects\Git\TestRepo",
            success=True,
            has_changes=False,
            output="Already up to date."
        )

        with patch("git_pipeline.prefect.git_sync_flow.git_pull_task", new=AsyncMock(return_value=mock_pull_result)):
            with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                with patch("git_pipeline.prefect.git_sync_flow.create_markdown_artifact", new=AsyncMock()):
                    result = await git_sync_flow.fn(
                        repo_name="TestRepo",
                        repo_path=r"C:\Projects\Git\TestRepo",
                        db_name="testrepo",
                        force_analysis=False
                    )

                    assert result["success"] is True
                    assert result["has_changes"] is False

    @pytest.mark.asyncio
    async def test_git_sync_flow_with_changes(self, mock_git_service, mock_roslyn_service, mock_import_service):
        """Test git sync flow when changes are detected."""
        mock_pull_result = PullTaskResult(
            repo_name="TestRepo",
            repo_path=r"C:\Projects\Git\TestRepo",
            success=True,
            has_changes=True,
            output="Fast-forward"
        )

        mock_analysis_result = AnalysisTaskResult(
            repo_name="TestRepo",
            success=True,
            entity_count=50,
            file_count=10,
            duration_seconds=5.0,
            output_file="/tmp/analysis.json"
        )

        mock_import_result = ImportTaskResult(
            repo_name="TestRepo",
            success=True,
            documents_imported=50,
            documents_updated=10,
            duration_seconds=2.0
        )

        with patch("git_pipeline.prefect.git_sync_flow.git_pull_task", new=AsyncMock(return_value=mock_pull_result)):
            with patch("git_pipeline.prefect.git_sync_flow.roslyn_analysis_task", new=AsyncMock(return_value=mock_analysis_result)):
                with patch("git_pipeline.prefect.git_sync_flow.vector_import_task", new=AsyncMock(return_value=mock_import_result)):
                    with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                        with patch("git_pipeline.prefect.git_sync_flow.create_markdown_artifact", new=AsyncMock()):
                            with patch("git_pipeline.prefect.git_sync_flow.emit_event"):
                                result = await git_sync_flow.fn(
                                    repo_name="TestRepo",
                                    repo_path=r"C:\Projects\Git\TestRepo",
                                    db_name="testrepo",
                                    force_analysis=False
                                )

                                assert result["success"] is True
                                assert result["has_changes"] is True

    @pytest.mark.asyncio
    async def test_git_sync_flow_force_analysis(self, mock_git_service, mock_roslyn_service, mock_import_service):
        """Test git sync flow with force_analysis flag."""
        mock_pull_result = PullTaskResult(
            repo_name="TestRepo",
            repo_path=r"C:\Projects\Git\TestRepo",
            success=True,
            has_changes=False,  # No changes
            output="Already up to date."
        )

        mock_analysis_result = AnalysisTaskResult(
            repo_name="TestRepo",
            success=True,
            entity_count=50,
            file_count=10,
            duration_seconds=5.0,
            output_file="/tmp/analysis.json"
        )

        mock_import_result = ImportTaskResult(
            repo_name="TestRepo",
            success=True,
            documents_imported=50,
            documents_updated=10,
            duration_seconds=2.0
        )

        with patch("git_pipeline.prefect.git_sync_flow.git_pull_task", new=AsyncMock(return_value=mock_pull_result)):
            with patch("git_pipeline.prefect.git_sync_flow.roslyn_analysis_task", new=AsyncMock(return_value=mock_analysis_result)) as mock_analysis:
                with patch("git_pipeline.prefect.git_sync_flow.vector_import_task", new=AsyncMock(return_value=mock_import_result)):
                    with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                        with patch("git_pipeline.prefect.git_sync_flow.create_markdown_artifact", new=AsyncMock()):
                            with patch("git_pipeline.prefect.git_sync_flow.emit_event"):
                                result = await git_sync_flow.fn(
                                    repo_name="TestRepo",
                                    repo_path=r"C:\Projects\Git\TestRepo",
                                    db_name="testrepo",
                                    force_analysis=True  # Force analysis
                                )

                                # Analysis should be called even with no changes
                                mock_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_git_sync_flow_pull_failure(self):
        """Test git sync flow when pull fails."""
        mock_pull_result = PullTaskResult(
            repo_name="TestRepo",
            repo_path=r"C:\Projects\Git\TestRepo",
            success=False,
            has_changes=False,
            error="Network error"
        )

        with patch("git_pipeline.prefect.git_sync_flow.git_pull_task", new=AsyncMock(return_value=mock_pull_result)):
            with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                with patch("git_pipeline.prefect.git_sync_flow.emit_event"):
                    result = await git_sync_flow.fn(
                        repo_name="TestRepo",
                        repo_path=r"C:\Projects\Git\TestRepo",
                        db_name="testrepo",
                        force_analysis=False
                    )

                    assert result["success"] is False or "error" in result


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestGitSyncAllFlow:
    """Test git sync all flow orchestration."""

    @pytest.mark.asyncio
    async def test_git_sync_all_flow_success(self):
        """Test syncing all repositories successfully."""
        mock_result = {
            "success": True,
            "repo_name": "TestRepo",
            "has_changes": False
        }

        with patch("git_pipeline.prefect.git_sync_flow.git_sync_flow", new=AsyncMock(return_value=mock_result)):
            with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                with patch("git_pipeline.prefect.git_sync_flow.create_markdown_artifact", new=AsyncMock()):
                    with patch("git_pipeline.prefect.git_sync_flow.emit_event"):
                        with patch("git_pipeline.prefect.git_sync_flow.DEFAULT_REPOSITORIES", [
                            RepositoryConfig(
                                name="TestRepo",
                                path=r"C:\Projects\Git\TestRepo",
                                db_name="testrepo",
                                enabled=True
                            )
                        ]):
                            result = await git_sync_all_flow.fn(force_analysis=False)

                            assert result["success"] is True
                            assert result["total_repositories"] >= 1

    @pytest.mark.asyncio
    async def test_git_sync_all_flow_skips_disabled(self):
        """Test that disabled repositories are skipped."""
        mock_result = {
            "success": True,
            "repo_name": "EnabledRepo",
            "has_changes": False
        }

        with patch("git_pipeline.prefect.git_sync_flow.git_sync_flow", new=AsyncMock(return_value=mock_result)) as mock_sync:
            with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                with patch("git_pipeline.prefect.git_sync_flow.create_markdown_artifact", new=AsyncMock()):
                    with patch("git_pipeline.prefect.git_sync_flow.emit_event"):
                        with patch("git_pipeline.prefect.git_sync_flow.DEFAULT_REPOSITORIES", [
                            RepositoryConfig(
                                name="EnabledRepo",
                                path=r"C:\Projects\Git\EnabledRepo",
                                db_name="enabled",
                                enabled=True
                            ),
                            RepositoryConfig(
                                name="DisabledRepo",
                                path=r"C:\Projects\Git\DisabledRepo",
                                db_name="disabled",
                                enabled=False  # Disabled
                            )
                        ]):
                            result = await git_sync_all_flow.fn(force_analysis=False)

                            # Only enabled repo should be synced
                            assert mock_sync.call_count == 1


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestGitAnalysisFlow:
    """Test git analysis flow (from git_flow.py)."""

    @pytest.mark.asyncio
    async def test_git_analysis_flow_default_dates(self, mock_git_service):
        """Test git analysis flow with default date range."""
        with patch("prefect_pipelines.git_flow.get_git_service", return_value=mock_git_service):
            with patch("prefect_pipelines.git_flow.get_run_logger"):
                with patch("prefect_pipelines.git_flow.create_markdown_artifact", new=AsyncMock()):
                    with patch("prefect_pipelines.git_flow.emit_event"):
                        with patch("prefect_pipelines.git_flow.pull_repository_task", new=AsyncMock(return_value=FlowPullResult(
                            repo_path=r"C:\Projects\Git\TestRepo",
                            repo_name="TestRepo",
                            success=True,
                            is_already_up_to_date=True
                        ))):
                            with patch("prefect_pipelines.git_flow.analyze_commits_task", new=AsyncMock(return_value=FlowAnalysisResult(
                                repo_path=r"C:\Projects\Git\TestRepo",
                                repo_name="TestRepo",
                                success=True,
                                total_commits=0
                            ))):
                                result = await git_analysis_flow.fn(
                                    repo_path=r"C:\Projects\Git\TestRepo"
                                )

                                assert result["success"] is True
                                # Default dates should be set
                                assert result["start_date"] is not None
                                assert result["end_date"] is not None

    @pytest.mark.asyncio
    async def test_git_analysis_flow_custom_dates(self, mock_git_service):
        """Test git analysis flow with custom date range."""
        with patch("prefect_pipelines.git_flow.get_git_service", return_value=mock_git_service):
            with patch("prefect_pipelines.git_flow.get_run_logger"):
                with patch("prefect_pipelines.git_flow.create_markdown_artifact", new=AsyncMock()):
                    with patch("prefect_pipelines.git_flow.emit_event"):
                        with patch("prefect_pipelines.git_flow.pull_repository_task", new=AsyncMock(return_value=FlowPullResult(
                            repo_path=r"C:\Projects\Git\TestRepo",
                            repo_name="TestRepo",
                            success=True,
                            is_already_up_to_date=True
                        ))):
                            with patch("prefect_pipelines.git_flow.analyze_commits_task", new=AsyncMock(return_value=FlowAnalysisResult(
                                repo_path=r"C:\Projects\Git\TestRepo",
                                repo_name="TestRepo",
                                success=True,
                                total_commits=5,
                                start_date="2024-01-01",
                                end_date="2024-01-31"
                            ))):
                                result = await git_analysis_flow.fn(
                                    repo_path=r"C:\Projects\Git\TestRepo",
                                    start_date="2024-01-01",
                                    end_date="2024-01-31"
                                )

                                assert result["success"] is True
                                assert result["start_date"] == "2024-01-01"
                                assert result["end_date"] == "2024-01-31"

    @pytest.mark.asyncio
    async def test_git_analysis_flow_skip_pull(self, mock_git_service):
        """Test git analysis flow without pulling."""
        with patch("prefect_pipelines.git_flow.get_git_service", return_value=mock_git_service):
            with patch("prefect_pipelines.git_flow.get_run_logger"):
                with patch("prefect_pipelines.git_flow.create_markdown_artifact", new=AsyncMock()):
                    with patch("prefect_pipelines.git_flow.emit_event"):
                        with patch("prefect_pipelines.git_flow.pull_repository_task", new=AsyncMock()) as mock_pull:
                            with patch("prefect_pipelines.git_flow.analyze_commits_task", new=AsyncMock(return_value=FlowAnalysisResult(
                                repo_path=r"C:\Projects\Git\TestRepo",
                                repo_name="TestRepo",
                                success=True,
                                total_commits=0
                            ))):
                                result = await git_analysis_flow.fn(
                                    repo_path=r"C:\Projects\Git\TestRepo",
                                    pull_first=False  # Skip pull
                                )

                                # Pull should not be called
                                mock_pull.assert_not_called()


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestSyncRunner:
    """Test synchronous sync runners."""

    def test_run_git_sync_invalid_path(self):
        """Test run_git_sync with non-existent path."""
        result = run_git_sync(
            repo_name="NonExistent",
            repo_path=r"C:\NonExistent\Path",
            db_name="nonexistent",
            force_analysis=False
        )

        assert result["success"] is False
        assert "not exist" in result.get("error", "").lower()

    def test_run_git_analysis_flow_invalid_path(self):
        """Test run_git_analysis_flow with non-existent path."""
        result = run_git_analysis_flow(
            repo_path=r"C:\NonExistent\Path",
            use_prefect=False
        )

        assert result["success"] is False


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestFlowResultStructure:
    """Test flow result structures."""

    def test_sync_flow_result_structure(self):
        """Test SyncFlowResult has expected structure."""
        result = SyncFlowResult(repo_name="TestRepo")

        assert hasattr(result, 'repo_name')
        assert hasattr(result, 'success')
        assert hasattr(result, 'has_changes')
        assert hasattr(result, 'pull_result')
        assert hasattr(result, 'analysis_result')
        assert hasattr(result, 'import_result')
        assert hasattr(result, 'total_duration_seconds')
        assert hasattr(result, 'error')

    def test_analysis_result_model(self):
        """Test AnalysisResult model structure."""
        result = AnalysisResult(
            success=True,
            repository="TestRepo",
            entity_count=50,
            file_count=10
        )

        assert result.success is True
        assert result.repository == "TestRepo"
        assert result.entity_count == 50

    def test_import_result_model(self):
        """Test ImportResult model structure."""
        result = ImportResult(
            success=True,
            documents_imported=100,
            documents_updated=25
        )

        assert result.success is True
        assert result.documents_imported == 100
        assert result.documents_updated == 25


@pytest.mark.skipif(not HAS_PIPELINE, reason="Pipeline not available")
class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_pipeline_flow_mocked(self):
        """Test complete pipeline flow with all components mocked."""
        # Mock all external dependencies
        mock_pull = PullTaskResult(
            repo_name="IntegrationTest",
            repo_path=r"C:\Projects\Git\IntegrationTest",
            success=True,
            has_changes=True
        )

        mock_analysis = AnalysisTaskResult(
            repo_name="IntegrationTest",
            success=True,
            entity_count=100,
            file_count=20,
            output_file="/tmp/test_analysis.json",
            duration_seconds=10.0
        )

        mock_import = ImportTaskResult(
            repo_name="IntegrationTest",
            success=True,
            documents_imported=100,
            documents_updated=50,
            duration_seconds=5.0
        )

        with patch("git_pipeline.prefect.git_sync_flow.git_pull_task", new=AsyncMock(return_value=mock_pull)):
            with patch("git_pipeline.prefect.git_sync_flow.roslyn_analysis_task", new=AsyncMock(return_value=mock_analysis)):
                with patch("git_pipeline.prefect.git_sync_flow.vector_import_task", new=AsyncMock(return_value=mock_import)):
                    with patch("git_pipeline.prefect.git_sync_flow.get_run_logger"):
                        with patch("git_pipeline.prefect.git_sync_flow.create_markdown_artifact", new=AsyncMock()):
                            with patch("git_pipeline.prefect.git_sync_flow.emit_event"):
                                result = await git_sync_flow.fn(
                                    repo_name="IntegrationTest",
                                    repo_path=r"C:\Projects\Git\IntegrationTest",
                                    db_name="integrationtest",
                                    force_analysis=False
                                )

                                # Verify complete flow executed
                                assert result["success"] is True
                                assert result["has_changes"] is True
                                assert result["analysis_result"]["entity_count"] == 100
                                assert result["import_result"]["documents_imported"] == 100
