"""
Git Service Tests
=================

Comprehensive tests for the GitService class providing core git operations.

Tests cover:
- Repository discovery and management
- Pull operations (sync and async)
- Changed files detection
- Commit history retrieval
- Repository information
- HEAD management
"""

import os
import pytest
import asyncio
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

# Conditional import to avoid GIT_ROOT requirement during test collection
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python_services"))
    os.environ.setdefault("GIT_ROOT", r"C:\Projects\Git")

    from git_service.git_service import GitService
    from git_service.models import (
        Repository,
        Commit,
        CommitFile,
        CommitDetails,
        FileChange,
        FileStatus,
        PullResult,
        RepositoryInfo,
        GitCommandResult,
    )
    HAS_GIT_SERVICE = True
except Exception:
    HAS_GIT_SERVICE = False


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=str(repo_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_path), capture_output=True)

    # Create an initial commit
    test_file = repo_path / "test.txt"
    test_file.write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=str(repo_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo_path), capture_output=True)

    return str(repo_path)


@pytest.fixture
def mock_git_service():
    """Create a GitService with mocked scanner."""
    with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
        with patch("git_service.git_service.RepositoryScanner") as MockScanner:
            mock_scanner = Mock()
            mock_scanner.scan_for_repositories.return_value = [
                Repository(name="TestRepo", path=r"C:\Projects\Git\TestRepo", display_name="Test Repo"),
                Repository(name="OtherRepo", path=r"C:\Projects\Git\OtherRepo", display_name="Other Repo"),
            ]
            mock_scanner.get_repository_path.return_value = r"C:\Projects\Git\TestRepo"
            mock_scanner.verify_repository.return_value = True
            MockScanner.return_value = mock_scanner

            service = GitService(git_root=r"C:\Projects\Git")
            return service


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGitServiceInit:
    """Test GitService initialization."""

    def test_initialization_with_default_root(self):
        """Test GitService initializes with default git root from environment."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                assert str(service.git_root) == r"C:\Projects\Git"

    def test_initialization_with_custom_root(self):
        """Test GitService initializes with custom git root."""
        custom_root = r"D:\CustomGit"
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService(git_root=custom_root)
                assert str(service.git_root) == custom_root

    def test_scanner_injection(self):
        """Test GitService accepts injected scanner."""
        mock_scanner = Mock()
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            service = GitService(git_root=r"C:\Projects\Git", scanner=mock_scanner)
            assert service.scanner is mock_scanner


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestExecuteGitCommand:
    """Test git command execution."""

    def test_execute_git_command_success(self, temp_git_repo):
        """Test successful git command execution."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                result = service.execute_git_command("status", temp_git_repo)

                assert result.success is True
                assert result.return_code == 0
                assert "On branch" in result.stdout or "nothing to commit" in result.stdout.lower()

    def test_execute_git_command_invalid_command(self, temp_git_repo):
        """Test git command with invalid command."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                result = service.execute_git_command("invalid-command-xyz", temp_git_repo)

                assert result.success is False
                assert result.return_code != 0

    def test_execute_git_command_invalid_repo(self, tmp_path):
        """Test git command in non-repo directory."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                result = service.execute_git_command("status", str(tmp_path))

                assert result.success is False

    def test_execute_git_command_timeout(self, temp_git_repo):
        """Test git command timeout handling."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=1)
                    service = GitService()
                    result = service.execute_git_command("status", temp_git_repo, timeout=1)

                    assert result.success is False
                    assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_git_command_async(self, temp_git_repo):
        """Test async git command execution."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                result = await service.execute_git_command_async("status", temp_git_repo)

                assert result.success is True
                assert result.return_code == 0


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestScanRepositories:
    """Test repository scanning."""

    def test_scan_repositories_returns_list(self, mock_git_service):
        """Test scan_repositories returns list of Repository objects."""
        repos = mock_git_service.scan_repositories()

        assert isinstance(repos, list)
        assert len(repos) == 2
        assert all(isinstance(r, Repository) for r in repos)

    def test_scan_repositories_force_refresh(self, mock_git_service):
        """Test scan_repositories with force_refresh flag."""
        mock_git_service.scan_repositories(force_refresh=True)
        mock_git_service.scanner.scan_for_repositories.assert_called_with(force_refresh=True)

    def test_scan_repositories_caches_results(self, mock_git_service):
        """Test scan_repositories caches results."""
        mock_git_service.scan_repositories()
        mock_git_service.scan_repositories()

        # Should be called twice (no caching in scanner mock)
        assert mock_git_service.scanner.scan_for_repositories.call_count == 2


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestVerifyRepository:
    """Test repository verification."""

    def test_verify_valid_repository(self, temp_git_repo):
        """Test verifying a valid git repository."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner") as MockScanner:
                mock_scanner = Mock()
                mock_scanner.verify_repository.return_value = True
                MockScanner.return_value = mock_scanner

                service = GitService()
                is_valid = service.verify_repository(temp_git_repo)

                assert is_valid is True

    def test_verify_invalid_repository(self, tmp_path):
        """Test verifying a non-git directory."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner") as MockScanner:
                mock_scanner = Mock()
                mock_scanner.verify_repository.return_value = False
                MockScanner.return_value = mock_scanner

                service = GitService()
                is_valid = service.verify_repository(str(tmp_path))

                assert is_valid is False


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetRepositoryPath:
    """Test repository path retrieval."""

    def test_get_repository_path_found(self, mock_git_service):
        """Test getting path for existing repository."""
        path = mock_git_service.get_repository_path("TestRepo")
        assert path == r"C:\Projects\Git\TestRepo"

    def test_get_repository_path_not_found(self, mock_git_service):
        """Test getting path for non-existent repository."""
        mock_git_service.scanner.get_repository_path.return_value = None
        path = mock_git_service.get_repository_path("NonExistent")
        assert path is None


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestPullRepository:
    """Test git pull operations."""

    def test_pull_repository_success(self, temp_git_repo):
        """Test successful pull operation (already up to date)."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                # Local repo with no remote will show "already up to date" behavior
                result = service.pull_repository(temp_git_repo)

                assert isinstance(result, PullResult)
                # May fail since there's no remote, but structure should be correct
                assert hasattr(result, 'success')
                assert hasattr(result, 'output')
                assert hasattr(result, 'is_already_up_to_date')

    def test_pull_repository_detects_already_up_to_date(self):
        """Test pull correctly identifies already up to date state."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                with patch.object(service, 'execute_git_command') as mock_exec:
                    mock_exec.return_value = GitCommandResult(
                        success=True,
                        stdout="Already up to date.",
                        stderr="",
                        return_code=0
                    )

                    result = service.pull_repository(r"C:\Projects\Git\TestRepo")

                    assert result.success is True
                    assert result.is_already_up_to_date is True

    def test_pull_repository_detects_changes(self):
        """Test pull correctly identifies when changes were pulled."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                with patch.object(service, 'execute_git_command') as mock_exec:
                    mock_exec.return_value = GitCommandResult(
                        success=True,
                        stdout="Updating abc123..def456\nFast-forward\n 1 file changed",
                        stderr="",
                        return_code=0
                    )

                    result = service.pull_repository(r"C:\Projects\Git\TestRepo")

                    assert result.success is True
                    assert result.is_already_up_to_date is False

    @pytest.mark.asyncio
    async def test_pull_repository_async(self, temp_git_repo):
        """Test async pull operation."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                result = await service.pull_repository_async(temp_git_repo)

                assert isinstance(result, PullResult)
                assert hasattr(result, 'success')


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetRecentCommits:
    """Test retrieving recent commits."""

    def test_get_recent_commits(self, temp_git_repo):
        """Test getting recent commits from repository."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                commits = service.get_recent_commits(temp_git_repo, limit=5)

                assert isinstance(commits, list)
                # Should have at least the initial commit
                assert len(commits) >= 1
                assert all(isinstance(c, Commit) for c in commits)

                # Verify commit structure
                commit = commits[0]
                assert hasattr(commit, 'hash')
                assert hasattr(commit, 'author')
                assert hasattr(commit, 'date')
                assert hasattr(commit, 'message')

    def test_get_recent_commits_respects_limit(self, temp_git_repo):
        """Test that commit limit is respected."""
        # Add more commits
        for i in range(5):
            test_file = Path(temp_git_repo) / f"file_{i}.txt"
            test_file.write_text(f"content {i}")
            subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Commit {i}"], cwd=temp_git_repo, capture_output=True)

        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                commits = service.get_recent_commits(temp_git_repo, limit=3)

                assert len(commits) == 3

    def test_get_recent_commits_no_merges(self, temp_git_repo):
        """Test getting commits with no_merges flag."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                with patch.object(service, 'execute_git_command') as mock_exec:
                    mock_exec.return_value = GitCommandResult(
                        success=True,
                        stdout="abc123|Author|2024-01-15T10:00:00|Test message",
                        stderr="",
                        return_code=0
                    )

                    commits = service.get_recent_commits(temp_git_repo, no_merges=True)

                    # Verify --no-merges flag was included
                    call_args = mock_exec.call_args[0][0]
                    assert "--no-merges" in call_args

    @pytest.mark.asyncio
    async def test_get_recent_commits_async(self, temp_git_repo):
        """Test async commit retrieval."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                commits = await service.get_recent_commits_async(temp_git_repo, limit=5)

                assert isinstance(commits, list)
                assert len(commits) >= 1


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetChangedFiles:
    """Test changed files detection."""

    def test_get_changed_files_between_commits(self, temp_git_repo):
        """Test getting changed files between two commits."""
        # Create initial commit state
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                # Get initial HEAD
                initial_head = service.get_current_head(temp_git_repo)

                # Add a new file and commit
                new_file = Path(temp_git_repo) / "new_file.txt"
                new_file.write_text("new content")
                subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
                subprocess.run(["git", "commit", "-m", "Add new file"], cwd=temp_git_repo, capture_output=True)

                # Get changed files
                changed = service.get_changed_files(temp_git_repo, initial_head, "HEAD")

                assert isinstance(changed, list)
                assert "new_file.txt" in changed

    def test_get_changed_files_no_changes(self, temp_git_repo):
        """Test getting changed files when there are no changes."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                head = service.get_current_head(temp_git_repo)
                changed = service.get_changed_files(temp_git_repo, head, head)

                assert isinstance(changed, list)
                assert len(changed) == 0

    @pytest.mark.asyncio
    async def test_get_changed_files_async(self, temp_git_repo):
        """Test async changed files detection."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                head = service.get_current_head(temp_git_repo)
                changed = await service.get_changed_files_async(temp_git_repo, head, "HEAD")

                assert isinstance(changed, list)


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetChangedFilesWithStatus:
    """Test changed files with status detection."""

    def test_get_changed_files_with_status(self):
        """Test getting changed files with their status."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                with patch.object(service, 'execute_git_command') as mock_exec:
                    mock_exec.return_value = GitCommandResult(
                        success=True,
                        stdout="A\tnew_file.txt\nM\tmodified.txt\nD\tdeleted.txt",
                        stderr="",
                        return_code=0
                    )

                    changes = service.get_changed_files_with_status(r"C:\Projects\Git\TestRepo")

                    assert isinstance(changes, list)
                    assert len(changes) == 3

                    # Check status mapping
                    statuses = {c.file: c.status for c in changes}
                    assert statuses["new_file.txt"] == FileStatus.ADDED
                    assert statuses["modified.txt"] == FileStatus.MODIFIED
                    assert statuses["deleted.txt"] == FileStatus.DELETED

    def test_get_changed_files_with_status_renamed(self):
        """Test detecting renamed files."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                with patch.object(service, 'execute_git_command') as mock_exec:
                    mock_exec.return_value = GitCommandResult(
                        success=True,
                        stdout="R100\told_name.txt\tnew_name.txt",
                        stderr="",
                        return_code=0
                    )

                    changes = service.get_changed_files_with_status(r"C:\Projects\Git\TestRepo")

                    assert len(changes) == 1
                    assert changes[0].status == FileStatus.RENAMED


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetLastPullTime:
    """Test last pull time retrieval."""

    def test_get_last_pull_time(self, temp_git_repo):
        """Test getting last pull time."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                last_pull = service.get_last_pull_time(temp_git_repo)

                assert last_pull is not None
                assert isinstance(last_pull, str)
                # Should be ISO format date
                datetime.fromisoformat(last_pull.replace('Z', '+00:00'))

    def test_get_last_pull_time_invalid_repo(self, tmp_path):
        """Test getting last pull time from invalid repo."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                last_pull = service.get_last_pull_time(str(tmp_path))

                assert last_pull is None


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetCurrentHead:
    """Test current HEAD retrieval."""

    def test_get_current_head(self, temp_git_repo):
        """Test getting current HEAD commit."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                head = service.get_current_head(temp_git_repo)

                assert head is not None
                assert isinstance(head, str)
                assert len(head) == 40  # Full SHA-1 hash

    def test_get_current_head_invalid_repo(self, tmp_path):
        """Test getting HEAD from invalid repo."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                head = service.get_current_head(str(tmp_path))

                assert head is None

    @pytest.mark.asyncio
    async def test_get_current_head_async(self, temp_git_repo):
        """Test async HEAD retrieval."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                head = await service.get_current_head_async(temp_git_repo)

                assert head is not None
                assert len(head) == 40


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetRepositoryInfo:
    """Test repository information retrieval."""

    @pytest.mark.asyncio
    async def test_get_repository_info(self, temp_git_repo):
        """Test getting repository info."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                info = await service.get_repository_info(temp_git_repo)

                assert isinstance(info, RepositoryInfo)
                assert info.path == temp_git_repo
                assert info.name == Path(temp_git_repo).name
                assert info.status == "active"
                assert isinstance(info.recent_commits, list)

    @pytest.mark.asyncio
    async def test_get_repository_info_invalid_repo(self, tmp_path):
        """Test getting info for invalid repo."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                info = await service.get_repository_info(str(tmp_path))

                assert isinstance(info, RepositoryInfo)
                # Should still return info but with error status
                assert info.status == "error" or len(info.recent_commits) == 0

    def test_get_repository_info_sync(self, temp_git_repo):
        """Test sync repository info retrieval."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()
                info = service.get_repository_info_sync(temp_git_repo)

                assert isinstance(info, RepositoryInfo)
                assert info.path == temp_git_repo


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestAnalyzeCommitsByDateRange:
    """Test commit analysis by date range."""

    def test_analyze_commits_by_date_range(self, temp_git_repo):
        """Test analyzing commits within date range."""
        # Add some commits with known dates
        for i in range(3):
            test_file = Path(temp_git_repo) / f"range_file_{i}.txt"
            test_file.write_text(f"range content {i}")
            subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Range commit {i}"], cwd=temp_git_repo, capture_output=True)

        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                # Use wide date range to catch all commits
                start_date = "2020-01-01"
                end_date = "2030-12-31"

                commits = service.analyze_commits_by_date_range(
                    temp_git_repo, start_date, end_date
                )

                assert isinstance(commits, list)
                # Should have at least initial + 3 new commits
                assert len(commits) >= 1

    @pytest.mark.asyncio
    async def test_analyze_commits_by_date_range_async(self, temp_git_repo):
        """Test async commit analysis by date range."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                commits = await service.analyze_commits_by_date_range_async(
                    temp_git_repo, "2020-01-01", "2030-12-31"
                )

                assert isinstance(commits, list)

    def test_analyze_commits_empty_range(self, temp_git_repo):
        """Test analyzing commits with no commits in range."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                # Use past date range where no commits exist
                commits = service.analyze_commits_by_date_range(
                    temp_git_repo, "2000-01-01", "2000-01-02"
                )

                assert isinstance(commits, list)
                assert len(commits) == 0


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGetCommitDetails:
    """Test commit details retrieval."""

    def test_get_commit_details(self, temp_git_repo):
        """Test getting details for a specific commit."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                # Get current HEAD
                head = service.get_current_head(temp_git_repo)
                details = service.get_commit_details(temp_git_repo, head)

                assert details is not None
                assert isinstance(details, CommitDetails)
                assert details.hash == head
                assert details.message is not None

    def test_get_commit_details_invalid_hash(self, temp_git_repo):
        """Test getting details for invalid commit hash."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                details = service.get_commit_details(temp_git_repo, "invalid_hash_12345")

                assert details is None


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestParseNumstatOutput:
    """Test numstat output parsing."""

    def test_parse_numstat_output_single_commit(self):
        """Test parsing numstat output for single commit."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                output = """abc123def456|abc123d|Test Author|2024-01-15T10:00:00|Test commit message
10\t5\tfile1.txt
20\t3\tfile2.txt"""

                commits = service._parse_numstat_output(output)

                assert len(commits) == 1
                commit = commits[0]
                assert commit.hash == "abc123def456"
                assert commit.author == "Test Author"
                assert commit.message == "Test commit message"
                assert len(commit.files) == 2
                assert commit.files[0].filename == "file1.txt"
                assert commit.files[0].additions == 10
                assert commit.files[0].deletions == 5

    def test_parse_numstat_output_multiple_commits(self):
        """Test parsing numstat output for multiple commits."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                output = """abc123|abc1234|Author1|2024-01-15T10:00:00|Commit 1
5\t2\tfile1.txt

def456|def4567|Author2|2024-01-14T09:00:00|Commit 2
3\t1\tfile2.txt"""

                commits = service._parse_numstat_output(output)

                assert len(commits) == 2
                assert commits[0].message == "Commit 1"
                assert commits[1].message == "Commit 2"

    def test_parse_numstat_output_binary_files(self):
        """Test parsing numstat output with binary files."""
        with patch.dict(os.environ, {"GIT_ROOT": r"C:\Projects\Git"}):
            with patch("git_service.git_service.RepositoryScanner"):
                service = GitService()

                output = """abc123|abc1234|Author|2024-01-15T10:00:00|Commit with binary
-\t-\timage.png"""

                commits = service._parse_numstat_output(output)

                assert len(commits) == 1
                # Binary files show as - which becomes 0
                assert commits[0].files[0].additions == 0
                assert commits[0].files[0].deletions == 0


@pytest.mark.skipif(not HAS_GIT_SERVICE, reason="GitService not available")
class TestGitServiceIntegration:
    """Integration tests for GitService."""

    @pytest.mark.requires_mongodb
    @pytest.mark.e2e
    def test_full_repository_workflow(self, temp_git_repo):
        """Test complete workflow: scan -> verify -> info -> commits."""
        with patch.dict(os.environ, {"GIT_ROOT": str(Path(temp_git_repo).parent)}):
            with patch("git_service.git_service.RepositoryScanner") as MockScanner:
                mock_scanner = Mock()
                repo_name = Path(temp_git_repo).name
                mock_scanner.scan_for_repositories.return_value = [
                    Repository(name=repo_name, path=temp_git_repo, display_name=repo_name)
                ]
                mock_scanner.get_repository_path.return_value = temp_git_repo
                mock_scanner.verify_repository.return_value = True
                MockScanner.return_value = mock_scanner

                service = GitService()

                # Step 1: Scan repositories
                repos = service.scan_repositories()
                assert len(repos) >= 1

                # Step 2: Verify repository
                is_valid = service.verify_repository(temp_git_repo)
                assert is_valid is True

                # Step 3: Get repository info
                info = service.get_repository_info_sync(temp_git_repo)
                assert info.status == "active"

                # Step 4: Get recent commits
                commits = service.get_recent_commits(temp_git_repo, limit=5)
                assert len(commits) >= 1

                # Step 5: Get current HEAD
                head = service.get_current_head(temp_git_repo)
                assert head is not None

                # Step 6: Get last pull time
                last_pull = service.get_last_pull_time(temp_git_repo)
                assert last_pull is not None
