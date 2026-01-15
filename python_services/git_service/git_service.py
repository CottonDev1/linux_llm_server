"""
Git Service

Core Git operations service providing repository management, commit analysis,
and file change tracking. Migrated from JavaScript GitService.js.
"""

import asyncio
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from .models import (
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
from .repository_scanner import RepositoryScanner

logger = logging.getLogger(__name__)


class GitService:
    """
    Git Service providing core git operations.

    This class handles all Git repository operations including:
    - Repository discovery and management
    - Commit history retrieval
    - Git diff analysis
    - Pull operations
    - File change tracking

    Migrated from JavaScript GitService.js
    """

    def __init__(
        self,
        git_root: str = r"C:\Projects\Git",
        scanner: Optional[RepositoryScanner] = None
    ):
        """
        Initialize the Git Service.

        Args:
            git_root: Root directory containing git repositories
            scanner: Optional RepositoryScanner instance
        """
        self.git_root = Path(git_root)
        self.scanner = scanner or RepositoryScanner(git_root=git_root)
        logger.info(f"GitService initialized with root: {git_root}")

    # =========================================================================
    # Git Command Execution
    # =========================================================================

    def execute_git_command(
        self,
        command: str,
        repo_path: str,
        timeout: int = 60
    ) -> GitCommandResult:
        """
        Execute a git command synchronously in the specified repository.

        Args:
            command: Git command to execute (without 'git' prefix)
            repo_path: Path to the repository
            timeout: Command timeout in seconds

        Returns:
            GitCommandResult with stdout, stderr, and status
        """
        full_command = f"git {command}"
        logger.debug(f"Executing: {full_command} in {repo_path}")

        try:
            result = subprocess.run(
                full_command,
                cwd=repo_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )

            return GitCommandResult(
                success=result.returncode == 0,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                return_code=result.returncode,
                error=result.stderr.strip() if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {full_command}")
            return GitCommandResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Git command failed: {e}")
            return GitCommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                error=str(e)
            )

    async def execute_git_command_async(
        self,
        command: str,
        repo_path: str,
        timeout: int = 60
    ) -> GitCommandResult:
        """
        Execute a git command asynchronously using thread pool.

        Note: Uses asyncio.to_thread with subprocess.run instead of
        asyncio.create_subprocess_shell for Windows compatibility.

        Args:
            command: Git command to execute (without 'git' prefix)
            repo_path: Path to the repository
            timeout: Command timeout in seconds

        Returns:
            GitCommandResult with stdout, stderr, and status
        """
        # Use sync subprocess in thread pool for Windows compatibility
        return await asyncio.to_thread(
            self.execute_git_command, command, repo_path, timeout
        )

    # =========================================================================
    # Repository Discovery
    # =========================================================================

    def scan_repositories(self, force_refresh: bool = False) -> List[Repository]:
        """
        Scan for git repositories in the configured root directory.

        Args:
            force_refresh: If True, bypass cache and rescan

        Returns:
            List of Repository objects
        """
        return self.scanner.scan_for_repositories(force_refresh=force_refresh)

    def get_repository_path(self, repo_name: str) -> Optional[str]:
        """
        Get the full path to a repository by name.

        Args:
            repo_name: Name of the repository

        Returns:
            Full path to repository or None if not found
        """
        return self.scanner.get_repository_path(repo_name)

    def verify_repository(self, repo_path: str) -> bool:
        """
        Verify that a path is a valid Git repository.

        Args:
            repo_path: Path to verify

        Returns:
            True if valid Git repository
        """
        # Quick check for .git folder
        if not self.scanner.verify_repository(repo_path):
            return False

        # Also verify git status works
        result = self.execute_git_command("status", repo_path, timeout=10)
        return result.success

    # =========================================================================
    # Commit History
    # =========================================================================

    def get_recent_commits(
        self,
        repo_path: str,
        limit: int = 5,
        no_merges: bool = True
    ) -> List[Commit]:
        """
        Get recent commits from a git repository.

        Args:
            repo_path: Full path to the git repository
            limit: Maximum number of commits to retrieve
            no_merges: If True, exclude merge commits

        Returns:
            List of Commit objects
        """
        merges_flag = "--no-merges" if no_merges else ""
        command = f'log {merges_flag} --pretty=format:"%H|%an|%aI|%s" -{limit}'

        result = self.execute_git_command(command, repo_path)
        if not result.success or not result.stdout:
            logger.warning(f"Could not get commits from {repo_path}: {result.error}")
            return []

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 3)
            if len(parts) >= 4:
                commits.append(Commit(
                    hash=parts[0],
                    author=parts[1],
                    date=parts[2],
                    message=parts[3]
                ))

        return commits

    async def get_recent_commits_async(
        self,
        repo_path: str,
        limit: int = 5,
        no_merges: bool = True
    ) -> List[Commit]:
        """
        Get recent commits asynchronously.

        Args:
            repo_path: Full path to the git repository
            limit: Maximum number of commits to retrieve
            no_merges: If True, exclude merge commits

        Returns:
            List of Commit objects
        """
        merges_flag = "--no-merges" if no_merges else ""
        command = f'log {merges_flag} --pretty=format:"%H|%an|%aI|%s" -{limit}'

        result = await self.execute_git_command_async(command, repo_path)
        if not result.success or not result.stdout:
            return []

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('|', 3)
            if len(parts) >= 4:
                commits.append(Commit(
                    hash=parts[0],
                    author=parts[1],
                    date=parts[2],
                    message=parts[3]
                ))

        return commits

    def get_last_pull_time(self, repo_path: str) -> Optional[str]:
        """
        Get the timestamp of the last commit (approximates last pull time).

        Args:
            repo_path: Full path to the git repository

        Returns:
            ISO date string or None on error
        """
        result = self.execute_git_command("log -1 --format=%cI", repo_path)
        if result.success and result.stdout:
            return result.stdout.strip()
        return None

    def get_current_head(self, repo_path: str) -> Optional[str]:
        """
        Get the current HEAD commit hash.

        Args:
            repo_path: Full path to the git repository

        Returns:
            Commit hash or None on error
        """
        result = self.execute_git_command("rev-parse HEAD", repo_path)
        if result.success and result.stdout:
            return result.stdout.strip()
        return None

    async def get_current_head_async(self, repo_path: str) -> Optional[str]:
        """
        Get the current HEAD commit hash asynchronously.

        Args:
            repo_path: Full path to the git repository

        Returns:
            Commit hash or None on error
        """
        result = await self.execute_git_command_async("rev-parse HEAD", repo_path)
        if result.success and result.stdout:
            return result.stdout.strip()
        return None

    # =========================================================================
    # Pull Operations
    # =========================================================================

    def pull_repository(self, repo_path: str) -> PullResult:
        """
        Execute git pull operation.

        Args:
            repo_path: Full path to the git repository

        Returns:
            PullResult with success status and output
        """
        result = self.execute_git_command("pull", repo_path, timeout=120)

        output = result.stdout or result.stderr
        is_up_to_date = (
            "Already up to date" in output or
            "Already up-to-date" in output
        )

        return PullResult(
            success=result.success,
            output=output,
            is_already_up_to_date=is_up_to_date,
            error=result.error if not result.success else None
        )

    async def pull_repository_async(self, repo_path: str) -> PullResult:
        """
        Execute git pull operation asynchronously.

        Args:
            repo_path: Full path to the git repository

        Returns:
            PullResult with success status and output
        """
        result = await self.execute_git_command_async("pull", repo_path, timeout=120)

        output = result.stdout or result.stderr
        is_up_to_date = (
            "Already up to date" in output or
            "Already up-to-date" in output
        )

        return PullResult(
            success=result.success,
            output=output,
            is_already_up_to_date=is_up_to_date,
            error=result.error if not result.success else None
        )

    # =========================================================================
    # File Change Analysis
    # =========================================================================

    def get_changed_files(
        self,
        repo_path: str,
        from_commit: str,
        to_commit: str = "HEAD"
    ) -> List[str]:
        """
        Get list of changed files between two commits.

        Args:
            repo_path: Full path to the git repository
            from_commit: Starting commit hash
            to_commit: Ending commit hash (default: HEAD)

        Returns:
            List of changed file paths
        """
        command = f"diff {from_commit} {to_commit} --name-only"
        result = self.execute_git_command(command, repo_path)

        if not result.success or not result.stdout:
            logger.warning(f"Could not get changed files: {result.error}")
            return []

        return [
            f.strip() for f in result.stdout.split('\n')
            if f.strip()
        ]

    async def get_changed_files_async(
        self,
        repo_path: str,
        from_commit: str,
        to_commit: str = "HEAD"
    ) -> List[str]:
        """
        Get list of changed files between two commits asynchronously.

        Args:
            repo_path: Full path to the git repository
            from_commit: Starting commit hash
            to_commit: Ending commit hash (default: HEAD)

        Returns:
            List of changed file paths
        """
        command = f"diff {from_commit} {to_commit} --name-only"
        result = await self.execute_git_command_async(command, repo_path)

        if not result.success or not result.stdout:
            return []

        return [
            f.strip() for f in result.stdout.split('\n')
            if f.strip()
        ]

    def get_changed_files_with_status(
        self,
        repo_path: str,
        from_commit: str = "HEAD@{1}",
        to_commit: str = "HEAD"
    ) -> List[FileChange]:
        """
        Get detailed diff with file status (Added, Modified, Deleted).

        Args:
            repo_path: Full path to the git repository
            from_commit: Starting commit hash
            to_commit: Ending commit hash

        Returns:
            List of FileChange objects with status
        """
        command = f"diff --name-status {from_commit} {to_commit}"
        result = self.execute_git_command(command, repo_path)

        if not result.success or not result.stdout:
            logger.warning(f"Could not get file status: {result.error}")
            return []

        files = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                status_code = parts[0].strip()
                file_path = parts[1].strip()

                # Map status codes to FileStatus enum
                status_map = {
                    'A': FileStatus.ADDED,
                    'M': FileStatus.MODIFIED,
                    'D': FileStatus.DELETED,
                    'R': FileStatus.RENAMED,
                    'C': FileStatus.COPIED,
                }
                status = status_map.get(status_code[0], FileStatus.MODIFIED)

                files.append(FileChange(
                    file=file_path,
                    status=status,
                    status_code=status_code
                ))

        return files

    # =========================================================================
    # Commit Analysis
    # =========================================================================

    def analyze_commits_by_date_range(
        self,
        repo_path: str,
        start_date: str,
        end_date: str
    ) -> List[CommitDetails]:
        """
        Analyze commits within a date range.

        Args:
            repo_path: Full path to the git repository
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of CommitDetails with file statistics
        """
        command = (
            f'log --since="{start_date}" --until="{end_date}" '
            f'--no-merges --pretty=format:"%H|%h|%an|%aI|%s" --numstat'
        )

        result = self.execute_git_command(command, repo_path, timeout=300)
        if not result.success or not result.stdout:
            logger.error(f"Failed to analyze commits: {result.error}")
            return []

        return self._parse_numstat_output(result.stdout)

    async def analyze_commits_by_date_range_async(
        self,
        repo_path: str,
        start_date: str,
        end_date: str
    ) -> List[CommitDetails]:
        """
        Analyze commits within a date range asynchronously.

        Args:
            repo_path: Full path to the git repository
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of CommitDetails with file statistics
        """
        command = (
            f'log --since="{start_date}" --until="{end_date}" '
            f'--no-merges --pretty=format:"%H|%h|%an|%aI|%s" --numstat'
        )

        result = await self.execute_git_command_async(command, repo_path, timeout=300)
        if not result.success or not result.stdout:
            return []

        return self._parse_numstat_output(result.stdout)

    def _parse_numstat_output(self, output: str) -> List[CommitDetails]:
        """
        Parse git log --numstat output into CommitDetails.

        Args:
            output: Raw git log output

        Returns:
            List of CommitDetails
        """
        commits = []
        current_commit: Optional[CommitDetails] = None
        lines = output.strip().split('\n')

        for line in lines:
            if not line.strip():
                # Empty line separates commits
                if current_commit:
                    commits.append(current_commit)
                    current_commit = None
                continue

            # Check if this is a commit header line
            if '|' in line and len(line.split('|')) == 5:
                # Save previous commit if exists
                if current_commit:
                    commits.append(current_commit)

                # Parse commit header
                parts = line.split('|')
                current_commit = CommitDetails(
                    hash=parts[0],
                    hash_short=parts[1],
                    author=parts[2],
                    date=parts[3],
                    message=parts[4],
                    files_changed=0,
                    files=[],
                    modified_items=[]
                )

            elif current_commit:
                # This is a numstat line (additions\tdeletions\tfilename)
                stat_parts = line.split('\t')
                if len(stat_parts) >= 3:
                    additions = 0 if stat_parts[0] == '-' else int(stat_parts[0] or 0)
                    deletions = 0 if stat_parts[1] == '-' else int(stat_parts[1] or 0)
                    filename = stat_parts[2]

                    current_commit.files.append(CommitFile(
                        filename=filename,
                        additions=additions,
                        deletions=deletions
                    ))
                    current_commit.files_changed += 1

        # Add the last commit if exists
        if current_commit:
            commits.append(current_commit)

        return commits

    def get_commit_details(self, repo_path: str, commit_hash: str) -> Optional[CommitDetails]:
        """
        Get details for a specific commit.

        Args:
            repo_path: Full path to the git repository
            commit_hash: Commit hash to analyze

        Returns:
            CommitDetails or None if not found
        """
        command = f'show {commit_hash} --format="%H|%h|%an|%aI|%s" --name-only'
        result = self.execute_git_command(command, repo_path)

        if not result.success or not result.stdout:
            logger.error(f"Failed to get commit details: {result.error}")
            return None

        lines = result.stdout.strip().split('\n')
        if not lines:
            return None

        # Parse header line
        header_line = lines[0]
        parts = header_line.split('|')
        if len(parts) < 5:
            return None

        # Get changed files (skip diff output, only get filenames)
        changed_files = [
            line.strip() for line in lines[1:]
            if line.strip() and not line.startswith('diff') and
            (line.endswith('.cs') or line.endswith('.js') or
             line.endswith('.ts') or line.endswith('.sql'))
        ]

        return CommitDetails(
            hash=parts[0],
            hash_short=parts[1],
            author=parts[2],
            date=parts[3],
            message=parts[4],
            files_changed=len(changed_files),
            files=[CommitFile(filename=f, additions=0, deletions=0) for f in changed_files],
            modified_items=[]
        )

    # =========================================================================
    # Repository Information
    # =========================================================================

    async def get_repository_info(self, repo_path: str) -> RepositoryInfo:
        """
        Get enriched repository information with status.

        Args:
            repo_path: Full path to the git repository

        Returns:
            RepositoryInfo with recent commits and status
        """
        repo_name = Path(repo_path).name

        try:
            recent_commits = await self.get_recent_commits_async(repo_path, limit=5)
            last_sync = self.get_last_pull_time(repo_path)

            return RepositoryInfo(
                path=repo_path,
                name=repo_name,
                recent_commits=recent_commits,
                last_sync=last_sync,
                status="active" if recent_commits else "error"
            )

        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return RepositoryInfo(
                path=repo_path,
                name=repo_name,
                recent_commits=[],
                last_sync=None,
                status="error",
                error=str(e)
            )

    def get_repository_info_sync(self, repo_path: str) -> RepositoryInfo:
        """
        Get enriched repository information synchronously.

        Args:
            repo_path: Full path to the git repository

        Returns:
            RepositoryInfo with recent commits and status
        """
        repo_name = Path(repo_path).name

        try:
            recent_commits = self.get_recent_commits(repo_path, limit=5)
            last_sync = self.get_last_pull_time(repo_path)

            return RepositoryInfo(
                path=repo_path,
                name=repo_name,
                recent_commits=recent_commits,
                last_sync=last_sync,
                status="active" if recent_commits else "error"
            )

        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return RepositoryInfo(
                path=repo_path,
                name=repo_name,
                recent_commits=[],
                last_sync=None,
                status="error",
                error=str(e)
            )
