"""
Prefect Git Analysis Pipeline

Processes git repositories through:
1. Pull Repository - Fetch latest changes from remote
2. Analyze Commits - Analyze commit history for a date range
3. Extract File Changes - Get detailed file changes for each commit
4. Store Results - Track analysis results for monitoring

Features:
- Integrated with GitService for repository operations
- Commit history analysis with file statistics
- Date range filtering
- Built-in retries for resilience
- Prefect metrics tracking via emit_event
"""

import asyncio
import time
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.events import emit_event

# Add paths for imports
_services_path = os.path.dirname(os.path.dirname(__file__))
if _services_path not in sys.path:
    sys.path.insert(0, _services_path)

# Configuration
GIT_ROOT = os.getenv("GIT_ROOT", r"C:\Projects\Git")


@dataclass
class PullResult:
    """Result from git pull task"""
    repo_path: str
    repo_name: str = ""
    success: bool = False
    output: str = ""
    is_already_up_to_date: bool = True
    error: str = ""
    commits_ahead: int = 0


@dataclass
class CommitInfo:
    """Information about a single commit"""
    hash: str
    hash_short: str = ""
    author: str = ""
    date: str = ""
    message: str = ""
    files_changed: int = 0
    files: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result from git analysis task"""
    repo_path: str
    repo_name: str = ""
    start_date: str = ""
    end_date: str = ""
    commits: List[CommitInfo] = field(default_factory=list)
    total_commits: int = 0
    total_files_changed: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    success: bool = True
    error: str = ""


# Lazy-loaded git service
_git_service = None


def get_git_service():
    """Get or create GitService singleton."""
    global _git_service
    if _git_service is None:
        try:
            from git_service.git_service import GitService
            _git_service = GitService(git_root=GIT_ROOT)
        except ImportError as e:
            raise RuntimeError(f"GitService not available: {e}")
    return _git_service


@task(
    name="pull_repository",
    description="Pull latest changes from git remote",
    retries=2,
    retry_delay_seconds=10,
    tags=["git", "pull"]
)
async def pull_repository_task(repo_path: str) -> PullResult:
    """
    Pull latest changes from a git repository.

    Args:
        repo_path: Path to the git repository

    Returns:
        PullResult with pull status and output
    """
    logger = get_run_logger()
    repo_name = os.path.basename(repo_path)

    logger.info(f"Pulling repository: {repo_name} ({repo_path})")

    result = PullResult(repo_path=repo_path, repo_name=repo_name)

    if not os.path.exists(repo_path):
        result.error = f"Repository path does not exist: {repo_path}"
        logger.error(result.error)
        return result

    try:
        git_service = get_git_service()
        pull_result = await git_service.pull_repository_async(repo_path)

        result.success = pull_result.success
        result.output = pull_result.output
        result.is_already_up_to_date = pull_result.is_already_up_to_date
        result.error = pull_result.error or ""

        if result.success:
            logger.info(f"Pull successful for {repo_name}: {'Already up to date' if result.is_already_up_to_date else 'Changes pulled'}")
        else:
            logger.error(f"Pull failed for {repo_name}: {result.error}")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Pull failed for {repo_name}: {e}")

    return result


@task(
    name="analyze_commits",
    description="Analyze commit history for date range",
    retries=1,
    retry_delay_seconds=5,
    tags=["git", "analysis"]
)
async def analyze_commits_task(
    repo_path: str,
    start_date: str,
    end_date: str
) -> AnalysisResult:
    """
    Analyze commits within a date range.

    Args:
        repo_path: Path to the git repository
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        AnalysisResult with commit statistics
    """
    logger = get_run_logger()
    repo_name = os.path.basename(repo_path)

    logger.info(f"Analyzing commits for {repo_name}: {start_date} to {end_date}")

    result = AnalysisResult(
        repo_path=repo_path,
        repo_name=repo_name,
        start_date=start_date,
        end_date=end_date
    )

    try:
        git_service = get_git_service()

        # Get commits in date range
        commits = await git_service.analyze_commits_by_date_range_async(
            repo_path, start_date, end_date
        )

        result.total_commits = len(commits)

        for commit in commits:
            commit_info = CommitInfo(
                hash=commit.hash,
                hash_short=commit.hash_short,
                author=commit.author,
                date=commit.date,
                message=commit.message,
                files_changed=len(commit.files),
                files=[{
                    "filename": f.filename,
                    "additions": f.additions,
                    "deletions": f.deletions
                } for f in commit.files]
            )
            result.commits.append(commit_info)

            # Aggregate statistics
            result.total_files_changed += len(commit.files)
            for f in commit.files:
                result.total_additions += f.additions
                result.total_deletions += f.deletions

        logger.info(f"Analysis complete: {result.total_commits} commits, {result.total_files_changed} files changed")

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"Analysis failed for {repo_name}: {e}")

    # Create Prefect artifact
    await create_markdown_artifact(
        key="git-analysis-result",
        markdown=f"""
## Git Analysis Results - {repo_name}

- **Repository**: {repo_path}
- **Date Range**: {start_date} to {end_date}
- **Total Commits**: {result.total_commits}
- **Files Changed**: {result.total_files_changed}
- **Lines Added**: {result.total_additions}
- **Lines Deleted**: {result.total_deletions}
- **Status**: {"Success" if result.success else "Failed"}

### Recent Commits
{chr(10).join([f"- [{c.hash_short}] {c.message[:60]}{'...' if len(c.message) > 60 else ''} ({c.author})" for c in result.commits[:5]])}
        """,
        description=f"Git analysis for {repo_name}"
    )

    return result


@flow(
    name="git-analysis-pipeline",
    description="Git Repository Analysis Pipeline",
    retries=1,
    retry_delay_seconds=30
)
async def git_analysis_flow(
    repo_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    pull_first: bool = True
) -> Dict[str, Any]:
    """
    Git Analysis Pipeline - Analyze repository commit history.

    This flow:
    1. Optionally pulls latest changes
    2. Analyzes commits within the date range
    3. Returns aggregated statistics

    Args:
        repo_path: Path to git repository
        start_date: Start date (YYYY-MM-DD format), defaults to 7 days ago
        end_date: End date (YYYY-MM-DD format), defaults to today
        pull_first: Whether to pull before analyzing

    Returns:
        Dict with analysis results
    """
    logger = get_run_logger()
    flow_start = time.time()

    repo_name = os.path.basename(repo_path)

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info(f"Starting git analysis for {repo_name}: {start_date} to {end_date}")

    pull_result = None

    # Step 1: Pull repository (optional)
    if pull_first:
        pull_result = await pull_repository_task(repo_path)
        if not pull_result.success:
            logger.warning(f"Pull failed, continuing with local state: {pull_result.error}")

    # Step 2: Analyze commits
    analysis_result = await analyze_commits_task(
        repo_path=repo_path,
        start_date=start_date,
        end_date=end_date
    )

    total_duration = time.time() - flow_start

    # Create flow summary artifact
    await create_markdown_artifact(
        key="git-pipeline-summary",
        markdown=f"""
# Git Analysis Complete - {repo_name}

## Overview
- **Repository**: {repo_path}
- **Date Range**: {start_date} to {end_date}
- **Processing Time**: {total_duration:.2f}s
- **Status**: {"Success" if analysis_result.success else "Failed"}

## Results
| Metric | Value |
|--------|-------|
| Total Commits | {analysis_result.total_commits} |
| Files Changed | {analysis_result.total_files_changed} |
| Lines Added | {analysis_result.total_additions} |
| Lines Deleted | {analysis_result.total_deletions} |

{"## Pull Status" + chr(10) + f"- Already Up to Date: {pull_result.is_already_up_to_date}" if pull_result else ""}

{"## Error" + chr(10) + analysis_result.error if analysis_result.error else ""}
        """,
        description=f"Git analysis summary for {repo_name}"
    )

    # Emit Prefect event for metrics tracking
    emit_event(
        event="git.analysis.completed",
        resource={"prefect.resource.id": "git-pipeline"},
        payload={
            "repository": repo_name,
            "repo_path": repo_path,
            "start_date": start_date,
            "end_date": end_date,
            "processing_time_seconds": total_duration,
            "total_commits": analysis_result.total_commits,
            "total_files_changed": analysis_result.total_files_changed,
            "total_additions": analysis_result.total_additions,
            "total_deletions": analysis_result.total_deletions,
            "pulled": pull_first,
            "had_changes": not (pull_result.is_already_up_to_date if pull_result else True),
            "success": analysis_result.success,
        }
    )

    if not analysis_result.success:
        emit_event(
            event="git.analysis.failed",
            resource={"prefect.resource.id": "git-pipeline"},
            payload={
                "repository": repo_name,
                "error": analysis_result.error,
                "severity": "error",
            }
        )

    return {
        "success": analysis_result.success,
        "repo_path": repo_path,
        "repo_name": repo_name,
        "start_date": start_date,
        "end_date": end_date,
        "processing_time_seconds": total_duration,
        "pull_result": {
            "success": pull_result.success if pull_result else None,
            "is_already_up_to_date": pull_result.is_already_up_to_date if pull_result else None,
        } if pull_result else None,
        "statistics": {
            "total_commits": analysis_result.total_commits,
            "total_files_changed": analysis_result.total_files_changed,
            "total_additions": analysis_result.total_additions,
            "total_deletions": analysis_result.total_deletions,
        },
        "commits": [
            {
                "hash": c.hash,
                "hash_short": c.hash_short,
                "author": c.author,
                "date": c.date,
                "message": c.message,
                "files_changed": c.files_changed,
                "files": c.files,
            }
            for c in analysis_result.commits
        ],
        "error": analysis_result.error if not analysis_result.success else None,
    }


def run_git_analysis_flow(
    repo_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    pull_first: bool = True,
    use_prefect: bool = True
) -> Dict[str, Any]:
    """
    Run the git analysis flow synchronously.

    This is the main entry point for API calls.

    Args:
        repo_path: Path to git repository
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        pull_first: Whether to pull before analyzing
        use_prefect: If True, run through Prefect for tracking

    Returns:
        Dict with analysis results
    """
    if not os.path.exists(repo_path):
        return {
            "success": False,
            "error": f"Repository path does not exist: {repo_path}"
        }

    if use_prefect:
        try:
            result = asyncio.run(git_analysis_flow(
                repo_path=repo_path,
                start_date=start_date,
                end_date=end_date,
                pull_first=pull_first
            ))
            return result
        except Exception as e:
            # Fall through to direct execution
            pass

    # Direct execution (fallback)
    try:
        git_service = get_git_service()

        # Set default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Pull if requested
        if pull_first:
            git_service.pull_repository(repo_path)

        # Analyze commits
        commits = git_service.analyze_commits_by_date_range(
            repo_path, start_date, end_date
        )

        repo_name = os.path.basename(repo_path)

        return {
            "success": True,
            "repo_path": repo_path,
            "repo_name": repo_name,
            "start_date": start_date,
            "end_date": end_date,
            "statistics": {
                "total_commits": len(commits),
                "total_files_changed": sum(len(c.files) for c in commits),
                "total_additions": sum(sum(f.additions for f in c.files) for c in commits),
                "total_deletions": sum(sum(f.deletions for f in c.files) for c in commits),
            },
            "commits": [
                {
                    "hash": c.hash,
                    "hash_short": c.hash_short,
                    "author": c.author,
                    "date": c.date,
                    "message": c.message,
                    "files_changed": len(c.files),
                }
                for c in commits
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
        start_date = sys.argv[2] if len(sys.argv) > 2 else None
        end_date = sys.argv[3] if len(sys.argv) > 3 else None

        result = run_git_analysis_flow(
            repo_path=repo_path,
            start_date=start_date,
            end_date=end_date,
            pull_first=False
        )
        print(f"Result: {result}")
    else:
        print("Usage: python git_flow.py <repo_path> [start_date] [end_date]")
