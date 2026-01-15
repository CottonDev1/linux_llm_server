"""
Git operations routes for repository management and commit analysis.

Provides endpoints for:
- Repository listing and information
- Git pull operations with code analysis
- Commit history and impact analysis
- File change tracking

NOTE: This is separate from git_routes.py which handles database/configuration
for git repositories. This module handles actual git operations.
"""
from typing import Optional

from fastapi import APIRouter, Request, Query, HTTPException

from git_service import (
    GitService, CodeAnalyzer,
    PullRequest, DateRangeRequest, CommitImpactRequest
)
from config import GIT_ROOT
from log_service import log_pipeline, log_error

# Create router with /git prefix
router = APIRouter(prefix="/git", tags=["Git"])

# Initialize git service (singleton pattern)
_git_service: Optional[GitService] = None
_code_analyzer: Optional[CodeAnalyzer] = None


def get_git_service() -> GitService:
    """Get or create GitService singleton."""
    global _git_service
    if _git_service is None:
        _git_service = GitService(git_root=GIT_ROOT)
    return _git_service


def get_code_analyzer() -> CodeAnalyzer:
    """Get or create CodeAnalyzer singleton."""
    global _code_analyzer
    if _code_analyzer is None:
        _code_analyzer = CodeAnalyzer()
    return _code_analyzer


@router.get("/repositories")
async def list_git_repositories(request: Request):
    """
    List all available git repositories.

    Scans the configured git root directory for repositories.
    Returns list of repositories with name, path, and display name.
    """
    user_ip = request.client.host if request.client else "Unknown"
    git_svc = get_git_service()
    repos = git_svc.scan_repositories()

    log_pipeline("GIT", user_ip, "Repository list retrieved",
                 details={"count": len(repos)})

    return {
        "success": True,
        "repositories": [repo.model_dump() for repo in repos]
    }


@router.get("/repositories/{repo_name}/info")
async def get_repository_info(repo_name: str, request: Request):
    """
    Get detailed information about a specific repository.

    Returns recent commits, last sync time, and status.
    """
    user_ip = request.client.host if request.client else "Unknown"
    git_svc = get_git_service()
    repo_path = git_svc.get_repository_path(repo_name)

    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_name}")

    info = await git_svc.get_repository_info(repo_path)

    log_pipeline("GIT", user_ip, "Repository info fetched",
                 repo_name, details={"commits_count": len(info.recent_commits) if info.recent_commits else 0})

    return {
        "success": True,
        "repository": info.model_dump()
    }


@router.get("/repositories/{repo_name}/commits")
async def get_repository_commits(
    repo_name: str,
    request: Request,
    limit: int = Query(10, ge=1, le=100, description="Number of commits to return"),
    no_merges: bool = Query(True, description="Exclude merge commits")
):
    """
    Get recent commits from a repository.
    """
    user_ip = request.client.host if request.client else "Unknown"
    git_svc = get_git_service()
    repo_path = git_svc.get_repository_path(repo_name)

    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_name}")

    commits = await git_svc.get_recent_commits_async(repo_path, limit=limit, no_merges=no_merges)

    log_pipeline("GIT", user_ip, "Recent commits retrieved",
                 details={"repo": repo_name, "limit": limit, "count": len(commits)})

    return {
        "success": True,
        "repo": repo_name,
        "commits": [c.model_dump() for c in commits]
    }


@router.post("/pull")
async def pull_repository(pull_req: PullRequest, http_request: Request):
    """
    Pull a git repository and optionally analyze changes.

    This is the main endpoint for syncing repositories.
    Returns pull status, changed files, and optional code analysis.
    """
    user_ip = http_request.client.host if http_request.client else "Unknown"
    log_pipeline("GIT", user_ip, "Git pull started", details={"repo": pull_req.repo})

    git_svc = get_git_service()
    analyzer = get_code_analyzer()

    repo_path = git_svc.get_repository_path(pull_req.repo)
    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {pull_req.repo}")

    # Get current HEAD before pull
    previous_head = await git_svc.get_current_head_async(repo_path)

    # Execute pull
    pull_result = await git_svc.pull_repository_async(repo_path)

    if not pull_result.success:
        log_error("GIT", user_ip, "Git pull failed",
                  details={"repo": pull_req.repo, "error": pull_result.error})
        return {
            "success": False,
            "repo": pull_req.repo,
            "output": pull_result.output,
            "message": f"Git pull failed: {pull_result.error}",
            "hasChanges": False,
            "changedFiles": []
        }

    response = {
        "success": True,
        "repo": pull_req.repo,
        "output": pull_result.output,
        "message": "Pull operation completed successfully",
        "hasChanges": False,
        "changedFiles": [],
        "codeAnalysis": None
    }

    # Check if already up to date
    if pull_result.is_already_up_to_date:
        response["message"] = "Repository is already up to date"
        log_pipeline("GIT", user_ip, "Git pull completed",
                     details={"repo": pull_req.repo, "success": True, "has_changes": False})
        return response

    response["hasChanges"] = True

    # Analyze changes if requested
    if pull_req.analyze_changes and previous_head:
        changed_files = await git_svc.get_changed_files_async(repo_path, previous_head)
        response["changedFiles"] = changed_files

        if pull_req.include_code_analysis and changed_files:
            # Filter for code files and limit
            code_files = analyzer.filter_code_files(changed_files)
            files_to_analyze = code_files[:pull_req.max_files_to_analyze]

            if files_to_analyze:
                analysis_results = await analyzer.analyze_multiple_files(
                    files_to_analyze, repo_path
                )
                response["codeAnalysis"] = [r.model_dump() for r in analysis_results]

    log_pipeline("GIT", user_ip, "Git pull completed",
                 details={"repo": pull_req.repo, "success": True, "has_changes": True,
                          "files_changed": len(response.get("changedFiles", []))})

    return response


@router.post("/analyze-past")
async def analyze_past_commits(date_req: DateRangeRequest, http_request: Request):
    """
    Analyze commits within a date range.

    Returns commit history with file statistics and modified code items.
    Useful for impact analysis and code review.
    """
    user_ip = http_request.client.host if http_request.client else "Unknown"
    git_svc = get_git_service()
    analyzer = get_code_analyzer()

    repo_path = git_svc.get_repository_path(date_req.repo)
    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {date_req.repo}")

    commits = await git_svc.analyze_commits_by_date_range_async(
        repo_path, date_req.start_date, date_req.end_date
    )

    # Analyze each commit's files for classes/methods
    for commit in commits:
        for file_info in commit.files:
            if analyzer.is_code_file(file_info.filename):
                result = analyzer.analyze_file_sync(file_info.filename, repo_path)
                if result.success:
                    for cls in result.classes:
                        commit.modified_items.append({"type": "class", "name": cls})
                    for method in result.methods:
                        commit.modified_items.append({"type": "method", "name": method})

    log_pipeline("GIT", user_ip, "Historical analysis",
                 details={"repo": date_req.repo, "start_date": date_req.start_date,
                          "end_date": date_req.end_date, "commit_count": len(commits)})

    return {
        "success": True,
        "repo": date_req.repo,
        "startDate": date_req.start_date,
        "endDate": date_req.end_date,
        "commitCount": len(commits),
        "commits": [c.model_dump() for c in commits]
    }


@router.post("/analyze-commit-impact")
async def analyze_commit_impact(impact_req: CommitImpactRequest, http_request: Request):
    """
    Analyze the impact of a specific commit.

    Returns changed files with detailed code analysis
    (classes, methods, and functions modified).
    """
    user_ip = http_request.client.host if http_request.client else "Unknown"
    git_svc = get_git_service()
    analyzer = get_code_analyzer()

    repo_path = git_svc.get_repository_path(impact_req.repo)
    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {impact_req.repo}")

    commit_details = git_svc.get_commit_details(repo_path, impact_req.commit_hash)
    if not commit_details:
        raise HTTPException(status_code=404, detail=f"Commit not found: {impact_req.commit_hash}")

    # Analyze files in the commit
    code_files = [f.filename for f in commit_details.files if analyzer.is_code_file(f.filename)]
    analysis_results = await analyzer.analyze_multiple_files(code_files, repo_path)

    # Build modified items list
    modified_items = []
    for result in analysis_results:
        if result.success:
            for cls in result.classes:
                modified_items.append({
                    "type": "class",
                    "name": cls,
                    "file": result.file
                })
            for method in result.methods:
                modified_items.append({
                    "type": "method",
                    "name": method,
                    "file": result.file
                })

    log_pipeline("GIT", user_ip, "Commit impact analysis",
                 details={"repo": impact_req.repo, "commit_hash": impact_req.commit_hash,
                          "files_analyzed": len(code_files)})

    return {
        "success": True,
        "repo": impact_req.repo,
        "commit": commit_details.model_dump(),
        "codeAnalysis": [r.model_dump() for r in analysis_results],
        "modifiedItems": modified_items
    }


@router.get("/changed-files/{repo_name}")
async def get_changed_files(
    repo_name: str,
    request: Request,
    from_commit: str = Query(..., description="Starting commit hash"),
    to_commit: str = Query("HEAD", description="Ending commit hash")
):
    """
    Get list of files changed between two commits.
    """
    user_ip = request.client.host if request.client else "Unknown"
    git_svc = get_git_service()

    repo_path = git_svc.get_repository_path(repo_name)
    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_name}")

    files = await git_svc.get_changed_files_async(repo_path, from_commit, to_commit)

    log_pipeline("GIT", user_ip, "Changed files retrieved",
                 details={"repo": repo_name, "from_commit": from_commit,
                          "to_commit": to_commit, "count": len(files)})

    return {
        "success": True,
        "repo": repo_name,
        "fromCommit": from_commit,
        "toCommit": to_commit,
        "changedFiles": files
    }


@router.get("/file-status/{repo_name}")
async def get_file_status(
    repo_name: str,
    request: Request,
    from_commit: str = Query("HEAD@{1}", description="Starting commit hash"),
    to_commit: str = Query("HEAD", description="Ending commit hash")
):
    """
    Get changed files with status (Added, Modified, Deleted).
    """
    user_ip = request.client.host if request.client else "Unknown"
    git_svc = get_git_service()

    repo_path = git_svc.get_repository_path(repo_name)
    if not repo_path:
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_name}")

    files = git_svc.get_changed_files_with_status(repo_path, from_commit, to_commit)

    # Count by status
    added = sum(1 for f in files if f.status == "Added")
    modified = sum(1 for f in files if f.status == "Modified")
    deleted = sum(1 for f in files if f.status == "Deleted")

    log_pipeline("GIT", user_ip, "File status fetched",
                 details={"repo": repo_name, "added": added, "modified": modified, "deleted": deleted})

    return {
        "success": True,
        "repo": repo_name,
        "fromCommit": from_commit,
        "toCommit": to_commit,
        "files": [f.model_dump() for f in files]
    }
