"""
Pydantic models for Git service data structures.

These models define the data structures used throughout the Git service,
matching the response formats from the original JavaScript implementation.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# Environment-based configuration
GIT_ROOT = os.environ.get('GIT_ROOT')
if not GIT_ROOT:
    raise EnvironmentError("GIT_ROOT environment variable is required")


class FileStatus(str, Enum):
    """Git file status codes"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    UNTRACKED = "untracked"


# ============================================================================
# Repository Models
# ============================================================================

class RepositoryConfig(BaseModel):
    """Configuration for a tracked repository"""
    name: str = Field(..., description="Repository identifier/key")
    path: str = Field(..., description="Full filesystem path to repository")
    display_name: str = Field(..., description="Human-readable name")
    file_extensions: List[str] = Field(
        default_factory=lambda: [".cs", ".js", ".sql", ".config", ".xml"],
        description="File extensions to track for this repository"
    )
    enabled: bool = Field(default=True, description="Whether repo is active for syncing")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "gin",
                "path": os.path.join(GIT_ROOT, "Gin"),
                "display_name": "Gin",
                "file_extensions": [".sql", ".cs", ".js", ".config", ".xml"],
                "enabled": True
            }
        }
    )


class Repository(BaseModel):
    """Basic repository information"""
    name: str = Field(..., description="Repository folder name")
    path: str = Field(..., description="Full filesystem path")
    display_name: str = Field(..., description="Human-readable display name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Gin",
                "path": os.path.join(GIT_ROOT, "Gin"),
                "display_name": "Gin"
            }
        }
    )


# ============================================================================
# Commit Models
# ============================================================================

class Commit(BaseModel):
    """Basic commit information from git log"""
    hash: str = Field(..., description="Full commit hash (40 chars)")
    author: str = Field(..., description="Commit author name")
    date: str = Field(..., description="Commit date in ISO format")
    message: str = Field(..., description="Commit message subject line")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hash": "abc123def456...",
                "author": "Developer Name",
                "date": "2024-01-15T10:30:00-05:00",
                "message": "Fix bug in order processing"
            }
        }
    )


class CommitFile(BaseModel):
    """File statistics from a commit (numstat output)"""
    filename: str = Field(..., description="Relative file path")
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")


class CommitDetails(BaseModel):
    """Detailed commit information with file statistics"""
    hash: str = Field(..., description="Full commit hash")
    hash_short: str = Field(..., description="Short commit hash (7 chars)")
    author: str = Field(..., description="Commit author name")
    date: str = Field(..., description="Commit date in ISO format")
    message: str = Field(..., description="Commit message subject line")
    files_changed: int = Field(default=0, description="Number of files changed")
    files: List[CommitFile] = Field(default_factory=list, description="File change details")
    modified_items: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Extracted classes/methods modified"
    )


# ============================================================================
# File Change Models
# ============================================================================

class FileChange(BaseModel):
    """File change information with status"""
    file: str = Field(..., description="Relative file path")
    status: FileStatus = Field(..., description="Change status (added/modified/deleted)")
    status_code: str = Field(..., description="Git status code (A/M/D/R/C)")


# ============================================================================
# Operation Result Models
# ============================================================================

class PullResult(BaseModel):
    """Result from git pull operation"""
    success: bool = Field(..., description="Whether pull succeeded")
    output: str = Field(default="", description="Git command output")
    is_already_up_to_date: bool = Field(default=False, description="True if no new commits")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RepositoryInfo(BaseModel):
    """Enriched repository information with status"""
    path: str = Field(..., description="Full filesystem path")
    name: str = Field(..., description="Repository folder name")
    recent_commits: List[Commit] = Field(default_factory=list, description="Recent commits")
    last_sync: Optional[str] = Field(default=None, description="Last sync timestamp")
    status: str = Field(default="unknown", description="Repository status (active/error)")
    error: Optional[str] = Field(default=None, description="Error message if any")


# ============================================================================
# Code Analysis Models
# ============================================================================

class CodeAnalysisResult(BaseModel):
    """Result from analyzing a code file"""
    file: str = Field(..., description="Relative file path")
    classes: List[str] = Field(default_factory=list, description="Extracted class names")
    methods: List[str] = Field(default_factory=list, description="Extracted method/function names")
    success: bool = Field(default=True, description="Whether analysis succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DateRangeAnalysis(BaseModel):
    """Analysis result for commits in a date range"""
    repo: str = Field(..., description="Repository name")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    commit_count: int = Field(default=0, description="Number of commits in range")
    commits: List[CommitDetails] = Field(default_factory=list, description="Commit details")


# ============================================================================
# API Request/Response Models
# ============================================================================

class PullRequest(BaseModel):
    """Request model for pull operation"""
    repo: str = Field(..., description="Repository name to pull")
    analyze_changes: bool = Field(default=True, description="Whether to analyze changed files")
    max_files_to_analyze: int = Field(default=20, description="Max files to analyze")
    include_code_analysis: bool = Field(default=True, description="Include class/method extraction")


class PullResponse(BaseModel):
    """Response model for pull operation"""
    success: bool
    repo: str
    output: str
    message: str
    has_changes: bool = False
    changed_files: List[str] = Field(default_factory=list)
    code_analysis: Optional[List[CodeAnalysisResult]] = None


class RepositoryListResponse(BaseModel):
    """Response model for listing repositories"""
    success: bool
    repositories: List[Repository]


class DateRangeRequest(BaseModel):
    """Request model for date range analysis"""
    repo: str = Field(..., description="Repository name")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class CommitImpactRequest(BaseModel):
    """Request model for commit impact analysis"""
    repo: str = Field(..., description="Repository name")
    commit_hash: str = Field(..., description="Commit hash to analyze")


class GitCommandResult(BaseModel):
    """Result from executing a git command"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error: Optional[str] = None
