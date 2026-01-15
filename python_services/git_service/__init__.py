"""
Git Service Module

Provides Git repository operations for the Python services layer.
Migrated from JavaScript GitService.js to Python.
"""

from .models import (
    Repository,
    RepositoryConfig,
    Commit,
    CommitFile,
    CommitDetails,
    FileChange,
    FileStatus,
    PullResult,
    RepositoryInfo,
    CodeAnalysisResult,
    DateRangeAnalysis,
    PullRequest,
    PullResponse,
    RepositoryListResponse,
    DateRangeRequest,
    CommitImpactRequest,
)
from .git_service import GitService
from .repository_scanner import RepositoryScanner
from .code_analyzer import CodeAnalyzer

__all__ = [
    # Models
    "Repository",
    "RepositoryConfig",
    "Commit",
    "CommitFile",
    "CommitDetails",
    "FileChange",
    "FileStatus",
    "PullResult",
    "RepositoryInfo",
    "CodeAnalysisResult",
    "DateRangeAnalysis",
    "PullRequest",
    "PullResponse",
    "RepositoryListResponse",
    "DateRangeRequest",
    "CommitImpactRequest",
    # Services
    "GitService",
    "RepositoryScanner",
    "CodeAnalyzer",
]
