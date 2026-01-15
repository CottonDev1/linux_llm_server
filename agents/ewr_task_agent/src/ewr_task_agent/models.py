"""
EWR Task Agent Models
=====================

Pydantic models for shell execution and workflow management.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ShellType(str, Enum):
    """Types of shell environments."""
    BASH = "bash"
    SH = "sh"
    POWERSHELL = "powershell"
    CMD = "cmd"
    ZSH = "zsh"


class ShellCommand(BaseModel):
    """A shell command to execute."""
    command: str
    shell_type: ShellType = ShellType.BASH
    working_dir: Optional[str] = None
    timeout_seconds: int = 300
    environment: Dict[str, str] = Field(default_factory=dict)
    capture_output: bool = True


class ShellResult(BaseModel):
    """Result from shell command execution."""
    command: str
    shell_type: ShellType
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0 and not self.timed_out


class GitFileStatus(str, Enum):
    """Git file status codes."""
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    UNTRACKED = "untracked"
    IGNORED = "ignored"
    STAGED = "staged"


class GitFile(BaseModel):
    """A file in a Git repository."""
    path: str
    status: GitFileStatus
    staged: bool = False
    original_path: Optional[str] = None  # For renamed files


class GitStatus(BaseModel):
    """Git repository status."""
    repo_path: str
    branch: str
    is_clean: bool = True
    ahead: int = 0
    behind: int = 0
    modified_files: List[GitFile] = Field(default_factory=list)
    staged_files: List[GitFile] = Field(default_factory=list)
    untracked_files: List[str] = Field(default_factory=list)
    has_conflicts: bool = False
    conflict_files: List[str] = Field(default_factory=list)


class GitCommit(BaseModel):
    """A Git commit."""
    hash: str
    short_hash: str
    message: str
    author: str
    author_email: str
    date: datetime
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


class GitBranch(BaseModel):
    """A Git branch."""
    name: str
    is_current: bool = False
    is_remote: bool = False
    tracking: Optional[str] = None
    last_commit: Optional[GitCommit] = None


class GitDiff(BaseModel):
    """A Git diff."""
    file_path: str
    old_path: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    is_binary: bool = False
    diff_content: str = ""


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    SHELL = "shell"
    POWERSHELL = "powershell"
    GIT = "git"
    CONDITION = "condition"
    PARALLEL = "parallel"
    AGENT_CALL = "agent_call"


class WorkflowStep(BaseModel):
    """A step in a workflow."""
    id: str
    name: str
    step_type: WorkflowStepType
    command: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None  # Expression to evaluate
    continue_on_error: bool = False
    timeout_seconds: int = 300
    depends_on: List[str] = Field(default_factory=list)


class WorkflowStepResult(BaseModel):
    """Result of a workflow step."""
    step_id: str
    step_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None


class WorkflowResult(BaseModel):
    """Result of a complete workflow execution."""
    workflow_id: str
    workflow_name: str
    success: bool
    step_results: List[WorkflowStepResult] = Field(default_factory=list)
    total_duration_ms: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskDefinition(BaseModel):
    """Definition of a task to execute."""
    task_id: str
    task_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 0
    retry_delay_seconds: int = 5


# =============================================================================
# Phase 3: Schema Change Detection Models
# =============================================================================

class SchemaChangeType(str, Enum):
    """Types of schema changes."""
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"


class SchemaObjectType(str, Enum):
    """Types of schema objects."""
    TABLE = "table"
    COLUMN = "column"
    INDEX = "index"
    PROCEDURE = "procedure"
    VIEW = "view"
    TRIGGER = "trigger"


class SchemaChange(BaseModel):
    """A single detected schema change."""
    file_path: str
    change_type: SchemaChangeType
    object_type: SchemaObjectType
    object_name: str
    details: str = ""
    line_number: Optional[int] = None


class SchemaChangeResult(BaseModel):
    """Result of schema change detection workflow."""
    success: bool
    changed_files: List[str] = Field(default_factory=list)
    changes: List[SchemaChange] = Field(default_factory=list)
    affected_tables: List[str] = Field(default_factory=list)
    cache_invalidated: bool = False
    reindex_triggered: bool = False
    error: Optional[str] = None
    compare_ref: str = "HEAD~1"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def has_changes(self) -> bool:
        """Check if any schema changes were detected."""
        return len(self.changes) > 0

    @property
    def has_destructive_changes(self) -> bool:
        """Check if any DROP operations were detected."""
        return any(c.change_type == SchemaChangeType.DROP for c in self.changes)
