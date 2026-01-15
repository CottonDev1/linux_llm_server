"""
EWR Code Agent Models
=====================

Pydantic models for code analysis and generation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class FileType(str, Enum):
    """Types of files."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class FileInfo(BaseModel):
    """Information about a single file."""
    path: str
    name: str
    extension: str
    file_type: FileType = FileType.UNKNOWN
    size_bytes: int = 0
    line_count: int = 0
    encoding: str = "utf-8"
    last_modified: Optional[datetime] = None
    is_binary: bool = False
    language: Optional[str] = None


class DirectoryInfo(BaseModel):
    """Information about a directory."""
    path: str
    name: str
    file_count: int = 0
    dir_count: int = 0
    total_size_bytes: int = 0
    files: List[FileInfo] = Field(default_factory=list)
    subdirs: List[str] = Field(default_factory=list)


class ProjectStructure(BaseModel):
    """Complete project structure analysis."""
    root_path: str
    name: str
    total_files: int = 0
    total_dirs: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    languages: Dict[str, int] = Field(default_factory=dict)  # language -> file count
    file_types: Dict[str, int] = Field(default_factory=dict)  # extension -> count
    entry_points: List[str] = Field(default_factory=list)
    config_files: List[str] = Field(default_factory=list)
    test_files: List[str] = Field(default_factory=list)
    source_files: List[FileInfo] = Field(default_factory=list)
    directories: List[DirectoryInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeBlock(BaseModel):
    """A block of code with metadata."""
    content: str
    language: str
    start_line: int = 1
    end_line: int = 1
    file_path: Optional[str] = None
    context: Optional[str] = None  # Surrounding context


class FunctionInfo(BaseModel):
    """Information about a function/method."""
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    docstring: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = Field(default_factory=list)


class ClassInfo(BaseModel):
    """Information about a class."""
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    base_classes: List[str] = Field(default_factory=list)
    methods: List[FunctionInfo] = Field(default_factory=list)
    properties: List[str] = Field(default_factory=list)
    decorators: List[str] = Field(default_factory=list)


class ImportInfo(BaseModel):
    """Information about an import statement."""
    module: str
    names: List[str] = Field(default_factory=list)  # Empty = import whole module
    alias: Optional[str] = None
    is_relative: bool = False
    line_number: int = 0


class CodeAnalysis(BaseModel):
    """Complete analysis of a code file."""
    file_path: str
    language: str
    line_count: int = 0
    char_count: int = 0
    functions: List[FunctionInfo] = Field(default_factory=list)
    classes: List[ClassInfo] = Field(default_factory=list)
    imports: List[ImportInfo] = Field(default_factory=list)
    global_variables: List[str] = Field(default_factory=list)
    complexity_score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from a code search."""
    file_path: str
    line_number: int
    line_content: str
    match_start: int = 0
    match_end: int = 0
    context_before: List[str] = Field(default_factory=list)
    context_after: List[str] = Field(default_factory=list)
    relevance_score: float = 1.0


class CodeGenerationRequest(BaseModel):
    """Request for code generation."""
    prompt: str
    language: str = "python"
    context_files: List[str] = Field(default_factory=list)
    style_guide: Optional[str] = None
    include_tests: bool = False
    include_docs: bool = True
    max_length: int = 500  # lines


class CodeGenerationResult(BaseModel):
    """Result from code generation."""
    code: str
    language: str
    explanation: Optional[str] = None
    tests: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CodeReviewResult(BaseModel):
    """Result from a code review."""
    file_path: str
    overall_score: float = 0.0  # 0-10
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    security_issues: List[Dict[str, Any]] = Field(default_factory=list)
    style_issues: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class RefactoringRequest(BaseModel):
    """Request for code refactoring."""
    file_path: str
    code: str
    refactoring_type: str  # extract_function, rename, simplify, etc.
    target: Optional[str] = None  # What to refactor
    new_name: Optional[str] = None  # For rename operations


class RefactoringResult(BaseModel):
    """Result from refactoring."""
    original_code: str
    refactored_code: str
    changes_description: str
    affected_lines: List[int] = Field(default_factory=list)


# =============================================================================
# Phase 3: SQL Performance Analysis Models
# =============================================================================

class IssueSeverity(str, Enum):
    """Severity levels for performance issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PerformanceIssue(BaseModel):
    """A single SQL performance issue."""
    severity: IssueSeverity = IssueSeverity.WARNING
    issue_type: str  # select_star, missing_where, function_in_where, etc.
    message: str
    location: str = ""  # Which clause or part of the SQL
    suggestion: Optional[str] = None
    line_number: Optional[int] = None


class PerformanceReport(BaseModel):
    """Complete SQL performance analysis report."""
    sql: str
    complexity_score: float = 0.0  # 0-10 scale
    issues: List[PerformanceIssue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    estimated_cost: str = "unknown"  # low, medium, high
    index_recommendations: List[str] = Field(default_factory=list)
    tables_analyzed: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def issue_count(self) -> int:
        """Total number of issues."""
        return len(self.issues)
