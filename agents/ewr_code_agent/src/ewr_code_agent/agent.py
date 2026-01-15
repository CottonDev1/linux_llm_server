"""
EWR Code Agent
==============

Specialized agent for code analysis, generation, review, and project scanning.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
import aiofiles
import chardet
import fnmatch

from ewr_agent_core import (
    BaseAgent,
    AgentType,
    AgentCapability,
    TaskResult,
    TaskStatus,
    AgentConfig,
)

from .models import (
    FileType,
    FileInfo,
    DirectoryInfo,
    ProjectStructure,
    CodeAnalysis,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    SearchResult,
    CodeGenerationRequest,
    CodeGenerationResult,
    CodeReviewResult,
    # Phase 3: SQL Performance Analysis
    IssueSeverity,
    PerformanceIssue,
    PerformanceReport,
)


# File extension to language mapping
EXTENSION_MAP = {
    ".py": (FileType.PYTHON, "python"),
    ".pyw": (FileType.PYTHON, "python"),
    ".js": (FileType.JAVASCRIPT, "javascript"),
    ".jsx": (FileType.JAVASCRIPT, "javascript"),
    ".ts": (FileType.TYPESCRIPT, "typescript"),
    ".tsx": (FileType.TYPESCRIPT, "typescript"),
    ".cs": (FileType.CSHARP, "csharp"),
    ".java": (FileType.JAVA, "java"),
    ".go": (FileType.GO, "go"),
    ".rs": (FileType.RUST, "rust"),
    ".sql": (FileType.SQL, "sql"),
    ".html": (FileType.HTML, "html"),
    ".htm": (FileType.HTML, "html"),
    ".css": (FileType.CSS, "css"),
    ".scss": (FileType.CSS, "scss"),
    ".json": (FileType.JSON, "json"),
    ".yaml": (FileType.YAML, "yaml"),
    ".yml": (FileType.YAML, "yaml"),
    ".md": (FileType.MARKDOWN, "markdown"),
    ".txt": (FileType.TEXT, "text"),
}

# Default ignore patterns
DEFAULT_IGNORE_PATTERNS = [
    "node_modules",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "*.pyc",
    "*.pyo",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    ".DS_Store",
    "Thumbs.db",
]


class CodeAgent(BaseAgent):
    """
    Code Agent - Specialized for code analysis and generation.

    Capabilities:
    - CODE_ANALYZE: Parse and analyze code structure
    - CODE_GENERATE: Generate code from natural language
    - CODE_REVIEW: Review code for issues and improvements
    - CODE_REFACTOR: Suggest refactoring improvements
    - CODE_EXPLAIN: Explain code in natural language
    - PROJECT_SCAN: Scan and index project structure
    """

    def __init__(
        self,
        config: AgentConfig = None,
        ignore_patterns: List[str] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CODE

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CODE_ANALYZE,
            AgentCapability.CODE_GENERATE,
            AgentCapability.CODE_REVIEW,
            AgentCapability.CODE_REFACTOR,
            AgentCapability.CODE_EXPLAIN,
            AgentCapability.PROJECT_SCAN,
            # Phase 3: SQL Performance Analysis
            AgentCapability.SQL_PERFORMANCE,
        ]

    async def _initialize(self) -> None:
        """Initialize the code agent."""
        self.logger.info("Code agent initialized")

    async def handle_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle incoming tasks."""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("task_type", "")
        params = task.get("params", {})

        try:
            if task_type == "code_analyze" or task_type == AgentCapability.CODE_ANALYZE.value:
                result = await self.analyze_file(params.get("file_path"))
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"analysis": result.model_dump() if result else None}
                )

            elif task_type == "code_generate" or task_type == AgentCapability.CODE_GENERATE.value:
                result = await self.generate_code(
                    prompt=params.get("prompt", ""),
                    language=params.get("language", "python"),
                    context_files=params.get("context_files", [])
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"code": result.code, "explanation": result.explanation}
                )

            elif task_type == "code_explain" or task_type == AgentCapability.CODE_EXPLAIN.value:
                explanation = await self.explain_code(
                    code=params.get("code", ""),
                    language=params.get("language")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"explanation": explanation}
                )

            elif task_type == "project_scan" or task_type == AgentCapability.PROJECT_SCAN.value:
                result = await self.scan_project(params.get("path", "."))
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"structure": result.model_dump() if result else None}
                )

            elif task_type == "code_search":
                results = await self.search_code(
                    pattern=params.get("pattern", ""),
                    path=params.get("path", "."),
                    file_pattern=params.get("file_pattern")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"results": [r.model_dump() for r in results]}
                )

            elif task_type == "read_file":
                content = await self.read_file(params.get("file_path"))
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"content": content}
                )

            # Phase 3: SQL Performance Analysis
            elif task_type == "sql_performance" or task_type == AgentCapability.SQL_PERFORMANCE.value:
                report = await self.analyze_sql_performance(
                    sql=params.get("sql", ""),
                    schema=params.get("schema"),
                    large_tables=params.get("large_tables", [])
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"report": report.model_dump()}
                )

            else:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task_type}"
                )

        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )

    # =========================================================================
    # File Operations
    # =========================================================================

    async def read_file(self, file_path: str) -> str:
        """
        Read a file with automatic encoding detection.

        Args:
            file_path: Path to the file

        Returns:
            File contents as string
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read raw bytes
        async with aiofiles.open(path, "rb") as f:
            raw_content = await f.read()

        # Detect encoding
        detected = chardet.detect(raw_content)
        encoding = detected.get("encoding", "utf-8") or "utf-8"

        try:
            return raw_content.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            return raw_content.decode("utf-8", errors="replace")

    async def write_file(
        self,
        file_path: str,
        content: str,
        backup: bool = True
    ) -> bool:
        """
        Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
            backup: Create backup before writing

        Returns:
            True if successful
        """
        path = Path(file_path)

        # Create backup
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            async with aiofiles.open(path, "r") as f:
                original = await f.read()
            async with aiofiles.open(backup_path, "w") as f:
                await f.write(original)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        async with aiofiles.open(path, "w") as f:
            await f.write(content)

        return True

    async def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            FileInfo with file metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = path.stat()
        extension = path.suffix.lower()

        # Determine file type and language
        file_type, language = EXTENSION_MAP.get(
            extension,
            (FileType.UNKNOWN, None)
        )

        # Check if binary
        is_binary = False
        try:
            async with aiofiles.open(path, "rb") as f:
                chunk = await f.read(8192)
            if b"\x00" in chunk:
                is_binary = True
                file_type = FileType.BINARY
        except Exception:
            pass

        # Count lines
        line_count = 0
        if not is_binary:
            try:
                content = await self.read_file(file_path)
                line_count = content.count("\n") + 1
            except Exception:
                pass

        return FileInfo(
            path=str(path.absolute()),
            name=path.name,
            extension=extension,
            file_type=file_type,
            size_bytes=stat.st_size,
            line_count=line_count,
            is_binary=is_binary,
            language=language,
        )

    # =========================================================================
    # Project Scanning
    # =========================================================================

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        name = path.name
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(str(path), pattern):
                return True
        return False

    async def scan_project(
        self,
        root_path: str,
        max_depth: int = 10,
        max_files: int = 10000
    ) -> ProjectStructure:
        """
        Scan a project directory and build structure analysis.

        Args:
            root_path: Root directory to scan
            max_depth: Maximum directory depth
            max_files: Maximum files to process

        Returns:
            ProjectStructure with complete analysis
        """
        root = Path(root_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root_path}")

        structure = ProjectStructure(
            root_path=str(root),
            name=root.name,
        )

        files_processed = 0
        languages: Dict[str, int] = {}
        file_types: Dict[str, int] = {}

        async def scan_dir(dir_path: Path, depth: int = 0) -> DirectoryInfo:
            nonlocal files_processed

            if depth > max_depth:
                return None
            if self._should_ignore(dir_path):
                return None

            dir_info = DirectoryInfo(
                path=str(dir_path),
                name=dir_path.name
            )

            try:
                entries = list(dir_path.iterdir())
            except PermissionError:
                return dir_info

            for entry in entries:
                if files_processed >= max_files:
                    break

                if self._should_ignore(entry):
                    continue

                if entry.is_file():
                    files_processed += 1
                    try:
                        file_info = await self.get_file_info(str(entry))
                        dir_info.files.append(file_info)
                        dir_info.file_count += 1
                        dir_info.total_size_bytes += file_info.size_bytes

                        # Track languages and types
                        if file_info.language:
                            languages[file_info.language] = languages.get(file_info.language, 0) + 1
                        ext = file_info.extension or "none"
                        file_types[ext] = file_types.get(ext, 0) + 1

                        structure.source_files.append(file_info)
                        structure.total_lines += file_info.line_count

                        # Identify special files
                        name_lower = entry.name.lower()
                        if name_lower in ("main.py", "app.py", "index.js", "main.go", "program.cs"):
                            structure.entry_points.append(str(entry))
                        elif name_lower in ("package.json", "requirements.txt", "setup.py", "pyproject.toml", "cargo.toml"):
                            structure.config_files.append(str(entry))
                        elif "test" in name_lower or name_lower.startswith("test_"):
                            structure.test_files.append(str(entry))

                    except Exception as e:
                        self.logger.warning(f"Error processing {entry}: {e}")

                elif entry.is_dir():
                    subdir_info = await scan_dir(entry, depth + 1)
                    if subdir_info:
                        dir_info.subdirs.append(str(entry))
                        dir_info.dir_count += 1
                        structure.directories.append(subdir_info)

            return dir_info

        await scan_dir(root)

        structure.total_files = files_processed
        structure.total_dirs = len(structure.directories)
        structure.total_size_bytes = sum(f.size_bytes for f in structure.source_files)
        structure.languages = languages
        structure.file_types = file_types

        return structure

    # =========================================================================
    # Code Search
    # =========================================================================

    async def search_code(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = None,
        max_results: int = 100,
        context_lines: int = 2,
        regex: bool = False
    ) -> List[SearchResult]:
        """
        Search for code patterns in files.

        Args:
            pattern: Pattern to search for
            path: Directory or file to search in
            file_pattern: Glob pattern for files (e.g., "*.py")
            max_results: Maximum results to return
            context_lines: Lines of context around matches
            regex: Treat pattern as regex

        Returns:
            List of SearchResult
        """
        results = []
        search_path = Path(path)

        if regex:
            pattern_re = re.compile(pattern, re.IGNORECASE)
        else:
            pattern_lower = pattern.lower()

        async def search_file(file_path: Path):
            if len(results) >= max_results:
                return

            try:
                content = await self.read_file(str(file_path))
                lines = content.split("\n")

                for i, line in enumerate(lines):
                    if len(results) >= max_results:
                        break

                    # Check for match
                    if regex:
                        match = pattern_re.search(line)
                        if not match:
                            continue
                        match_start = match.start()
                        match_end = match.end()
                    else:
                        line_lower = line.lower()
                        idx = line_lower.find(pattern_lower)
                        if idx == -1:
                            continue
                        match_start = idx
                        match_end = idx + len(pattern)

                    # Get context
                    start_ctx = max(0, i - context_lines)
                    end_ctx = min(len(lines), i + context_lines + 1)

                    results.append(SearchResult(
                        file_path=str(file_path),
                        line_number=i + 1,
                        line_content=line,
                        match_start=match_start,
                        match_end=match_end,
                        context_before=lines[start_ctx:i],
                        context_after=lines[i + 1:end_ctx],
                    ))

            except Exception as e:
                self.logger.warning(f"Error searching {file_path}: {e}")

        if search_path.is_file():
            await search_file(search_path)
        else:
            # Walk directory
            for root, dirs, files in os.walk(search_path):
                # Filter ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]

                for file in files:
                    file_path = Path(root) / file
                    if self._should_ignore(file_path):
                        continue
                    if file_pattern and not fnmatch.fnmatch(file, file_pattern):
                        continue

                    await search_file(file_path)

                    if len(results) >= max_results:
                        break

        return results

    # =========================================================================
    # Code Analysis
    # =========================================================================

    async def analyze_file(self, file_path: str) -> CodeAnalysis:
        """
        Analyze a source code file.

        Args:
            file_path: Path to the file

        Returns:
            CodeAnalysis with structure and metrics
        """
        file_info = await self.get_file_info(file_path)
        content = await self.read_file(file_path)

        analysis = CodeAnalysis(
            file_path=file_path,
            language=file_info.language or "unknown",
            line_count=file_info.line_count,
            char_count=len(content),
        )

        # Language-specific parsing
        if file_info.language == "python":
            await self._analyze_python(content, analysis)
        elif file_info.language in ("javascript", "typescript"):
            await self._analyze_javascript(content, analysis)
        elif file_info.language == "csharp":
            await self._analyze_csharp(content, analysis)

        # Use LLM for summary if available
        try:
            summary_prompt = f"""Analyze this {file_info.language} code and provide a brief summary (2-3 sentences):

```{file_info.language}
{content[:3000]}
```

Summary:"""

            response = await self.llm.generate(
                prompt=summary_prompt,
                system="You are a code analysis expert. Provide concise, accurate summaries.",
                max_tokens=200
            )
            analysis.summary = response.content.strip()

        except Exception as e:
            self.logger.warning(f"LLM summary failed: {e}")

        return analysis

    async def _analyze_python(self, content: str, analysis: CodeAnalysis) -> None:
        """Parse Python code and extract structure."""
        lines = content.split("\n")

        # Simple regex-based parsing
        function_re = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?:")
        class_re = re.compile(r"^class\s+(\w+)(?:\((.*?)\))?:")
        import_re = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)")

        current_class = None
        indent_stack = []

        for i, line in enumerate(lines):
            # Check imports
            import_match = import_re.match(line)
            if import_match:
                module = import_match.group(1) or import_match.group(2).split(",")[0].strip()
                names = []
                if import_match.group(1):  # from X import Y
                    names = [n.strip() for n in import_match.group(2).split(",")]
                analysis.imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_relative=module.startswith(".") if import_match.group(1) else False,
                    line_number=i + 1
                ))

            # Check classes
            class_match = class_re.match(line)
            if class_match:
                current_class = ClassInfo(
                    name=class_match.group(1),
                    file_path=analysis.file_path,
                    start_line=i + 1,
                    end_line=i + 1,
                    base_classes=class_match.group(2).split(",") if class_match.group(2) else []
                )
                analysis.classes.append(current_class)

            # Check functions
            func_match = function_re.match(line)
            if func_match:
                indent = len(func_match.group(1))
                is_async = func_match.group(2) is not None
                name = func_match.group(3)
                params = func_match.group(4)
                return_type = func_match.group(5)

                func_info = FunctionInfo(
                    name=name,
                    file_path=analysis.file_path,
                    start_line=i + 1,
                    end_line=i + 1,
                    signature=f"{'async ' if is_async else ''}def {name}({params})",
                    parameters=[p.strip().split(":")[0].split("=")[0].strip()
                               for p in params.split(",") if p.strip()],
                    return_type=return_type.strip() if return_type else None,
                    is_async=is_async,
                    is_method=indent > 0,
                    class_name=current_class.name if current_class and indent > 0 else None
                )

                if current_class and indent > 0:
                    current_class.methods.append(func_info)
                else:
                    analysis.functions.append(func_info)

        # Calculate complexity (simple metric based on structure)
        analysis.complexity_score = (
            len(analysis.functions) * 1.0 +
            len(analysis.classes) * 2.0 +
            sum(len(c.methods) for c in analysis.classes) * 0.5
        )

    async def _analyze_javascript(self, content: str, analysis: CodeAnalysis) -> None:
        """Parse JavaScript/TypeScript code."""
        # Basic function detection
        func_re = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)|"
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\((.*?)\)\s*=>"
        )

        for i, line in enumerate(content.split("\n")):
            match = func_re.search(line)
            if match:
                name = match.group(1) or match.group(3)
                params = match.group(2) or match.group(4) or ""
                if name:
                    analysis.functions.append(FunctionInfo(
                        name=name,
                        file_path=analysis.file_path,
                        start_line=i + 1,
                        end_line=i + 1,
                        signature=f"function {name}({params})",
                        is_async="async" in line,
                    ))

    async def _analyze_csharp(self, content: str, analysis: CodeAnalysis) -> None:
        """Parse C# code."""
        # Basic method detection
        method_re = re.compile(
            r"(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(\w+)\s+(\w+)\s*\((.*?)\)"
        )
        class_re = re.compile(
            r"(?:public|private|protected|internal)?\s*(?:partial\s+)?class\s+(\w+)"
        )

        for i, line in enumerate(content.split("\n")):
            class_match = class_re.search(line)
            if class_match:
                analysis.classes.append(ClassInfo(
                    name=class_match.group(1),
                    file_path=analysis.file_path,
                    start_line=i + 1,
                    end_line=i + 1,
                ))

            method_match = method_re.search(line)
            if method_match:
                analysis.functions.append(FunctionInfo(
                    name=method_match.group(2),
                    file_path=analysis.file_path,
                    start_line=i + 1,
                    end_line=i + 1,
                    signature=f"{method_match.group(1)} {method_match.group(2)}({method_match.group(3)})",
                    return_type=method_match.group(1),
                    is_async="async" in line,
                ))

    # =========================================================================
    # Code Generation
    # =========================================================================

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context_files: List[str] = None,
        style_guide: str = None,
    ) -> CodeGenerationResult:
        """
        Generate code from a natural language prompt.

        Args:
            prompt: Description of what code to generate
            language: Target programming language
            context_files: Optional files to provide as context
            style_guide: Optional style guidelines

        Returns:
            CodeGenerationResult with generated code
        """
        # Build context from files
        context = ""
        if context_files:
            for file_path in context_files[:5]:  # Limit context files
                try:
                    content = await self.read_file(file_path)
                    context += f"\n--- {file_path} ---\n{content[:2000]}\n"
                except Exception as e:
                    self.logger.warning(f"Could not read context file {file_path}: {e}")

        system_prompt = f"""You are an expert {language} developer. Generate clean, well-documented code.
Follow best practices and coding standards for {language}.
{f'Style guide: {style_guide}' if style_guide else ''}
Only output the code, no explanations unless asked."""

        user_prompt = f"""Generate {language} code for:

{prompt}

{f'Context from existing code:{context}' if context else ''}

Output only the code:"""

        response = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=2000
        )

        # Extract code from response
        code = response.content
        # Try to extract code block if present
        code_block_re = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
        match = code_block_re.search(code)
        if match:
            code = match.group(1)

        return CodeGenerationResult(
            code=code.strip(),
            language=language,
        )

    async def explain_code(
        self,
        code: str,
        language: str = None,
        detail_level: str = "medium"
    ) -> str:
        """
        Explain what code does in natural language.

        Args:
            code: Code to explain
            language: Programming language (auto-detected if not provided)
            detail_level: low, medium, or high

        Returns:
            Natural language explanation
        """
        detail_instructions = {
            "low": "Give a brief, one-paragraph summary.",
            "medium": "Explain the main logic and key functions.",
            "high": "Provide a detailed line-by-line explanation."
        }

        prompt = f"""Explain this {language or 'code'}:

```
{code}
```

{detail_instructions.get(detail_level, detail_instructions['medium'])}"""

        response = await self.llm.generate(
            prompt=prompt,
            system="You are a code explanation expert. Explain code clearly and accurately.",
            max_tokens=1000
        )

        return response.content.strip()

    async def review_code(
        self,
        file_path: str = None,
        code: str = None,
        language: str = None,
    ) -> CodeReviewResult:
        """
        Review code for issues and improvements.

        Args:
            file_path: Path to file to review
            code: Or provide code directly
            language: Programming language

        Returns:
            CodeReviewResult with findings
        """
        if file_path:
            code = await self.read_file(file_path)
            file_info = await self.get_file_info(file_path)
            language = language or file_info.language

        prompt = f"""Review this {language or 'code'} for:
1. Bugs and logic errors
2. Security issues
3. Performance problems
4. Code style and readability
5. Best practices

```
{code}
```

Provide your review in this format:
SCORE: (0-10)
ISSUES:
- issue 1
- issue 2
SECURITY:
- security concern 1
SUGGESTIONS:
- suggestion 1
SUMMARY: brief overall assessment"""

        response = await self.llm.generate(
            prompt=prompt,
            system="You are a senior code reviewer. Be thorough but constructive.",
            max_tokens=1500
        )

        # Parse response (simple parsing)
        content = response.content
        result = CodeReviewResult(
            file_path=file_path or "inline",
            summary=content
        )

        # Try to extract score
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", content)
        if score_match:
            result.overall_score = float(score_match.group(1))

        return result

    # =========================================================================
    # Phase 3: SQL Performance Analysis
    # =========================================================================

    async def analyze_sql_performance(
        self,
        sql: str,
        schema: Dict[str, Any] = None,
        large_tables: List[str] = None
    ) -> PerformanceReport:
        """
        Analyze SQL query for performance issues.

        Args:
            sql: SQL query to analyze
            schema: Optional schema information
            large_tables: List of table names considered "large"

        Returns:
            PerformanceReport with issues and recommendations
        """
        schema = schema or {}
        large_tables = large_tables or schema.get("large_tables", [])

        report = PerformanceReport(sql=sql)
        sql_upper = sql.upper()

        # Extract tables for analysis
        report.tables_analyzed = self._extract_sql_tables(sql)

        # 1. Check for SELECT *
        if re.search(r'\bSELECT\s+\*', sql, re.IGNORECASE):
            report.issues.append(PerformanceIssue(
                severity=IssueSeverity.WARNING,
                issue_type="select_star",
                message="SELECT * retrieves all columns - consider selecting only needed columns",
                location="SELECT clause",
                suggestion="Replace SELECT * with explicit column list for better performance"
            ))

        # 2. Check for missing WHERE clause on large tables
        for table in report.tables_analyzed:
            if table.lower() in [t.lower() for t in large_tables]:
                if not self._has_where_for_table(sql, table):
                    report.issues.append(PerformanceIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type="missing_where",
                        message=f"No WHERE clause for large table '{table}' may cause full table scan",
                        location=f"FROM {table}",
                        suggestion=f"Add filtering conditions for table '{table}'"
                    ))

        # 3. Check for function calls in WHERE clause
        function_where = re.search(
            r'WHERE\s+.*?\b(\w+)\s*\([^)]+\)\s*[=<>]',
            sql, re.IGNORECASE | re.DOTALL
        )
        if function_where:
            report.issues.append(PerformanceIssue(
                severity=IssueSeverity.WARNING,
                issue_type="function_in_where",
                message="Function call in WHERE clause may prevent index usage",
                location="WHERE clause",
                suggestion="Move function to the comparison value side, or create a computed column"
            ))

        # 4. Check for date function anti-patterns
        date_funcs = re.findall(
            r'\b(YEAR|MONTH|DAY|DATEPART)\s*\(\s*[^)]+\)\s*=',
            sql, re.IGNORECASE
        )
        if date_funcs:
            report.issues.append(PerformanceIssue(
                severity=IssueSeverity.WARNING,
                issue_type="date_function_in_where",
                message=f"Date function(s) {list(set(date_funcs))} in WHERE prevent index usage",
                location="WHERE clause",
                suggestion="Use date range comparisons: WHERE column >= 'start' AND column < 'end'"
            ))

        # 5. Check for missing row limit
        if "SELECT" in sql_upper and "TOP " not in sql_upper and "LIMIT " not in sql_upper:
            if "COUNT(" not in sql_upper and "SUM(" not in sql_upper and "GROUP BY" not in sql_upper:
                report.issues.append(PerformanceIssue(
                    severity=IssueSeverity.INFO,
                    issue_type="no_row_limit",
                    message="Query has no row limit (TOP/LIMIT) - may return excessive results",
                    location="SELECT clause",
                    suggestion="Add TOP N or LIMIT to restrict result set size"
                ))

        # 6. Check for CURSOR usage
        if "DECLARE" in sql_upper and "CURSOR" in sql_upper:
            report.issues.append(PerformanceIssue(
                severity=IssueSeverity.WARNING,
                issue_type="cursor_usage",
                message="CURSOR usage detected - row-by-row processing is slower than set-based",
                location="DECLARE CURSOR",
                suggestion="Replace cursor with set-based UPDATE, INSERT, or DELETE operations"
            ))

        # 7. Check for NOLOCK hint
        if "NOLOCK" in sql_upper or "READUNCOMMITTED" in sql_upper:
            report.issues.append(PerformanceIssue(
                severity=IssueSeverity.INFO,
                issue_type="nolock_hint",
                message="NOLOCK hint may cause dirty reads, phantom reads, or missing rows",
                location="Table hint",
                suggestion="Consider if dirty reads are acceptable for this query"
            ))

        # Calculate complexity score
        report.complexity_score = self._calculate_sql_complexity(sql)

        # Estimate cost
        report.estimated_cost = self._estimate_query_cost(
            sql, report.complexity_score, report.tables_analyzed, large_tables
        )

        # Generate index recommendations
        report.index_recommendations = self._generate_index_recommendations(sql)

        # Generate general suggestions
        if report.issues:
            report.suggestions = [
                f"Address {len(report.issues)} performance issue(s) identified",
            ]
            if report.estimated_cost == "high":
                report.suggestions.append("Consider query optimization before production use")

        return report

    def _extract_sql_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = set()

        # Match FROM/JOIN clauses
        patterns = [
            r"FROM\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?(?:\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)?)",
            r"JOIN\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?(?:\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)?)",
            r"INTO\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?(?:\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)?)",
            r"UPDATE\s+(\[?[a-zA-Z_][a-zA-Z0-9_]*\]?(?:\.\[?[a-zA-Z_][a-zA-Z0-9_]*\]?)?)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            # Remove brackets and add to set
            tables.update(m.replace("[", "").replace("]", "") for m in matches)

        return list(tables)

    def _has_where_for_table(self, sql: str, table: str) -> bool:
        """Check if SQL has a WHERE clause filtering on the specified table."""
        sql_upper = sql.upper()
        table_upper = table.upper()

        # Check if WHERE exists
        if "WHERE" not in sql_upper:
            return False

        # Count tables in the query
        tables_in_query = self._extract_sql_tables(sql)

        # If this is a single-table query and has WHERE, assume filtered
        if len(tables_in_query) == 1 and table_upper in [t.upper() for t in tables_in_query]:
            return True

        # For multi-table queries, check if table alias or name appears in WHERE
        # Extract potential alias
        alias_match = re.search(
            rf"(?:FROM|JOIN)\s+\[?{re.escape(table)}\]?\s+(?:AS\s+)?(\w+)",
            sql, re.IGNORECASE
        )
        if alias_match:
            alias = alias_match.group(1).upper()
            # Check if alias is used in WHERE
            where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|$)', sql, re.IGNORECASE | re.DOTALL)
            if where_match and alias in where_match.group(1).upper():
                return True

        # Check if table name itself appears in WHERE
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match and table_upper in where_match.group(1).upper():
            return True

        return False

    def _calculate_sql_complexity(self, sql: str) -> float:
        """
        Calculate SQL complexity score (0-10).

        Scoring:
        - SELECT: +1.0
        - WHERE: +0.5
        - ORDER BY: +0.5
        - Each JOIN: +1.0
        - GROUP BY: +1.0
        - HAVING: +0.5
        - Each subquery: +1.5
        - WITH...AS (CTE): +1.5
        - OVER() (Window): +1.0
        """
        score = 0.0
        sql_upper = sql.upper()

        if "SELECT" in sql_upper:
            score += 1.0
        if "WHERE" in sql_upper:
            score += 0.5
        if "ORDER BY" in sql_upper:
            score += 0.5

        # Count JOINs
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        score += join_count * 1.0

        if "GROUP BY" in sql_upper:
            score += 1.0
        if "HAVING" in sql_upper:
            score += 0.5

        # Count subqueries (SELECT inside parens)
        subquery_count = len(re.findall(r'\(\s*SELECT\b', sql, re.IGNORECASE))
        score += subquery_count * 1.5

        # CTE (WITH...AS)
        if re.search(r'\bWITH\b.*?\bAS\s*\(', sql, re.IGNORECASE | re.DOTALL):
            score += 1.5

        # Window functions
        if "OVER(" in sql_upper or "OVER (" in sql_upper:
            score += 1.0

        # Cap at 10
        return min(score, 10.0)

    def _estimate_query_cost(
        self,
        sql: str,
        complexity: float,
        tables: List[str],
        large_tables: List[str]
    ) -> str:
        """Estimate query cost based on complexity and tables."""
        sql_upper = sql.upper()

        # Large table without filter = high
        for table in tables:
            if table.lower() in [t.lower() for t in large_tables]:
                if "WHERE" not in sql_upper:
                    return "high"

        # High complexity = high
        if complexity > 7:
            return "high"

        # Medium complexity or large table with filter
        if complexity > 4:
            return "medium"

        for table in tables:
            if table.lower() in [t.lower() for t in large_tables]:
                return "medium"

        return "low"

    def _generate_index_recommendations(self, sql: str) -> List[str]:
        """Generate index recommendations based on SQL patterns."""
        recommendations = []

        # Find columns in WHERE with equality
        equality_cols = re.findall(
            r'WHERE\s+.*?(\w+)\s*=\s*[^=]',
            sql, re.IGNORECASE | re.DOTALL
        )
        for col in equality_cols:
            if col.upper() not in ("AND", "OR", "NOT"):
                recommendations.append(f"Consider index on column: {col}")

        # Find columns in WHERE with range
        range_cols = re.findall(
            r'(\w+)\s*(?:>=|<=|>|<|BETWEEN)\s*',
            sql, re.IGNORECASE
        )
        for col in range_cols:
            if col.upper() not in ("AND", "OR", "NOT", "BETWEEN"):
                recommendations.append(f"Consider index on column: {col} (range query)")

        # Find columns in ORDER BY
        order_match = re.search(
            r'ORDER BY\s+([\w\s,]+?)(?:ASC|DESC|$)',
            sql, re.IGNORECASE
        )
        if order_match:
            order_cols = [c.strip() for c in order_match.group(1).split(",")]
            for col in order_cols[:2]:  # First 2 columns
                if col:
                    recommendations.append(f"Consider including {col} in covering index")

        # Deduplicate
        return list(dict.fromkeys(recommendations))[:5]  # Max 5 recommendations
