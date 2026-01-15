"""
Code Analyzer

Extracts classes, methods, and functions from C#, JavaScript, TypeScript, and SQL files.
Migrated from JavaScript GitService.js extraction methods.
"""

import re
import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import List, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .models import CodeAnalysisResult

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Analyzes code files to extract structural information.

    Supports:
    - C# classes and methods
    - JavaScript/TypeScript classes and functions
    - SQL stored procedures and functions
    """

    # C# keywords that shouldn't be matched as class or method names
    CSHARP_KEYWORDS = frozenset([
        'if', 'while', 'for', 'foreach', 'switch', 'catch', 'using',
        'lock', 'fixed', 'checked', 'unchecked', 'try', 'finally',
        'return', 'throw', 'new', 'typeof', 'sizeof', 'default',
        'delegate', 'event', 'operator', 'implicit', 'explicit',
        'class', 'interface', 'struct', 'enum', 'namespace', 'void',
        'int', 'string', 'bool', 'double', 'float', 'decimal', 'long',
        'short', 'byte', 'char', 'object', 'var', 'dynamic', 'async',
        'await', 'get', 'set', 'value', 'add', 'remove', 'partial',
        'where', 'select', 'from', 'join', 'on', 'equals', 'into',
        'orderby', 'ascending', 'descending', 'group', 'by', 'let'
    ])

    # JavaScript keywords that shouldn't be matched
    JS_KEYWORDS = frozenset([
        'if', 'while', 'for', 'switch', 'catch', 'try', 'finally',
        'return', 'throw', 'new', 'typeof', 'delete', 'void',
        'instanceof', 'in', 'of', 'with', 'debugger'
    ])

    # Regex patterns compiled for performance
    CSHARP_CLASS_PATTERN = re.compile(
        r'(?:public|private|protected|internal)?\s*'
        r'(?:static|abstract|sealed)?\s*'
        r'(?:partial)?\s*'
        r'class\s+(\w+)',
        re.MULTILINE
    )

    CSHARP_INTERFACE_PATTERN = re.compile(
        r'(?:public|private|protected|internal)?\s*'
        r'interface\s+(I\w+)',
        re.MULTILINE
    )

    CSHARP_METHOD_PATTERN = re.compile(
        r'(?:public|private|protected|internal)?\s*'
        r'(?:static|virtual|override|async|abstract|sealed)?\s*'
        r'(?:\w+(?:<[\w,\s<>]+>)?)\s+'  # Return type with optional generics
        r'(\w+)\s*\([^)]*\)',
        re.MULTILINE
    )

    CSHARP_PROPERTY_PATTERN = re.compile(
        r'(?:public|private|protected|internal)?\s*'
        r'(?:static|virtual|override|abstract)?\s*'
        r'(\w+(?:<[\w,\s<>]+>)?)\s+'  # Type
        r'(\w+)\s*'
        r'(?:\{|=>)',  # Property accessor or expression body
        re.MULTILINE
    )

    JS_CLASS_PATTERN = re.compile(
        r'(?:export\s+)?(?:default\s+)?class\s+(\w+)',
        re.MULTILINE
    )

    JS_FUNCTION_PATTERN = re.compile(
        r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\('
        r'|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
        re.MULTILINE
    )

    JS_METHOD_PATTERN = re.compile(
        r'^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )

    SQL_PROCEDURE_PATTERN = re.compile(
        r'CREATE\s+(?:OR\s+ALTER\s+)?PROCEDURE\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?',
        re.IGNORECASE | re.MULTILINE
    )

    SQL_FUNCTION_PATTERN = re.compile(
        r'CREATE\s+(?:OR\s+ALTER\s+)?FUNCTION\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?',
        re.IGNORECASE | re.MULTILINE
    )

    SQL_VIEW_PATTERN = re.compile(
        r'CREATE\s+(?:OR\s+ALTER\s+)?VIEW\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?',
        re.IGNORECASE | re.MULTILINE
    )

    SQL_TRIGGER_PATTERN = re.compile(
        r'CREATE\s+(?:OR\s+ALTER\s+)?TRIGGER\s+(?:\[?dbo\]?\.)?\[?(\w+)\]?',
        re.IGNORECASE | re.MULTILINE
    )

    def __init__(self, max_workers: int = 5):
        """
        Initialize the code analyzer.

        Args:
            max_workers: Maximum parallel file processing workers
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # =========================================================================
    # C# Extraction
    # =========================================================================

    def extract_csharp_classes(self, content: str) -> List[str]:
        """
        Extract C# class names from code content.

        Args:
            content: File content

        Returns:
            List of class names
        """
        classes = []

        # Extract classes
        for match in self.CSHARP_CLASS_PATTERN.finditer(content):
            class_name = match.group(1)
            if class_name not in self.CSHARP_KEYWORDS:
                classes.append(class_name)

        # Extract interfaces
        for match in self.CSHARP_INTERFACE_PATTERN.finditer(content):
            interface_name = match.group(1)
            if interface_name not in self.CSHARP_KEYWORDS:
                classes.append(interface_name)

        return classes

    def extract_csharp_methods(self, content: str) -> List[str]:
        """
        Extract C# method names from code content.

        Args:
            content: File content

        Returns:
            List of unique method names
        """
        methods: Set[str] = set()

        for match in self.CSHARP_METHOD_PATTERN.finditer(content):
            method_name = match.group(1)
            if method_name and method_name not in self.CSHARP_KEYWORDS:
                methods.add(method_name)

        return list(methods)

    def extract_csharp_properties(self, content: str) -> List[str]:
        """
        Extract C# property names from code content.

        Args:
            content: File content

        Returns:
            List of unique property names
        """
        properties: Set[str] = set()

        for match in self.CSHARP_PROPERTY_PATTERN.finditer(content):
            prop_name = match.group(2)
            if prop_name and prop_name not in self.CSHARP_KEYWORDS:
                properties.add(prop_name)

        return list(properties)

    # =========================================================================
    # JavaScript/TypeScript Extraction
    # =========================================================================

    def extract_js_classes(self, content: str) -> List[str]:
        """
        Extract JavaScript/TypeScript class names from code content.

        Args:
            content: File content

        Returns:
            List of class names
        """
        classes = []

        for match in self.JS_CLASS_PATTERN.finditer(content):
            classes.append(match.group(1))

        return classes

    def extract_js_functions(self, content: str) -> List[str]:
        """
        Extract JavaScript/TypeScript function names from code content.

        Args:
            content: File content

        Returns:
            List of unique function names
        """
        functions: Set[str] = set()

        for match in self.JS_FUNCTION_PATTERN.finditer(content):
            # Match group 1 is function declaration, group 2 is arrow function
            func_name = match.group(1) or match.group(2)
            if func_name and func_name not in self.JS_KEYWORDS:
                functions.add(func_name)

        return list(functions)

    # =========================================================================
    # SQL Extraction
    # =========================================================================

    def extract_sql_procedures(self, content: str) -> List[str]:
        """
        Extract SQL stored procedure names from code content.

        Args:
            content: File content

        Returns:
            List of procedure names
        """
        procedures = []

        for match in self.SQL_PROCEDURE_PATTERN.finditer(content):
            procedures.append(match.group(1))

        return procedures

    def extract_sql_functions(self, content: str) -> List[str]:
        """
        Extract SQL function names from code content.

        Args:
            content: File content

        Returns:
            List of function names
        """
        functions = []

        for match in self.SQL_FUNCTION_PATTERN.finditer(content):
            functions.append(match.group(1))

        return functions

    def extract_sql_objects(self, content: str) -> Tuple[List[str], List[str]]:
        """
        Extract all SQL objects (procedures, functions, views, triggers).

        Args:
            content: File content

        Returns:
            Tuple of (object_names, object_types)
        """
        objects = []
        types = []

        for match in self.SQL_PROCEDURE_PATTERN.finditer(content):
            objects.append(match.group(1))
            types.append('procedure')

        for match in self.SQL_FUNCTION_PATTERN.finditer(content):
            objects.append(match.group(1))
            types.append('function')

        for match in self.SQL_VIEW_PATTERN.finditer(content):
            objects.append(match.group(1))
            types.append('view')

        for match in self.SQL_TRIGGER_PATTERN.finditer(content):
            objects.append(match.group(1))
            types.append('trigger')

        return objects, types

    # =========================================================================
    # File Analysis
    # =========================================================================

    def analyze_file_sync(self, file_path: str, repo_path: str) -> CodeAnalysisResult:
        """
        Analyze a code file and extract classes/methods synchronously.

        Args:
            file_path: Relative file path within repository
            repo_path: Repository root path

        Returns:
            CodeAnalysisResult with extracted information
        """
        full_path = Path(repo_path) / file_path
        file_ext = full_path.suffix.lower()

        # Check if file exists
        if not full_path.exists():
            return CodeAnalysisResult(
                file=file_path,
                classes=[],
                methods=[],
                success=False,
                error='File not found (possibly deleted)'
            )

        try:
            content = full_path.read_text(encoding='utf-8', errors='replace')

            classes: List[str] = []
            methods: List[str] = []

            # Extract based on file type
            if file_ext == '.cs':
                classes = self.extract_csharp_classes(content)
                methods = self.extract_csharp_methods(content)

            elif file_ext in ('.js', '.ts', '.jsx', '.tsx'):
                classes = self.extract_js_classes(content)
                methods = self.extract_js_functions(content)

            elif file_ext == '.sql':
                # For SQL, "classes" are objects (procs/functions/views)
                # and "methods" stays empty (or could list parameters)
                objects, _ = self.extract_sql_objects(content)
                classes = objects

            return CodeAnalysisResult(
                file=file_path,
                classes=classes,
                methods=methods,
                success=True
            )

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return CodeAnalysisResult(
                file=file_path,
                classes=[],
                methods=[],
                success=False,
                error=str(e)
            )

    async def analyze_file(self, file_path: str, repo_path: str) -> CodeAnalysisResult:
        """
        Analyze a code file asynchronously.

        Args:
            file_path: Relative file path within repository
            repo_path: Repository root path

        Returns:
            CodeAnalysisResult with extracted information
        """
        full_path = Path(repo_path) / file_path
        file_ext = full_path.suffix.lower()

        if not full_path.exists():
            return CodeAnalysisResult(
                file=file_path,
                classes=[],
                methods=[],
                success=False,
                error='File not found (possibly deleted)'
            )

        try:
            async with aiofiles.open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()

            classes: List[str] = []
            methods: List[str] = []

            if file_ext == '.cs':
                classes = self.extract_csharp_classes(content)
                methods = self.extract_csharp_methods(content)

            elif file_ext in ('.js', '.ts', '.jsx', '.tsx'):
                classes = self.extract_js_classes(content)
                methods = self.extract_js_functions(content)

            elif file_ext == '.sql':
                objects, _ = self.extract_sql_objects(content)
                classes = objects

            return CodeAnalysisResult(
                file=file_path,
                classes=classes,
                methods=methods,
                success=True
            )

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return CodeAnalysisResult(
                file=file_path,
                classes=[],
                methods=[],
                success=False,
                error=str(e)
            )

    async def analyze_multiple_files(
        self,
        files: List[str],
        repo_path: str,
        batch_size: int = 5
    ) -> List[CodeAnalysisResult]:
        """
        Analyze multiple files in parallel with batching.

        Args:
            files: List of relative file paths
            repo_path: Repository root path
            batch_size: Number of files to process in parallel

        Returns:
            List of CodeAnalysisResult for all files
        """
        results: List[CodeAnalysisResult] = []

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_tasks = [self.analyze_file(f, repo_path) for f in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        return results

    def analyze_multiple_files_sync(
        self,
        files: List[str],
        repo_path: str
    ) -> List[CodeAnalysisResult]:
        """
        Analyze multiple files synchronously.

        Args:
            files: List of relative file paths
            repo_path: Repository root path

        Returns:
            List of CodeAnalysisResult for all files
        """
        return [self.analyze_file_sync(f, repo_path) for f in files]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_supported_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer supports."""
        return ['.cs', '.js', '.ts', '.jsx', '.tsx', '.sql']

    def is_code_file(self, file_path: str) -> bool:
        """Check if a file is a supported code file."""
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_extensions()

    def filter_code_files(self, files: List[str]) -> List[str]:
        """Filter a list of files to only include supported code files."""
        return [f for f in files if self.is_code_file(f)]
