"""
Roslyn Service
==============

Service for invoking the .NET Roslyn code analyzer.

This service handles:
- Running the Roslyn analyzer via dotnet command
- Managing temporary output files
- Parsing analysis results
- Error handling for subprocess failures

The Roslyn analyzer extracts:
- Classes, methods, properties, interfaces
- Database operations (SQL commands, stored procedure calls)
- Event handlers
- Code documentation
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any

from git_pipeline.models.pipeline_models import (
    AnalysisResult,
    RoslynAnalysisOutput,
    CodeEntity,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


class RoslynService:
    """
    Service for running Roslyn code analysis on .NET projects.

    The Roslyn analyzer is a .NET application that uses the Roslyn compiler
    platform to extract semantic information from C# code. This service
    invokes the analyzer via subprocess and processes its JSON output.

    Attributes:
        config: Pipeline configuration with paths and timeouts
        analyzer_path: Full path to the Roslyn analyzer DLL
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the Roslyn service.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.analyzer_path = self._resolve_analyzer_path()

        logger.info(f"RoslynService initialized with analyzer: {self.analyzer_path}")

    def _resolve_analyzer_path(self) -> str:
        """
        Resolve the full path to the Roslyn analyzer DLL.

        Returns:
            Absolute path to the analyzer DLL

        Raises:
            FileNotFoundError: If analyzer DLL cannot be found
        """
        # Try relative to project root first
        relative_path = os.path.join(
            self.config.project_root,
            self.config.roslyn_analyzer_path
        )

        if os.path.exists(relative_path):
            return os.path.abspath(relative_path)

        # Try as absolute path
        if os.path.exists(self.config.roslyn_analyzer_path):
            return os.path.abspath(self.config.roslyn_analyzer_path)

        # Log warning but don't raise - analyzer may be built later
        logger.warning(f"Roslyn analyzer not found at: {relative_path}")
        return relative_path

    def is_analyzer_available(self) -> bool:
        """
        Check if the Roslyn analyzer is available.

        Returns:
            True if analyzer DLL exists
        """
        return os.path.exists(self.analyzer_path)

    async def analyze_repository(
        self,
        repo_path: str,
        repo_name: str,
    ) -> AnalysisResult:
        """
        Run Roslyn analysis on a repository.

        This method:
        1. Creates a temporary file for analysis output
        2. Invokes the Roslyn analyzer via dotnet command
        3. Parses the JSON output
        4. Cleans up the temporary file

        Args:
            repo_path: Full path to the git repository
            repo_name: Name identifier for the repository

        Returns:
            AnalysisResult with extracted code entities

        Note:
            Uses asyncio.to_thread for Windows subprocess compatibility.
            The Roslyn analyzer is CPU-bound, so async doesn't help much,
            but it allows other async operations to proceed.
        """
        start_time = time.time()

        logger.info(f"Starting Roslyn analysis for {repo_name} at {repo_path}")

        # Check if analyzer exists
        if not self.is_analyzer_available():
            return AnalysisResult(
                success=False,
                repository=repo_name,
                error=f"Roslyn analyzer not found at: {self.analyzer_path}",
                duration_seconds=time.time() - start_time
            )

        # Create temporary output file
        temp_dir = self.config.temp_dir or tempfile.gettempdir()
        output_file = os.path.join(
            temp_dir,
            f"roslyn-analysis-{repo_name}-{int(time.time() * 1000)}.json"
        )

        try:
            # Run analysis in thread pool (Windows subprocess compatibility)
            result = await asyncio.to_thread(
                self._run_analyzer,
                repo_path,
                output_file
            )

            if not result["success"]:
                return AnalysisResult(
                    success=False,
                    repository=repo_name,
                    error=result.get("error", "Unknown error"),
                    duration_seconds=time.time() - start_time
                )

            # Parse output file
            if os.path.exists(output_file):
                analysis_data = await asyncio.to_thread(
                    self._parse_analysis_output,
                    output_file,
                    repo_name
                )

                return AnalysisResult(
                    success=True,
                    repository=repo_name,
                    output_file=output_file,
                    entity_count=analysis_data.total_entities if analysis_data else 0,
                    file_count=analysis_data.file_count if analysis_data else 0,
                    duration_seconds=time.time() - start_time,
                    analysis_data=analysis_data
                )
            else:
                return AnalysisResult(
                    success=False,
                    repository=repo_name,
                    error="Analysis output file not created",
                    duration_seconds=time.time() - start_time
                )

        except Exception as e:
            logger.error(f"Roslyn analysis failed for {repo_name}: {e}", exc_info=True)
            return AnalysisResult(
                success=False,
                repository=repo_name,
                error=str(e),
                duration_seconds=time.time() - start_time
            )

        finally:
            # Cleanup temp file (optional - may want to keep for debugging)
            # Note: The caller (pipeline) may clean this up after import
            pass

    def _run_analyzer(self, repo_path: str, output_file: str) -> Dict[str, Any]:
        """
        Run the Roslyn analyzer synchronously.

        This method is designed to be called from asyncio.to_thread().

        Args:
            repo_path: Path to repository to analyze
            output_file: Path to write JSON output

        Returns:
            Dict with success status and any error message
        """
        import subprocess

        # Build the dotnet command
        # The analyzer expects: dotnet RoslynCodeAnalyzer.dll <repo_path> <output_file>
        command = [
            "dotnet",
            self.analyzer_path,
            repo_path,
            output_file
        ]

        logger.debug(f"Running Roslyn analyzer: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.analysis_timeout,
                cwd=self.config.project_root,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or f"Exit code: {result.returncode}"
                logger.error(f"Roslyn analyzer failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

            logger.info(f"Roslyn analysis completed successfully")
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Roslyn analysis timed out after {self.config.analysis_timeout}s")
            return {
                "success": False,
                "error": f"Analysis timed out after {self.config.analysis_timeout} seconds"
            }

        except FileNotFoundError:
            logger.error("dotnet command not found - is .NET SDK installed?")
            return {
                "success": False,
                "error": "dotnet command not found. Ensure .NET SDK is installed and in PATH."
            }

        except Exception as e:
            logger.error(f"Failed to run Roslyn analyzer: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _parse_analysis_output(
        self,
        output_file: str,
        repo_name: str
    ) -> Optional[RoslynAnalysisOutput]:
        """
        Parse the JSON output from Roslyn analysis.

        Args:
            output_file: Path to the analysis JSON file
            repo_name: Repository name for the output model

        Returns:
            RoslynAnalysisOutput with parsed entities, or None on error
        """
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse classes
            classes = []
            for cls in data.get("classes", []):
                classes.append(CodeEntity(
                    type="class",
                    name=cls.get("name", ""),
                    full_name=cls.get("fullName", cls.get("name", "")),
                    file_path=cls.get("filePath", ""),
                    line_number=cls.get("lineNumber", 0),
                    namespace=cls.get("namespace"),
                    modifiers=cls.get("modifiers", []),
                    doc_comment=cls.get("documentation")
                ))

            # Parse methods
            methods = []
            for method in data.get("methods", []):
                methods.append(CodeEntity(
                    type="method",
                    name=method.get("name", ""),
                    full_name=method.get("fullName", method.get("name", "")),
                    file_path=method.get("filePath", ""),
                    line_number=method.get("lineNumber", 0),
                    namespace=method.get("namespace"),
                    parent_class=method.get("className"),
                    signature=method.get("signature"),
                    return_type=method.get("returnType"),
                    parameters=method.get("parameters"),
                    modifiers=method.get("modifiers", []),
                    doc_comment=method.get("documentation"),
                    body_preview=method.get("bodyPreview")
                ))

            # Parse properties
            properties = []
            for prop in data.get("properties", []):
                properties.append(CodeEntity(
                    type="property",
                    name=prop.get("name", ""),
                    full_name=prop.get("fullName", prop.get("name", "")),
                    file_path=prop.get("filePath", ""),
                    line_number=prop.get("lineNumber", 0),
                    parent_class=prop.get("className"),
                    return_type=prop.get("type"),
                    modifiers=prop.get("modifiers", []),
                    doc_comment=prop.get("documentation")
                ))

            # Parse interfaces
            interfaces = []
            for iface in data.get("interfaces", []):
                interfaces.append(CodeEntity(
                    type="interface",
                    name=iface.get("name", ""),
                    full_name=iface.get("fullName", iface.get("name", "")),
                    file_path=iface.get("filePath", ""),
                    line_number=iface.get("lineNumber", 0),
                    namespace=iface.get("namespace"),
                    modifiers=iface.get("modifiers", []),
                    doc_comment=iface.get("documentation")
                ))

            # Parse enums
            enums = []
            for enum in data.get("enums", []):
                enums.append(CodeEntity(
                    type="enum",
                    name=enum.get("name", ""),
                    full_name=enum.get("fullName", enum.get("name", "")),
                    file_path=enum.get("filePath", ""),
                    line_number=enum.get("lineNumber", 0),
                    namespace=enum.get("namespace"),
                    modifiers=enum.get("modifiers", []),
                    doc_comment=enum.get("documentation")
                ))

            return RoslynAnalysisOutput(
                repository=repo_name,
                file_count=data.get("fileCount", 0),
                classes=classes,
                methods=methods,
                properties=properties,
                interfaces=interfaces,
                enums=enums,
                database_operations=data.get("databaseOperations", []),
                event_handlers=data.get("eventHandlers", []),
                errors=data.get("errors", [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis output: {e}")
            return None

        except Exception as e:
            logger.error(f"Error parsing analysis output: {e}", exc_info=True)
            return None

    def cleanup_output_file(self, output_file: str) -> bool:
        """
        Remove the temporary analysis output file.

        Args:
            output_file: Path to the output file

        Returns:
            True if cleanup succeeded
        """
        try:
            if output_file and os.path.exists(output_file):
                os.unlink(output_file)
                logger.debug(f"Cleaned up analysis output: {output_file}")
                return True
        except Exception as e:
            logger.warning(f"Failed to cleanup output file: {e}")

        return False

    def analyze_repository_sync(
        self,
        repo_path: str,
        repo_name: str
    ) -> AnalysisResult:
        """
        Run Roslyn analysis synchronously.

        Convenience method for non-async contexts.

        Args:
            repo_path: Full path to the git repository
            repo_name: Name identifier for the repository

        Returns:
            AnalysisResult with extracted code entities
        """
        return asyncio.run(self.analyze_repository(repo_path, repo_name))
