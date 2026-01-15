"""
EWR Code Agent
==============

Specialized agent for code analysis, generation, review, and project scanning.

Capabilities:
- CODE_ANALYZE: Analyze code structure, complexity, and patterns
- CODE_GENERATE: Generate code from natural language descriptions
- CODE_REVIEW: Review code for issues, style, and best practices
- CODE_REFACTOR: Suggest and apply refactoring improvements
- CODE_EXPLAIN: Explain what code does in natural language
- PROJECT_SCAN: Scan project structure and build dependency graphs

Usage:
    from ewr_code_agent import CodeAgent

    agent = CodeAgent(name="code-agent")
    await agent.start()

    # Analyze a file
    result = await agent.analyze_file("/path/to/file.py")

    # Generate code
    code = await agent.generate_code(
        "Create a function to validate email addresses",
        language="python"
    )
"""

from .agent import CodeAgent
from .models import (
    FileInfo,
    ProjectStructure,
    CodeAnalysis,
    SearchResult,
    CodeGenerationRequest,
)

__version__ = "1.0.0"

__all__ = [
    "CodeAgent",
    "FileInfo",
    "ProjectStructure",
    "CodeAnalysis",
    "SearchResult",
    "CodeGenerationRequest",
]
