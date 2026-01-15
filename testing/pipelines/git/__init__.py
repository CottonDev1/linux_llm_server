"""
Git Pipeline Tests
==================

Comprehensive E2E tests for the Git Pipeline.

Test modules:
- test_git_e2e.py - End-to-end git pipeline tests
- test_git_generation.py - Git context generation tests
- test_git_retrieval.py - Git data retrieval tests
- test_git_storage.py - Git storage operations tests
- test_git_service.py - Core GitService tests
- test_git_pipeline.py - Pipeline orchestration tests
- test_roslyn_service.py - Roslyn code analysis tests
- test_code_import_service.py - Vector import service tests
- test_git_routes.py - API endpoint tests

Coverage targets:
- GitService: 80%+
- RoslynService: 80%+
- CodeImportService: 80%+
- Pipeline orchestration: 80%+
- API endpoints: 80%+
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "LLM Website Team"

# Test module paths
TEST_DIR = Path(__file__).parent
TEST_MODULES = [
    "test_git_e2e",
    "test_git_generation",
    "test_git_retrieval",
    "test_git_storage",
    "test_git_service",
    "test_git_pipeline",
    "test_roslyn_service",
    "test_code_import_service",
    "test_git_routes",
]
