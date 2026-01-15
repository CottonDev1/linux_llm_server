"""
Git Pipeline Test Configuration
===============================

Pytest configuration and fixtures specific to git pipeline tests.

This module provides:
- Git-specific test fixtures
- Mock services for testing
- Test data generators
- Shared test utilities

Usage:
    # In test files, fixtures are automatically available:
    def test_something(temp_git_repo, mock_git_service):
        ...
"""

import os
import sys
import tempfile
import subprocess
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Generator, Dict, Any, List

# Ensure python_services is in path
PYTHON_SERVICES = Path(__file__).parent.parent.parent.parent / "python_services"
sys.path.insert(0, str(PYTHON_SERVICES))

# Set environment variables before imports
os.environ.setdefault("GIT_ROOT", r"C:\Projects\Git")
os.environ.setdefault("PROJECT_ROOT", r"C:\Projects\llm_website")


# =============================================================================
# Git Repository Fixtures
# =============================================================================

@pytest.fixture
def temp_git_repo(tmp_path) -> str:
    """
    Create a temporary git repository for testing.

    Creates a fully initialized git repository with:
    - Initial configuration
    - First commit
    - Test file

    Returns:
        Path to the temporary git repository
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )

    # Create initial commit
    test_file = repo_path / "test.txt"
    test_file.write_text("initial content")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )

    return str(repo_path)


@pytest.fixture
def temp_git_repo_with_history(temp_git_repo) -> str:
    """
    Create a temporary git repository with multiple commits.

    Extends temp_git_repo with additional commits for testing
    commit history features.

    Returns:
        Path to the temporary git repository
    """
    repo_path = Path(temp_git_repo)

    # Add multiple commits
    for i in range(5):
        test_file = repo_path / f"file_{i}.txt"
        test_file.write_text(f"content {i}")
        subprocess.run(
            ["git", "add", "."],
            cwd=str(repo_path),
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add file {i}"],
            cwd=str(repo_path),
            capture_output=True,
            check=True
        )

    return str(repo_path)


@pytest.fixture
def temp_git_repo_with_csharp(temp_git_repo) -> str:
    """
    Create a temporary git repository with C# files.

    Creates sample C# files for testing Roslyn analysis.

    Returns:
        Path to the temporary git repository
    """
    repo_path = Path(temp_git_repo)

    # Create src directory
    src_dir = repo_path / "src"
    src_dir.mkdir()

    # Create sample C# files
    csharp_content = '''
using System;

namespace TestApp
{
    public class UserService
    {
        private readonly IUserRepository _repository;

        public UserService(IUserRepository repository)
        {
            _repository = repository;
        }

        /// <summary>
        /// Gets a user by their ID.
        /// </summary>
        public User GetUserById(int userId)
        {
            return _repository.GetById(userId);
        }

        public void SaveUser(User user)
        {
            _repository.Save(user);
        }
    }
}
'''

    cs_file = src_dir / "UserService.cs"
    cs_file.write_text(csharp_content)

    subprocess.run(
        ["git", "add", "."],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Add UserService"],
        cwd=str(repo_path),
        capture_output=True,
        check=True
    )

    return str(repo_path)


# =============================================================================
# Mock Service Fixtures
# =============================================================================

@pytest.fixture
def mock_git_service():
    """
    Create a mock GitService for testing.

    Returns a mock with all common methods pre-configured.
    """
    mock = Mock()

    # Sync methods
    mock.scan_repositories.return_value = []
    mock.get_repository_path.return_value = r"C:\Projects\Git\TestRepo"
    mock.verify_repository.return_value = True
    mock.get_current_head.return_value = "abc123def456789012345678901234567890abcd"
    mock.get_recent_commits.return_value = []
    mock.get_changed_files.return_value = []
    mock.get_changed_files_with_status.return_value = []
    mock.get_last_pull_time.return_value = "2024-01-15T10:00:00Z"

    # Async methods
    mock.pull_repository_async = AsyncMock(return_value=Mock(
        success=True,
        output="Already up to date.",
        is_already_up_to_date=True,
        error=None
    ))
    mock.get_repository_info = AsyncMock()
    mock.get_recent_commits_async = AsyncMock(return_value=[])
    mock.get_changed_files_async = AsyncMock(return_value=[])
    mock.get_current_head_async = AsyncMock(return_value="abc123def456789012345678901234567890abcd")

    return mock


@pytest.fixture
def mock_roslyn_service():
    """
    Create a mock RoslynService for testing.

    Returns a mock with analysis methods pre-configured.
    """
    mock = Mock()

    mock.is_analyzer_available.return_value = True
    mock.analyze_repository = AsyncMock(return_value=Mock(
        success=True,
        repository="TestRepo",
        entity_count=10,
        file_count=5,
        output_file="/tmp/analysis.json",
        duration_seconds=1.5,
        error=None
    ))
    mock.analyze_repository_sync.return_value = Mock(
        success=True,
        repository="TestRepo",
        entity_count=10,
        file_count=5
    )
    mock.cleanup_output_file.return_value = True

    return mock


@pytest.fixture
def mock_code_import_service():
    """
    Create a mock CodeImportService for testing.

    Returns a mock with import methods pre-configured.
    """
    mock = Mock()

    mock.import_analysis = AsyncMock(return_value=Mock(
        success=True,
        documents_imported=10,
        documents_updated=5,
        documents_deleted=0,
        duration_seconds=0.5,
        error=None
    ))
    mock.import_analysis_sync.return_value = Mock(
        success=True,
        documents_imported=10,
        documents_updated=5
    )
    mock.delete_repository_data = AsyncMock(return_value={
        "code_classes": 5,
        "code_methods": 20,
        "code_context": 10,
        "code_dboperations": 3,
        "code_eventhandlers": 2
    })

    return mock


@pytest.fixture
def mock_mongodb_service():
    """
    Create a mock MongoDB service for testing.
    """
    mock_collection = AsyncMock()
    mock_collection.update_one = AsyncMock(return_value=Mock(
        upserted_id="new_id_123",
        modified_count=0
    ))
    mock_collection.delete_many = AsyncMock(return_value=Mock(deleted_count=10))
    mock_collection.find = Mock(return_value=AsyncMock())

    mock_db = Mock()
    mock_db.__getitem__ = Mock(return_value=mock_collection)

    mock_service = Mock()
    mock_service.db = mock_db
    mock_service.is_initialized = True
    mock_service.initialize = AsyncMock()

    return mock_service


@pytest.fixture
def mock_embedding_service():
    """
    Create a mock embedding service for testing.
    """
    mock_service = Mock()
    mock_service.is_initialized = True
    mock_service.initialize = AsyncMock()
    mock_service.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    mock_service.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * 384 for _ in texts]
    )
    return mock_service


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_analysis_output() -> Dict[str, Any]:
    """
    Create sample Roslyn analysis output for testing.
    """
    return {
        "fileCount": 10,
        "classes": [
            {
                "name": "UserService",
                "fullName": "MyApp.Services.UserService",
                "filePath": "/src/Services/UserService.cs",
                "lineNumber": 15,
                "namespace": "MyApp.Services",
                "modifiers": ["public"],
                "documentation": "Service for user operations"
            }
        ],
        "methods": [
            {
                "name": "GetUserById",
                "fullName": "MyApp.Services.UserService.GetUserById",
                "filePath": "/src/Services/UserService.cs",
                "lineNumber": 25,
                "namespace": "MyApp.Services",
                "className": "UserService",
                "signature": "public User GetUserById(int userId)",
                "returnType": "User",
                "parameters": [{"name": "userId", "type": "int"}],
                "modifiers": ["public"],
                "documentation": "Retrieves a user by ID"
            }
        ],
        "properties": [],
        "interfaces": [],
        "enums": [],
        "databaseOperations": [],
        "eventHandlers": [],
        "errors": []
    }


@pytest.fixture
def sample_commit_list() -> List[Dict[str, Any]]:
    """
    Create sample commit list for testing.
    """
    return [
        {
            "hash": "abc123def456789012345678901234567890abcd",
            "author": "Test Author",
            "date": "2024-01-15T10:00:00",
            "message": "Add UserService"
        },
        {
            "hash": "def456abc789012345678901234567890abcdef",
            "author": "Test Author",
            "date": "2024-01-14T09:00:00",
            "message": "Initial commit"
        }
    ]


@pytest.fixture
def sample_file_changes() -> List[Dict[str, Any]]:
    """
    Create sample file changes for testing.
    """
    return [
        {"file": "src/UserService.cs", "status": "modified", "status_code": "M"},
        {"file": "src/NewFile.cs", "status": "added", "status_code": "A"},
        {"file": "src/Deleted.cs", "status": "deleted", "status_code": "D"}
    ]


# =============================================================================
# Pipeline Configuration Fixtures
# =============================================================================

@pytest.fixture
def git_pipeline_config():
    """
    Create a PipelineConfig for git-specific testing.

    Note: This is separate from pipeline_config (PipelineTestConfig) in parent conftest.
    """
    try:
        from git_pipeline.models.pipeline_models import PipelineConfig
        return PipelineConfig(
            git_root=r"C:\Projects\Git",
            project_root=r"C:\Projects\llm_website",
            roslyn_analyzer_path="tools/RoslynCodeAnalyzer/RoslynCodeAnalyzer.dll",
            analysis_timeout=300,
            max_concurrent_syncs=3
        )
    except ImportError:
        pytest.skip("PipelineConfig not available")


@pytest.fixture
def repository_config():
    """
    Create a RepositoryConfig for testing.
    """
    try:
        from git_pipeline.models.pipeline_models import RepositoryConfig
        return RepositoryConfig(
            name="TestRepo",
            path=r"C:\Projects\Git\TestRepo",
            db_name="testrepo",
            display_name="Test Repository",
            enabled=True,
            auto_sync=False
        )
    except ImportError:
        pytest.skip("RepositoryConfig not available")


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers for git pipeline tests."""
    config.addinivalue_line(
        "markers", "requires_git: mark test as requiring git installation"
    )
    config.addinivalue_line(
        "markers", "requires_dotnet: mark test as requiring .NET SDK"
    )
    config.addinivalue_line(
        "markers", "slow_analysis: mark test as slow due to code analysis"
    )
