"""
Shared pytest configuration and fixtures for agent tests.

This file provides common fixtures used across all agent test files.
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add agents directory to path for imports
AGENTS_DIR = Path(__file__).parent.parent
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))


# =============================================================================
# Event Loop Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing without actual LLM calls."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value="Mock LLM response")
    mock.generate_stream = AsyncMock()
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client for testing without database."""
    mock = MagicMock()
    mock.admin.command = AsyncMock(return_value={"ok": 1})
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for vector operations."""
    import numpy as np
    mock = MagicMock()
    mock.embed = MagicMock(return_value=np.random.rand(384).tolist())
    mock.embed_batch = MagicMock(
        side_effect=lambda texts: [np.random.rand(384).tolist() for _ in texts]
    )
    return mock


# =============================================================================
# Integration Fixtures (require external services)
# =============================================================================

@pytest.fixture
def mongodb_client():
    """
    Real MongoDB client fixture for integration tests.

    Requires MONGODB_URI environment variable or uses default.
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongodb_uri = os.environ.get(
            "MONGODB_URI",
            "mongodb://localhost:27017/?directConnection=true"
        )
        client = AsyncIOMotorClient(mongodb_uri)
        yield client
        client.close()
    except ImportError:
        pytest.skip("motor not installed")


@pytest.fixture
def llm_service():
    """
    Real LLM service fixture for integration tests.

    Requires LLM service running at configured endpoint.
    """
    try:
        # Import from python_services if available
        python_services_path = AGENTS_DIR.parent / "python_services"
        if str(python_services_path) not in sys.path:
            sys.path.insert(0, str(python_services_path))

        from services.llm_service import get_llm_service
        return get_llm_service()
    except ImportError:
        pytest.skip("LLM service not available")


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_code_context():
    """Sample code context for testing code agent."""
    return {
        "file_path": "test/sample.cs",
        "language": "csharp",
        "content": """
public class SampleClass
{
    public void SampleMethod()
    {
        Console.WriteLine("Hello, World!");
    }
}
""",
        "project": "TestProject"
    }


@pytest.fixture
def sample_task_context():
    """Sample task context for testing task agent."""
    return {
        "task_id": "test-task-001",
        "description": "Fix bug in authentication module",
        "priority": "high",
        "project": "TestProject",
        "files": ["auth/login.cs", "auth/session.cs"]
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks test as unit test (fast, no external deps)"
    )
    config.addinivalue_line(
        "markers", "integration: marks test as integration test (requires services)"
    )
    config.addinivalue_line(
        "markers", "e2e: marks test as end-to-end test (full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks test as slow (> 30 seconds)"
    )
