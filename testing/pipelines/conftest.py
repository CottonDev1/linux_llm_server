"""
Pytest Configuration for Pipeline Tests
========================================

Centralized configuration for all pipeline tests in the testing directory.

This conftest.py:
- Adds testing root to sys.path for proper imports
- Imports from config.settings (centralized configuration)
- Re-exports fixtures from fixtures/ directory
- Provides pipeline-specific fixtures and configuration

Usage in pipeline tests:
    from fixtures.llm_fixtures import LocalLLMClient
    from fixtures.mongodb_fixtures import create_test_collection

    def test_something(mongodb_client, llm_client, pipeline_config):
        # Use fixtures directly
        pass
"""

import os
import sys
import uuid
import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from typing import Generator, AsyncGenerator

# Add testing root to sys.path (first, so testing/config takes precedence)
TESTING_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TESTING_ROOT))

# Add project root to sys.path to support "from testing.utils..." imports
PROJECT_ROOT = TESTING_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add python_services to sys.path for pipeline imports (after testing root)
# Using append to ensure testing/config takes precedence over python_services/config.py
PYTHON_SERVICES_ROOT = TESTING_ROOT.parent / "python_services"
sys.path.append(str(PYTHON_SERVICES_ROOT))

# Import centralized settings
from config.settings import settings
from config.test_config import TestConfig, PipelineTestConfig, get_test_config

# Re-export fixture modules for convenience
from fixtures import llm_fixtures
from fixtures import mongodb_fixtures
from fixtures import shared_fixtures

# Import shared fixtures to make them available in pipeline tests
from fixtures.shared_fixtures import (
    mock_embedding_service,
    mock_vector_search,
    sse_consumer,
    token_assertions,
    test_http_client,
    test_http_client_node,
    node_api_client,
    sync_http_client,
    test_document_generator,
    response_validator,
)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """
    Get test configuration for the entire test session.

    Configuration is loaded from settings and environment variables.
    All LLM endpoints must be local llama.cpp only.
    """
    config = get_test_config(
        test_run_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    return config


@pytest.fixture(scope="function")
def pipeline_config(test_config: TestConfig, request) -> PipelineTestConfig:
    """
    Get pipeline-specific configuration for individual tests.

    The pipeline name is determined from the test module path.
    """
    # Extract pipeline name from module path
    module_path = request.module.__file__
    pipeline_name = "unknown"

    pipeline_dirs = ["sql", "audio", "query", "git", "code_flow", "code_assistance", "document_agent"]
    for p in pipeline_dirs:
        if f"/{p}/" in module_path or f"\\{p}\\" in module_path:
            pipeline_name = p
            break

    return PipelineTestConfig(
        pipeline_name=pipeline_name,
        mongodb=test_config.mongodb,
        llm=test_config.llm,
        test_run_id=test_config.test_run_id,
        cleanup_after_test=test_config.cleanup_after_test,
    )


# =============================================================================
# MongoDB Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def mongodb_client(test_config: TestConfig):
    """
    Get MongoDB client for the test session.

    Provides a pymongo MongoClient connected to the configured MongoDB instance.
    """
    from pymongo import MongoClient

    client = MongoClient(
        test_config.mongodb.uri,
        serverSelectionTimeoutMS=test_config.mongodb.timeout_ms
    )

    # Verify connection
    try:
        client.admin.command('ping')
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")

    yield client

    client.close()


@pytest.fixture(scope="session")
def mongodb_database(mongodb_client, test_config: TestConfig):
    """Get the MongoDB database for tests."""
    return mongodb_client[test_config.mongodb.database]


@pytest.fixture(scope="function")
def test_collection(mongodb_database, pipeline_config: PipelineTestConfig):
    """
    Get a test-specific collection with automatic cleanup.

    Creates a collection prefixed with 'test_' for isolation.
    Automatically cleaned up after the test.
    """
    collection_name = f"test_{pipeline_config.pipeline_name}_{uuid.uuid4().hex[:8]}"
    collection = mongodb_database[collection_name]

    yield collection

    # Cleanup
    if pipeline_config.cleanup_after_test:
        collection.drop()


@pytest.fixture(scope="function")
async def async_mongodb_client(test_config: TestConfig):
    """
    Get async MongoDB client using motor.

    For tests requiring async MongoDB operations.
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        pytest.skip("motor not installed for async MongoDB")

    client = AsyncIOMotorClient(
        test_config.mongodb.uri,
        serverSelectionTimeoutMS=test_config.mongodb.timeout_ms
    )

    yield client

    client.close()


# =============================================================================
# LLM Fixtures (Local llama.cpp Only)
# =============================================================================

@pytest.fixture(scope="session")
def llm_client(test_config: TestConfig):
    """
    Get local LLM client for llama.cpp.

    IMPORTANT: Only local endpoints are permitted.
    No external APIs (OpenAI, Anthropic, etc.) allowed.
    """
    from fixtures.llm_fixtures import LocalLLMClient

    return LocalLLMClient(
        sql_endpoint=test_config.llm.sql_endpoint,
        general_endpoint=test_config.llm.general_endpoint,
        code_endpoint=test_config.llm.code_endpoint,
        timeout=test_config.llm.request_timeout,
    )


@pytest.fixture(scope="function")
async def async_llm_client(test_config: TestConfig):
    """
    Get async local LLM client for llama.cpp.
    """
    from fixtures.llm_fixtures import AsyncLocalLLMClient

    client = AsyncLocalLLMClient(
        sql_endpoint=test_config.llm.sql_endpoint,
        general_endpoint=test_config.llm.general_endpoint,
        code_endpoint=test_config.llm.code_endpoint,
        timeout=test_config.llm.request_timeout,
    )

    yield client

    await client.close()


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def test_document(pipeline_config: PipelineTestConfig) -> dict:
    """Generate a test document with proper markers."""
    return {
        "_id": f"test_{uuid.uuid4().hex}",
        "name": f"test_document_{uuid.uuid4().hex[:8]}",
        "content": "This is test content for pipeline testing.",
        "is_test": True,
        "test_run_id": pipeline_config.test_run_id,
        "test_marker": True,
        "created_at": datetime.utcnow(),
        "pipeline": pipeline_config.pipeline_name,
    }


@pytest.fixture(scope="function")
def test_documents(pipeline_config: PipelineTestConfig) -> list:
    """Generate multiple test documents."""
    docs = []
    for i in range(5):
        docs.append({
            "_id": f"test_{uuid.uuid4().hex}",
            "name": f"test_document_{i}_{uuid.uuid4().hex[:8]}",
            "content": f"Test content {i} for pipeline testing.",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
            "test_marker": True,
            "created_at": datetime.utcnow(),
            "pipeline": pipeline_config.pipeline_name,
        })
    return docs


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(scope="function", autouse=False)
def cleanup_test_data(mongodb_database, pipeline_config: PipelineTestConfig):
    """
    Cleanup fixture that removes test data after each test.

    Enable by including in test function parameters.
    """
    yield

    if pipeline_config.cleanup_after_test and pipeline_config.test_run_id:
        for collection_name in mongodb_database.list_collection_names():
            collection = mongodb_database[collection_name]
            result = collection.delete_many({
                "$or": [
                    {"test_run_id": pipeline_config.test_run_id},
                    {"is_test": True},
                    {"_id": {"$regex": "^test_"}},
                ]
            })
            if result.deleted_count > 0:
                print(f"Cleaned up {result.deleted_count} docs from {collection_name}")


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Skip Conditions and Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_mongodb: mark test as requiring MongoDB connection"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring local LLM endpoint"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
