"""
Test Fixtures Package
=====================

Exports MongoDB, LLM, and shared fixtures for pipeline tests.
"""

# Make fixture modules available
from . import llm_fixtures
from . import mongodb_fixtures
from . import shared_fixtures

# Import commonly used fixtures/classes
from .llm_fixtures import (
    LocalLLMClient,
    AsyncLocalLLMClient,
    LLMResponse,
    check_llm_health,
)

from .mongodb_fixtures import (
    get_test_mongodb_client,
    create_test_collection,
    insert_test_documents,
    cleanup_test_documents,
    create_mock_document,
    create_mock_sql_query,
    create_mock_audio_analysis,
    create_mock_code_method,
    create_mock_document_chunk,
)

from .shared_fixtures import (
    # Embedding fixtures
    MockEmbeddingService,
    mock_embedding_service,
    # Vector search fixtures
    MockVectorSearch,
    mock_vector_search,
    # SSE fixtures
    SSEEvent,
    SSEConsumer,
    sse_consumer,
    # Token assertion fixtures
    TokenAssertions,
    token_assertions,
    # HTTP client fixtures
    test_http_client,
    test_http_client_node,
    sync_http_client,
    # Test data generators
    TestDocumentGenerator,
    test_document_generator,
    # Response validators
    ResponseValidator,
    response_validator,
)

__all__ = [
    # Modules
    "llm_fixtures",
    "mongodb_fixtures",
    "shared_fixtures",
    # LLM fixtures
    "LocalLLMClient",
    "AsyncLocalLLMClient",
    "LLMResponse",
    "check_llm_health",
    # MongoDB fixtures
    "get_test_mongodb_client",
    "create_test_collection",
    "insert_test_documents",
    "cleanup_test_documents",
    "create_mock_document",
    "create_mock_sql_query",
    "create_mock_audio_analysis",
    "create_mock_code_method",
    "create_mock_document_chunk",
    # Shared fixtures - Embedding
    "MockEmbeddingService",
    "mock_embedding_service",
    # Shared fixtures - Vector search
    "MockVectorSearch",
    "mock_vector_search",
    # Shared fixtures - SSE
    "SSEEvent",
    "SSEConsumer",
    "sse_consumer",
    # Shared fixtures - Token assertions
    "TokenAssertions",
    "token_assertions",
    # Shared fixtures - HTTP clients
    "test_http_client",
    "test_http_client_node",
    "sync_http_client",
    # Shared fixtures - Test data generators
    "TestDocumentGenerator",
    "test_document_generator",
    # Shared fixtures - Response validators
    "ResponseValidator",
    "response_validator",
]
