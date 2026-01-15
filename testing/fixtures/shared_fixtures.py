"""
Shared Test Fixtures
====================

Common fixtures for E2E tests across all pipelines.
Includes mock services for embeddings, vector search, SSE streaming,
token assertions, and HTTP clients.
"""

import json
import pytest
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field


# =============================================================================
# Mock Embedding Service
# =============================================================================


class MockEmbeddingService:
    """
    Mock embedding service that returns deterministic embeddings.

    Uses text hashing to generate reproducible embeddings for testing,
    eliminating the need for actual embedding models during tests.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize mock embedding service.

        Args:
            dimension: Embedding vector dimension (default: 384 for sentence-transformers)
        """
        self.dimension = dimension
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        """
        Generate deterministic embedding based on text hash.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if text not in self._cache:
            # Use hash for determinism
            hash_val = hash(text)
            self._cache[text] = [
                (hash_val >> i & 0xFF) / 255.0
                for i in range(self.dimension)
            ]
        return self._cache[text]

    async def embed_async(self, text: str) -> List[float]:
        """Async version of embed."""
        return self.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts at once.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed(t) for t in texts]

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_batch."""
        return self.embed_batch(texts)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


@pytest.fixture
def mock_embedding_service():
    """Returns deterministic embeddings for reproducible tests."""
    return MockEmbeddingService()


# =============================================================================
# Mock Vector Search
# =============================================================================


class MockVectorSearch:
    """
    Mock vector search using cosine similarity.

    Provides a simple in-memory vector search implementation
    for testing without requiring external vector databases.
    """

    def __init__(self, embedding_service: MockEmbeddingService):
        """
        Initialize mock vector search.

        Args:
            embedding_service: Embedding service to use for query embeddings
        """
        self.embedding_service = embedding_service
        self.documents: List[Dict[str, Any]] = []

    def index(self, documents: List[Dict[str, Any]]):
        """
        Index documents for search.

        Args:
            documents: List of documents to index. Each document should have
                      a 'content' field or pre-computed 'embedding' field.
        """
        for doc in documents:
            if 'embedding' not in doc:
                doc['embedding'] = self.embedding_service.embed(doc.get('content', ''))
        self.documents = documents

    def add_document(self, document: Dict[str, Any]):
        """
        Add a single document to the index.

        Args:
            document: Document to add
        """
        if 'embedding' not in document:
            document['embedding'] = self.embedding_service.embed(document.get('content', ''))
        self.documents.append(document)

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 - 1.0)

        Returns:
            List of documents with similarity scores, sorted by similarity descending
        """
        query_embedding = self.embedding_service.embed(query)
        results = []

        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            if similarity >= threshold:
                result = {**doc, 'similarity': similarity}
                results.append(result)

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    async def search_async(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Async version of search."""
        return self.search(query, limit, threshold)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity value (0.0 - 1.0)
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def clear(self):
        """Clear all indexed documents."""
        self.documents = []


@pytest.fixture
def mock_vector_search(mock_embedding_service):
    """Vector search using cosine similarity."""
    return MockVectorSearch(mock_embedding_service)


# =============================================================================
# SSE Stream Consumer
# =============================================================================


@dataclass
class SSEEvent:
    """Represents a parsed SSE event."""
    event_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    retry: Optional[int] = None
    raw: str = ""


class SSEConsumer:
    """
    Helper for consuming Server-Sent Events (SSE) streams in tests.

    Parses SSE responses and provides utilities for event inspection
    and assertion.
    """

    def __init__(self):
        """Initialize SSE consumer."""
        self.events: List[Dict[str, Any]] = []
        self.raw_lines: List[str] = []

    async def consume(self, response) -> List[Dict[str, Any]]:
        """
        Consume SSE response and return parsed events.

        Args:
            response: HTTP response with SSE content (supports aiter_lines)

        Returns:
            List of parsed event dictionaries
        """
        self.events = []
        self.raw_lines = []

        async for line in response.aiter_lines():
            self.raw_lines.append(line)
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    self.events.append(data)
                except json.JSONDecodeError:
                    # Non-JSON data, store as raw string
                    self.events.append({'raw_data': line[6:]})

        return self.events

    def consume_sync(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Consume SSE lines synchronously.

        Args:
            lines: List of SSE response lines

        Returns:
            List of parsed event dictionaries
        """
        self.events = []
        self.raw_lines = lines

        for line in lines:
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    self.events.append(data)
                except json.JSONDecodeError:
                    self.events.append({'raw_data': line[6:]})

        return self.events

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """
        Filter events by type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of events matching the specified type
        """
        return [e for e in self.events if e.get('type') == event_type]

    def get_first_event_by_type(self, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the first event of a specific type.

        Args:
            event_type: Event type to find

        Returns:
            First matching event or None
        """
        events = self.get_events_by_type(event_type)
        return events[0] if events else None

    def get_last_event_by_type(self, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the last event of a specific type.

        Args:
            event_type: Event type to find

        Returns:
            Last matching event or None
        """
        events = self.get_events_by_type(event_type)
        return events[-1] if events else None

    def assert_event_sequence(self, expected_types: List[str]):
        """
        Assert that events appear in the expected sequence.

        Args:
            expected_types: List of expected event types in order

        Raises:
            AssertionError: If actual sequence doesn't match expected
        """
        actual_types = [e.get('type') for e in self.events]
        assert actual_types == expected_types, \
            f"Expected event sequence {expected_types}, got {actual_types}"

    def assert_contains_event_type(self, event_type: str, message: str = None):
        """
        Assert that at least one event of the specified type exists.

        Args:
            event_type: Event type to check for
            message: Optional custom assertion message
        """
        events = self.get_events_by_type(event_type)
        msg = message or f"Expected at least one event of type '{event_type}'"
        assert len(events) > 0, msg

    def assert_no_errors(self):
        """Assert that no error events occurred."""
        error_events = self.get_events_by_type('error')
        assert len(error_events) == 0, \
            f"Expected no errors, but got: {error_events}"

    def get_event_count(self) -> int:
        """Return the total number of events consumed."""
        return len(self.events)

    def get_event_types(self) -> List[str]:
        """Return list of all event types in order."""
        return [e.get('type') for e in self.events]


@pytest.fixture
def sse_consumer():
    """Helper for consuming SSE streams in tests."""
    return SSEConsumer()


# =============================================================================
# Token Usage Assertions
# =============================================================================


class TokenAssertions:
    """
    Helpers for asserting token usage in LLM responses.

    Provides utilities to verify that responses contain valid
    token usage information including capture, ranges, limits,
    estimation accuracy, and cost calculations.
    """

    # Cost per 1K tokens for different model types (example rates)
    COST_PER_1K_TOKENS = {
        "sql": {"prompt": 0.0001, "completion": 0.0002},
        "general": {"prompt": 0.0001, "completion": 0.0002},
        "code": {"prompt": 0.00015, "completion": 0.0003},
        "default": {"prompt": 0.0001, "completion": 0.0002},
    }

    def _get_token_values(self, response: Union[Dict, Any]) -> Dict[str, int]:
        """
        Extract token values from response.

        Args:
            response: Response object or dictionary

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens
        """
        if isinstance(response, dict):
            return {
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
                "total_tokens": response.get("total_tokens", 0),
            }
        return {
            "prompt_tokens": getattr(response, "prompt_tokens", 0),
            "completion_tokens": getattr(response, "completion_tokens", 0),
            "total_tokens": getattr(response, "total_tokens", 0),
        }

    def assert_tokens_captured(self, response: Union[Dict, Any]):
        """
        Assert that token counts are present in the response.

        Args:
            response: Response object or dictionary

        Raises:
            AssertionError: If token counts are missing
        """
        if isinstance(response, dict):
            has_prompt = "prompt_tokens" in response
            has_completion = "completion_tokens" in response
        else:
            has_prompt = hasattr(response, "prompt_tokens")
            has_completion = hasattr(response, "completion_tokens")

        assert has_prompt or has_completion, \
            "Response must contain prompt_tokens or completion_tokens"

    def assert_prompt_tokens_captured(self, response: Union[Dict, Any]):
        """
        Assert that prompt_tokens is present and non-negative.

        Args:
            response: Response object or dictionary
        """
        tokens = self._get_token_values(response)
        assert tokens["prompt_tokens"] >= 0, \
            f"prompt_tokens must be >= 0, got {tokens['prompt_tokens']}"

    def assert_completion_tokens_captured(self, response: Union[Dict, Any]):
        """
        Assert that completion_tokens is present and non-negative.

        Args:
            response: Response object or dictionary
        """
        tokens = self._get_token_values(response)
        assert tokens["completion_tokens"] >= 0, \
            f"completion_tokens must be >= 0, got {tokens['completion_tokens']}"

    def assert_total_equals_sum(self, response: Union[Dict, Any]):
        """
        Assert that total_tokens equals prompt_tokens + completion_tokens.

        Args:
            response: Response object or dictionary
        """
        tokens = self._get_token_values(response)
        expected_total = tokens["prompt_tokens"] + tokens["completion_tokens"]
        actual_total = tokens["total_tokens"]

        assert actual_total == expected_total, \
            f"total_tokens ({actual_total}) != prompt_tokens ({tokens['prompt_tokens']}) + " \
            f"completion_tokens ({tokens['completion_tokens']}) = {expected_total}"

    def assert_tokens_in_range(
        self,
        response: Union[Dict, Any],
        min_tokens: int = 1,
        max_tokens: int = 10000
    ):
        """
        Assert that total token count is within expected range.

        Args:
            response: Response object or dictionary
            min_tokens: Minimum expected tokens (inclusive)
            max_tokens: Maximum expected tokens (inclusive)

        Raises:
            AssertionError: If tokens are outside the specified range
        """
        tokens = self._get_token_values(response)
        total = tokens["total_tokens"]

        assert min_tokens <= total <= max_tokens, \
            f"Total tokens {total} not in range [{min_tokens}, {max_tokens}]"

    def assert_prompt_tokens_in_range(
        self,
        response: Union[Dict, Any],
        min_tokens: int = 1,
        max_tokens: int = 10000
    ):
        """
        Assert that prompt token count is within expected range.

        Args:
            response: Response object or dictionary
            min_tokens: Minimum expected tokens (inclusive)
            max_tokens: Maximum expected tokens (inclusive)
        """
        tokens = self._get_token_values(response)
        prompt = tokens["prompt_tokens"]

        assert min_tokens <= prompt <= max_tokens, \
            f"Prompt tokens {prompt} not in range [{min_tokens}, {max_tokens}]"

    def assert_completion_tokens_in_range(
        self,
        response: Union[Dict, Any],
        min_tokens: int = 1,
        max_tokens: int = 10000
    ):
        """
        Assert that completion token count is within expected range.

        Args:
            response: Response object or dictionary
            min_tokens: Minimum expected tokens (inclusive)
            max_tokens: Maximum expected tokens (inclusive)
        """
        tokens = self._get_token_values(response)
        completion = tokens["completion_tokens"]

        assert min_tokens <= completion <= max_tokens, \
            f"Completion tokens {completion} not in range [{min_tokens}, {max_tokens}]"

    def assert_prompt_tokens(
        self,
        response: Union[Dict, Any],
        expected: int,
        tolerance: float = 0.1
    ):
        """
        Assert that prompt tokens are within tolerance of expected value.

        Args:
            response: Response object or dictionary
            expected: Expected prompt token count
            tolerance: Allowed relative tolerance (0.1 = 10%)

        Raises:
            AssertionError: If prompt tokens are outside tolerance
        """
        tokens = self._get_token_values(response)
        actual = tokens["prompt_tokens"]

        if expected == 0:
            assert actual == 0, f"Expected 0 prompt tokens, got {actual}"
        else:
            diff = abs(actual - expected) / expected
            assert diff <= tolerance, \
                f"Prompt tokens {actual} not within {tolerance*100}% of {expected}"

    def assert_completion_tokens(
        self,
        response: Union[Dict, Any],
        expected: int,
        tolerance: float = 0.1
    ):
        """
        Assert that completion tokens are within tolerance of expected value.

        Args:
            response: Response object or dictionary
            expected: Expected completion token count
            tolerance: Allowed relative tolerance (0.1 = 10%)

        Raises:
            AssertionError: If completion tokens are outside tolerance
        """
        tokens = self._get_token_values(response)
        actual = tokens["completion_tokens"]

        if expected == 0:
            assert actual == 0, f"Expected 0 completion tokens, got {actual}"
        else:
            diff = abs(actual - expected) / expected
            assert diff <= tolerance, \
                f"Completion tokens {actual} not within {tolerance*100}% of {expected}"

    def assert_nonzero_tokens(self, response: Union[Dict, Any]):
        """
        Assert that response has non-zero token counts.

        Args:
            response: Response object or dictionary
        """
        tokens = self._get_token_values(response)
        total = tokens["total_tokens"]

        assert total > 0, "Expected non-zero token count"

    def assert_max_tokens_respected(
        self,
        response: Union[Dict, Any],
        max_tokens_requested: int,
        tolerance: float = 0.05
    ):
        """
        Assert that completion tokens do not exceed max_tokens parameter.

        Args:
            response: Response object or dictionary
            max_tokens_requested: The max_tokens value passed to the LLM
            tolerance: Allowed overage tolerance (0.05 = 5%)
        """
        tokens = self._get_token_values(response)
        completion = tokens["completion_tokens"]
        max_allowed = int(max_tokens_requested * (1 + tolerance))

        assert completion <= max_allowed, \
            f"Completion tokens ({completion}) exceeded max_tokens ({max_tokens_requested}) " \
            f"even with {tolerance*100}% tolerance"

    def assert_token_estimation_accuracy(
        self,
        estimated_tokens: int,
        actual_tokens: int,
        tolerance: float = 0.2
    ):
        """
        Assert that estimated tokens are within tolerance of actual tokens.

        Useful for comparing pre-computed estimates with actual LLM usage.

        Args:
            estimated_tokens: Pre-computed token estimate
            actual_tokens: Actual tokens from LLM response
            tolerance: Allowed relative tolerance (0.2 = 20%)
        """
        if actual_tokens == 0:
            assert estimated_tokens == 0, \
                f"Estimated {estimated_tokens} but actual was 0"
            return

        diff = abs(estimated_tokens - actual_tokens) / actual_tokens
        assert diff <= tolerance, \
            f"Token estimation ({estimated_tokens}) differs from actual ({actual_tokens}) " \
            f"by {diff*100:.1f}% (tolerance: {tolerance*100}%)"

    def calculate_cost(
        self,
        response: Union[Dict, Any],
        model_type: str = "default"
    ) -> float:
        """
        Calculate the cost for a response based on token usage.

        Args:
            response: Response object or dictionary
            model_type: Model type for cost lookup ("sql", "general", "code", "default")

        Returns:
            Estimated cost in dollars
        """
        tokens = self._get_token_values(response)
        rates = self.COST_PER_1K_TOKENS.get(model_type, self.COST_PER_1K_TOKENS["default"])

        prompt_cost = (tokens["prompt_tokens"] / 1000) * rates["prompt"]
        completion_cost = (tokens["completion_tokens"] / 1000) * rates["completion"]

        return prompt_cost + completion_cost

    def assert_cost_within_budget(
        self,
        response: Union[Dict, Any],
        max_cost: float,
        model_type: str = "default"
    ):
        """
        Assert that the response cost is within budget.

        Args:
            response: Response object or dictionary
            max_cost: Maximum allowed cost in dollars
            model_type: Model type for cost lookup
        """
        actual_cost = self.calculate_cost(response, model_type)
        assert actual_cost <= max_cost, \
            f"Response cost ${actual_cost:.6f} exceeds budget ${max_cost:.6f}"

    def get_token_summary(self, response: Union[Dict, Any]) -> Dict[str, Any]:
        """
        Get a summary of token usage from response.

        Args:
            response: Response object or dictionary

        Returns:
            Dict with token counts and derived metrics
        """
        tokens = self._get_token_values(response)
        return {
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"],
            "completion_ratio": (
                tokens["completion_tokens"] / tokens["total_tokens"]
                if tokens["total_tokens"] > 0 else 0
            ),
        }

    def assert_completion_ratio(
        self,
        response: Union[Dict, Any],
        min_ratio: float = 0.0,
        max_ratio: float = 1.0
    ):
        """
        Assert that completion tokens as a ratio of total is within range.

        Useful for detecting issues like overly verbose responses or
        responses that are too short relative to context.

        Args:
            response: Response object or dictionary
            min_ratio: Minimum completion/total ratio
            max_ratio: Maximum completion/total ratio
        """
        summary = self.get_token_summary(response)
        ratio = summary["completion_ratio"]

        assert min_ratio <= ratio <= max_ratio, \
            f"Completion ratio {ratio:.2f} not in range [{min_ratio}, {max_ratio}]"


@pytest.fixture
def token_assertions():
    """Helpers for asserting token usage."""
    return TokenAssertions()


# =============================================================================
# HTTP Test Client
# =============================================================================


@pytest.fixture
async def test_http_client():
    """
    Async HTTP client configured for testing.

    Pre-configured with:
    - Base URL: http://localhost:8001 (Python FastAPI service)
    - Timeout: 30 seconds
    - Content-Type: application/json

    Yields:
        httpx.AsyncClient instance
    """
    import httpx

    async with httpx.AsyncClient(
        base_url="http://localhost:8001",
        timeout=30.0,
        headers={"Content-Type": "application/json"}
    ) as client:
        yield client


@pytest.fixture
async def test_http_client_node():
    """
    Async HTTP client configured for Node.js server testing.

    Pre-configured with:
    - Base URL: http://localhost:3000 (Node.js RAG server)
    - Timeout: 60 seconds (longer for LLM operations)
    - Content-Type: application/json

    Yields:
        httpx.AsyncClient instance
    """
    import httpx

    async with httpx.AsyncClient(
        base_url="http://localhost:3000",
        timeout=60.0,
        headers={"Content-Type": "application/json"}
    ) as client:
        yield client


@pytest.fixture
async def node_api_client():
    """
    APITestClient configured for Node.js server testing.

    Pre-configured with:
    - Base URL: http://localhost:3000 (Node.js RAG server)
    - Timeout: 60 seconds (longer for LLM operations)

    Yields:
        APITestClient instance
    """
    from testing.utils.api_test_client import APITestClient

    async with APITestClient(base_url="http://localhost:3000", timeout=60.0) as client:
        yield client


@pytest.fixture
def sync_http_client():
    """
    Synchronous HTTP client for non-async tests.

    Returns:
        httpx.Client instance
    """
    import httpx

    with httpx.Client(
        base_url="http://localhost:8001",
        timeout=30.0,
        headers={"Content-Type": "application/json"}
    ) as client:
        yield client


# =============================================================================
# Test Data Generators
# =============================================================================


@dataclass
class TestDocumentGenerator:
    """Generator for test documents with embeddings."""
    embedding_service: MockEmbeddingService = field(default_factory=MockEmbeddingService)

    def create_document(
        self,
        content: str,
        doc_type: str = "test",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a test document with embedding.

        Args:
            content: Document content
            doc_type: Document type
            metadata: Optional metadata dictionary

        Returns:
            Document dictionary with embedding
        """
        doc = {
            'content': content,
            'type': doc_type,
            'embedding': self.embedding_service.embed(content),
        }
        if metadata:
            doc['metadata'] = metadata
        return doc

    def create_documents(
        self,
        contents: List[str],
        doc_type: str = "test"
    ) -> List[Dict[str, Any]]:
        """
        Create multiple test documents.

        Args:
            contents: List of document contents
            doc_type: Document type for all documents

        Returns:
            List of document dictionaries
        """
        return [self.create_document(c, doc_type) for c in contents]


@pytest.fixture
def test_document_generator(mock_embedding_service):
    """Generator for test documents with embeddings."""
    return TestDocumentGenerator(embedding_service=mock_embedding_service)


# =============================================================================
# Response Validators
# =============================================================================


class ResponseValidator:
    """Utilities for validating API responses."""

    @staticmethod
    def assert_success_response(response: Dict[str, Any]):
        """
        Assert that response indicates success.

        Args:
            response: API response dictionary
        """
        assert response.get('success', False) is True or \
               response.get('status') == 'success' or \
               'error' not in response, \
               f"Expected success response, got: {response}"

    @staticmethod
    def assert_error_response(response: Dict[str, Any], expected_error: Optional[str] = None):
        """
        Assert that response indicates an error.

        Args:
            response: API response dictionary
            expected_error: Optional expected error message substring
        """
        is_error = response.get('success') is False or \
                   response.get('status') == 'error' or \
                   'error' in response
        assert is_error, f"Expected error response, got: {response}"

        if expected_error:
            error_msg = response.get('error', response.get('message', ''))
            assert expected_error in str(error_msg), \
                f"Expected error containing '{expected_error}', got: {error_msg}"

    @staticmethod
    def assert_has_fields(response: Dict[str, Any], fields: List[str]):
        """
        Assert that response contains all required fields.

        Args:
            response: API response dictionary
            fields: List of required field names
        """
        missing = [f for f in fields if f not in response]
        assert not missing, f"Missing required fields: {missing}"

    @staticmethod
    def assert_field_type(response: Dict[str, Any], field: str, expected_type: type):
        """
        Assert that a field has the expected type.

        Args:
            response: API response dictionary
            field: Field name to check
            expected_type: Expected Python type
        """
        assert field in response, f"Field '{field}' not found in response"
        assert isinstance(response[field], expected_type), \
            f"Field '{field}' should be {expected_type.__name__}, got {type(response[field]).__name__}"


@pytest.fixture
def response_validator():
    """Utilities for validating API responses."""
    return ResponseValidator()
