"""
Document API Endpoints Tests
============================

Tests for the FastAPI document pipeline API endpoints.

Endpoints tested:
- POST /api/documents/query - RAG query with LLM generation
- POST /api/documents/query-stream - Streaming RAG query with SSE
- POST /api/documents/search - Direct vector search without LLM
- POST /api/documents/feedback - Submit user feedback
- GET /api/documents/cache/stats - Get cache statistics
- GET /api/documents/health - Health check endpoint
- GET /api/documents/projects - List available projects

These tests verify API request/response handling, validation, and error cases.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Test utilities
from fixtures.shared_fixtures import sse_consumer


# =============================================================================
# Mock Services for API Tests
# =============================================================================

class MockQueryResponse:
    """Mock QueryResponse for testing."""

    def __init__(
        self,
        answer: str = "Test answer",
        sources: List[Dict] = None,
        error: str = None,
    ):
        self.query_id = "test_query_123"
        self.query = "Test query"
        self.answer = answer
        self.sources = sources or []
        self.confidence = 0.85
        self.validation_passed = True
        self.validation_details = {"relevancy": 0.9, "faithfulness": 0.95, "completeness": 0.8}
        self.query_intent = MagicMock(value="factual")
        self.retrieval_used = True
        self.correction_applied = False
        self.total_time_ms = 150
        self.stage_timings = {"query_understanding": 20, "retrieval": 50, "generation": 80}
        self.token_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        self.error = error


class MockOrchestrator:
    """Mock KnowledgeBaseOrchestrator for API tests."""

    def __init__(
        self,
        response: MockQueryResponse = None,
        stream_events: List[Dict] = None,
    ):
        self.response = response or MockQueryResponse()
        self.stream_events = stream_events or []
        self._initialized = True
        self._semantic_cache = None

    async def process_query(self, request):
        return self.response

    async def process_query_stream(self, request):
        for event in self.stream_events:
            yield event

    async def record_feedback(self, feedback):
        return True


class MockMongoDBService:
    """Mock MongoDB service for API tests."""

    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []
        self.is_initialized = True
        self.db = MagicMock()

    async def initialize(self):
        pass

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> List[Dict]:
        return self.documents[:limit]

    @classmethod
    def get_instance(cls):
        return cls()


# =============================================================================
# Query Endpoint Tests
# =============================================================================

class TestQueryEndpoint:
    """Test the /api/documents/query endpoint."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock query response."""
        return MockQueryResponse(
            answer="The CRAG pattern adds corrective retrieval to improve answer quality.",
            sources=[
                {
                    "document_id": "doc_1",
                    "title": "CRAG Guide",
                    "content_preview": "CRAG is a pattern that...",
                    "score": 0.85,
                    "relevance": "relevant",
                }
            ],
        )

    @pytest.mark.asyncio
    async def test_query_endpoint_success(self, mock_response):
        """Test successful query processing."""
        try:
            from api.document_pipeline_routes import query_documents, DocumentQueryRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator(response=mock_response)

            request = DocumentQueryRequest(
                query="What is the CRAG pattern?",
                limit=5,
            )

            response = await query_documents(request)

            assert response.answer == mock_response.answer
            assert len(response.sources) > 0
            assert response.search_strategy == "crag-hybrid"
            assert response.confidence > 0

    @pytest.mark.asyncio
    async def test_query_endpoint_with_filters(self, mock_response):
        """Test query with filters."""
        try:
            from api.document_pipeline_routes import query_documents, DocumentQueryRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator(response=mock_response)

            request = DocumentQueryRequest(
                query="Show me engineering documents",
                project="knowledge_base",
                department="Engineering",
                doc_type="guide",
                limit=10,
            )

            response = await query_documents(request)
            assert response is not None

    @pytest.mark.asyncio
    async def test_query_endpoint_with_history(self, mock_response):
        """Test query with conversation history."""
        try:
            from api.document_pipeline_routes import query_documents, DocumentQueryRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator(response=mock_response)

            request = DocumentQueryRequest(
                query="What about that?",
                history=[
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG is Retrieval-Augmented Generation."},
                ],
                limit=5,
            )

            response = await query_documents(request)
            assert response is not None

    @pytest.mark.asyncio
    async def test_query_endpoint_skip_validation(self, mock_response):
        """Test query with validation skipped."""
        try:
            from api.document_pipeline_routes import query_documents, DocumentQueryRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator(response=mock_response)

            request = DocumentQueryRequest(
                query="Quick question",
                skip_validation=True,
                limit=3,
            )

            response = await query_documents(request)
            assert response is not None

    @pytest.mark.asyncio
    async def test_query_endpoint_error_handling(self):
        """Test query endpoint error handling."""
        try:
            from api.document_pipeline_routes import query_documents, DocumentQueryRequest
            from fastapi import HTTPException
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        error_response = MockQueryResponse(error="Test error occurred")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator(response=error_response)

            request = DocumentQueryRequest(query="Test query")

            with pytest.raises(HTTPException) as exc_info:
                await query_documents(request)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_query_endpoint_validation_error(self):
        """Test query endpoint with invalid request."""
        try:
            from api.document_pipeline_routes import DocumentQueryRequest
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        # Empty query should fail validation
        with pytest.raises(ValidationError):
            DocumentQueryRequest(query="")  # min_length=1


# =============================================================================
# Query Stream Endpoint Tests
# =============================================================================

class TestQueryStreamEndpoint:
    """Test the /api/documents/query-stream endpoint."""

    @pytest.fixture
    def mock_stream_events(self):
        """Create mock streaming events."""
        try:
            from orchestrator.models import StreamEvent, StreamEventType, PipelineStage
        except ImportError:
            return []

        return [
            StreamEvent(
                event_type=StreamEventType.STAGE_START,
                query_id="test_123",
                stage=PipelineStage.QUERY_UNDERSTANDING,
                message="Analyzing query...",
                elapsed_ms=0,
            ),
            StreamEvent(
                event_type=StreamEventType.STAGE_COMPLETE,
                query_id="test_123",
                stage=PipelineStage.QUERY_UNDERSTANDING,
                data={"intent": "factual"},
                elapsed_ms=20,
            ),
            StreamEvent(
                event_type=StreamEventType.STAGE_START,
                query_id="test_123",
                stage=PipelineStage.RETRIEVAL,
                message="Searching...",
                elapsed_ms=20,
            ),
            StreamEvent(
                event_type=StreamEventType.DOCUMENT_FOUND,
                query_id="test_123",
                stage=PipelineStage.RETRIEVAL,
                data={"title": "Test Doc", "score": 0.85},
                elapsed_ms=50,
            ),
            StreamEvent(
                event_type=StreamEventType.COMPLETE,
                query_id="test_123",
                stage=PipelineStage.COMPLETE,
                data={"answer": "Test answer"},
                elapsed_ms=150,
            ),
        ]

    @pytest.mark.asyncio
    async def test_stream_endpoint_returns_streaming_response(self, mock_stream_events):
        """Test that stream endpoint returns StreamingResponse."""
        try:
            from api.document_pipeline_routes import query_documents_stream, DocumentQueryRequest
            from fastapi.responses import StreamingResponse
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_orch = MockOrchestrator(stream_events=mock_stream_events)
            mock_get_orch.return_value = mock_orch

            request = DocumentQueryRequest(query="Stream test")
            response = await query_documents_stream(request)

            assert isinstance(response, StreamingResponse)
            assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_stream_endpoint_headers(self, mock_stream_events):
        """Test that stream endpoint has correct headers."""
        try:
            from api.document_pipeline_routes import query_documents_stream, DocumentQueryRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator(stream_events=mock_stream_events)

            request = DocumentQueryRequest(query="Stream test")
            response = await query_documents_stream(request)

            # Check headers
            headers = dict(response.headers)
            assert headers.get("cache-control") == "no-cache"


# =============================================================================
# Search Endpoint Tests
# =============================================================================

class TestSearchEndpoint:
    """Test the /api/documents/search endpoint."""

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        return [
            {
                "id": "chunk_1",
                "parent_id": "doc_1",
                "content": "This is the first document about RAG.",
                "title": "RAG Guide",
                "relevance_score": 0.92,
                "department": "Engineering",
            },
            {
                "id": "chunk_2",
                "parent_id": "doc_2",
                "content": "Vector search enables semantic matching.",
                "title": "Vector Search",
                "relevance_score": 0.85,
                "department": "Engineering",
            },
        ]

    @pytest.mark.asyncio
    async def test_search_endpoint_success(self, mock_search_results):
        """Test successful search operation."""
        try:
            from api.document_pipeline_routes import search_documents, SearchRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService(documents=mock_search_results)

        with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
            mock_class.get_instance.return_value = mock_mongo

            request = SearchRequest(
                query="RAG system",
                limit=10,
            )

            response = await search_documents(request)

            assert response.total_results == 2
            assert len(response.sources) == 2
            assert response.search_time_ms > 0

    @pytest.mark.asyncio
    async def test_search_endpoint_with_filters(self, mock_search_results):
        """Test search with filters."""
        try:
            from api.document_pipeline_routes import search_documents, SearchRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService(documents=mock_search_results)

        with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
            mock_class.get_instance.return_value = mock_mongo

            request = SearchRequest(
                query="documents",
                project="knowledge_base",
                department="Engineering",
                limit=5,
            )

            response = await search_documents(request)
            assert response is not None

    @pytest.mark.asyncio
    async def test_search_endpoint_empty_results(self):
        """Test search with no results."""
        try:
            from api.document_pipeline_routes import search_documents, SearchRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService(documents=[])

        with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
            mock_class.get_instance.return_value = mock_mongo

            request = SearchRequest(query="nonexistent topic")
            response = await search_documents(request)

            assert response.total_results == 0
            assert len(response.sources) == 0


# =============================================================================
# Feedback Endpoint Tests
# =============================================================================

class TestFeedbackEndpoint:
    """Test the /api/documents/feedback endpoint."""

    @pytest.mark.asyncio
    async def test_feedback_positive(self):
        """Test submitting positive feedback."""
        try:
            from api.document_pipeline_routes import submit_feedback, FeedbackRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator()

            request = FeedbackRequest(
                query_id="test_123",
                query="What is RAG?",
                answer="RAG is Retrieval-Augmented Generation.",
                feedback="positive",
            )

            response = await submit_feedback(request)

            assert response.success is True
            assert "successfully" in response.message.lower()

    @pytest.mark.asyncio
    async def test_feedback_negative(self):
        """Test submitting negative feedback."""
        try:
            from api.document_pipeline_routes import submit_feedback, FeedbackRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator()

            request = FeedbackRequest(
                query="Test question",
                answer="Wrong answer",
                feedback="negative",
                comment="The answer was incorrect",
            )

            response = await submit_feedback(request)

            assert response.success is True

    @pytest.mark.asyncio
    async def test_feedback_with_sources(self):
        """Test feedback with source documents."""
        try:
            from api.document_pipeline_routes import submit_feedback, FeedbackRequest
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = MockOrchestrator()

            request = FeedbackRequest(
                query="Test",
                answer="Answer",
                feedback="positive",
                sources=[
                    {"id": "doc_1", "title": "Source 1"},
                    {"id": "doc_2", "title": "Source 2"},
                ],
            )

            response = await submit_feedback(request)
            assert response.success is True


# =============================================================================
# Cache Stats Endpoint Tests
# =============================================================================

class TestCacheStatsEndpoint:
    """Test the /api/documents/cache/stats endpoint."""

    @pytest.mark.asyncio
    async def test_cache_stats_available(self):
        """Test cache stats when cache is available."""
        try:
            from api.document_pipeline_routes import get_cache_stats
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_cache = MagicMock()
        mock_cache.is_available = True
        mock_cache.get_stats = AsyncMock(return_value={
            "embedding_entries": 100,
            "results_entries": 50,
            "response_entries": 25,
            "total_hits": 500,
        })

        mock_orch = MockOrchestrator()
        mock_orch._semantic_cache = mock_cache

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = mock_orch

            response = await get_cache_stats()

            assert response["success"] is True
            assert "embedding_entries" in response

    @pytest.mark.asyncio
    async def test_cache_stats_unavailable(self):
        """Test cache stats when cache is not available."""
        try:
            from api.document_pipeline_routes import get_cache_stats
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_orch = MockOrchestrator()
        mock_orch._semantic_cache = None

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            mock_get_orch.return_value = mock_orch

            response = await get_cache_stats()

            assert response["success"] is True
            assert response["available"] is False


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Test the /api/documents/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when all services are healthy."""
        try:
            from api.document_pipeline_routes import health_check
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService()
        mock_llm = MagicMock()
        mock_llm.health_check = AsyncMock(return_value={"healthy": True})

        mock_orch = MockOrchestrator()
        mock_orch._llm_service = mock_llm
        mock_orch._semantic_cache = MagicMock(is_available=True)

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
                mock_get_orch.return_value = mock_orch
                mock_class.get_instance.return_value = mock_mongo

                response = await health_check()

                assert response.status in ["healthy", "degraded"]
                assert response.mongodb is True

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check when some services are down."""
        try:
            from api.document_pipeline_routes import health_check
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService()
        mock_mongo.is_initialized = False

        mock_orch = MockOrchestrator()
        mock_orch._llm_service = None
        mock_orch._semantic_cache = None

        with patch("api.document_pipeline_routes.get_orchestrator") as mock_get_orch:
            with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
                mock_get_orch.return_value = mock_orch
                mock_class.get_instance.return_value = mock_mongo

                response = await health_check()

                # Should still return a response even if degraded
                assert response.service == "document_pipeline"


# =============================================================================
# Projects Endpoint Tests
# =============================================================================

class TestProjectsEndpoint:
    """Test the /api/documents/projects endpoint."""

    @pytest.mark.asyncio
    async def test_list_projects(self):
        """Test listing available projects."""
        try:
            from api.document_pipeline_routes import list_projects
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService()
        mock_collection = MagicMock()
        mock_collection.distinct = AsyncMock(return_value=["gin", "warehouse", "marketing"])
        mock_mongo.db = {"documents": mock_collection}

        with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
            mock_class.get_instance.return_value = mock_mongo

            response = await list_projects()

            assert "projects" in response
            assert len(response["projects"]) > 0

    @pytest.mark.asyncio
    async def test_list_projects_fallback(self):
        """Test projects list fallback when MongoDB fails."""
        try:
            from api.document_pipeline_routes import list_projects
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        mock_mongo = MockMongoDBService()
        mock_collection = MagicMock()
        mock_collection.distinct = AsyncMock(side_effect=Exception("MongoDB error"))
        mock_mongo.db = {"documents": mock_collection}

        with patch("api.document_pipeline_routes.MongoDBService") as mock_class:
            mock_class.get_instance.return_value = mock_mongo

            response = await list_projects()

            # Should return fallback static list
            assert "projects" in response
            assert len(response["projects"]) > 0


# =============================================================================
# Request Validation Tests
# =============================================================================

class TestRequestValidation:
    """Test request validation for all endpoints."""

    def test_query_request_validation(self):
        """Test DocumentQueryRequest validation."""
        try:
            from api.document_pipeline_routes import DocumentQueryRequest
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        # Valid request
        valid = DocumentQueryRequest(query="Valid query")
        assert valid.limit == 5  # default
        assert valid.temperature == 0.1  # default

        # Invalid: empty query
        with pytest.raises(ValidationError):
            DocumentQueryRequest(query="")

        # Invalid: limit out of range
        with pytest.raises(ValidationError):
            DocumentQueryRequest(query="test", limit=100)  # max is 50

        # Invalid: temperature out of range
        with pytest.raises(ValidationError):
            DocumentQueryRequest(query="test", temperature=3.0)  # max is 2.0

    def test_search_request_validation(self):
        """Test SearchRequest validation."""
        try:
            from api.document_pipeline_routes import SearchRequest
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        # Valid request
        valid = SearchRequest(query="Search query")
        assert valid.limit == 10  # default

        # Invalid: empty query
        with pytest.raises(ValidationError):
            SearchRequest(query="")

        # Invalid: limit out of range
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=200)  # max is 100

    def test_feedback_request_validation(self):
        """Test FeedbackRequest validation."""
        try:
            from api.document_pipeline_routes import FeedbackRequest
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Document pipeline routes not available")

        # Valid request
        valid = FeedbackRequest(
            query="Test query",
            answer="Test answer",
            feedback="positive",
        )
        assert valid.comment is None  # optional

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            FeedbackRequest(query="test")  # missing answer and feedback
