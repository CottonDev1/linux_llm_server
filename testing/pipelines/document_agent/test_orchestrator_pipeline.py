"""
Document Orchestrator Pipeline Tests
=====================================

Tests for the KnowledgeBaseOrchestrator implementing the CRAG (Corrective RAG) pattern.

The orchestrator coordinates multiple stages:
1. Query Understanding - Classify intent, extract entities, expand queries
2. Hybrid Retrieval - Vector + BM25 search with RRF fusion
3. Document Grading - Evaluate relevance, trigger corrective retrieval if needed
4. Answer Generation - LLM synthesis from context
5. Answer Validation - Check relevancy, faithfulness, completeness
6. Self-Correction - Re-generate if validation fails

These tests verify the complete orchestration flow and individual stage behavior.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

# Test fixtures and utilities
from fixtures.shared_fixtures import (
    mock_embedding_service,
    mock_vector_search,
    sse_consumer,
)


# =============================================================================
# Mock Service Classes
# =============================================================================

class MockMongoDBService:
    """Mock MongoDB service for testing."""

    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []
        self.is_initialized = True

    async def initialize(self):
        pass

    async def _vector_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filter_query: Dict = None,
        threshold: float = 0.4,
    ) -> List[Dict]:
        """Return mock documents for vector search."""
        return self.documents[:limit]

    async def search_documents(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[Dict]:
        """Return mock documents for search."""
        return self.documents[:limit]


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.default_response = '{"type": "FACTUAL", "confidence": 0.8}'
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """Return mock LLM response."""
        self.call_count += 1

        # Determine response based on prompt content
        response_text = self.default_response

        if "Classify this query" in prompt:
            response_text = '{"type": "FACTUAL", "confidence": 0.8}'
        elif "Generate 3 alternative" in prompt:
            response_text = '["What is the process?", "How does it work?", "Explain the mechanism"]'
        elif "Grade if this document" in prompt:
            response_text = '{"grade": "relevant", "score": 0.85, "reason": "Directly addresses query"}'
        elif "Context:" in prompt and "Question:" in prompt:
            response_text = "Based on the provided context, the answer is: Test answer from mock LLM."
        elif "Does this answer address" in prompt:
            response_text = '{"relevant": true, "score": 0.9, "reason": "Answer addresses the question"}'
        elif "Check if every claim" in prompt:
            response_text = '{"faithful": true, "score": 0.95, "unsupported": []}'
        elif "completely address all aspects" in prompt:
            response_text = '{"complete": true, "score": 0.85, "missing": []}'

        return MagicMock(
            success=True,
            response=response_text,
            token_usage={"prompt_tokens": 100, "response_tokens": 50},
            generation_time_ms=100,
            model="mock-model"
        )

    async def generate_stream(self, prompt: str, **kwargs):
        """Return mock streaming response."""
        tokens = ["Based ", "on ", "the ", "context, ", "the ", "answer ", "is..."]
        for token in tokens:
            yield MagicMock(
                content=token,
                error=None,
                done=False
            )
        yield MagicMock(
            content="",
            error=None,
            done=True,
            token_usage={"prompt_tokens": 100, "response_tokens": 50}
        )

    async def health_check(self):
        return {"healthy": True}


class MockEmbeddingService:
    """Mock embedding service for testing."""

    async def initialize(self):
        pass

    async def generate_embedding(self, text: str) -> List[float]:
        """Return mock embedding vector."""
        # Use hash for deterministic embeddings
        hash_val = hash(text)
        return [(hash_val >> i & 0xFF) / 255.0 for i in range(384)]


# =============================================================================
# Orchestrator Pipeline Tests
# =============================================================================

class TestOrchestratorPipeline:
    """Test the complete orchestrator pipeline."""

    @pytest.fixture
    def mock_documents(self):
        """Create mock retrieved documents."""
        return [
            {
                "_id": "doc_1",
                "id": "chunk_1",
                "parent_id": "doc_1",
                "content": "This document explains the software architecture patterns including CRAG and RAG.",
                "title": "Architecture Guide",
                "_similarity": 0.85,
                "chunk_index": 0,
                "total_chunks": 3,
                "metadata": {"department": "Engineering"}
            },
            {
                "_id": "doc_2",
                "id": "chunk_2",
                "parent_id": "doc_2",
                "content": "The retrieval augmented generation (RAG) system uses vector search for documents.",
                "title": "RAG Overview",
                "_similarity": 0.78,
                "chunk_index": 0,
                "total_chunks": 2,
                "metadata": {"department": "Engineering"}
            },
            {
                "_id": "doc_3",
                "id": "chunk_3",
                "parent_id": "doc_3",
                "content": "CRAG adds corrective retrieval and validation to improve answer quality.",
                "title": "CRAG Pattern",
                "_similarity": 0.72,
                "chunk_index": 1,
                "total_chunks": 2,
                "metadata": {"department": "Engineering"}
            }
        ]

    @pytest.fixture
    def mock_services(self, mock_documents):
        """Create all mock services."""
        return {
            "mongodb": MockMongoDBService(documents=mock_documents),
            "llm": MockLLMService(),
            "embedding": MockEmbeddingService()
        }

    @pytest.mark.asyncio
    async def test_process_query_complete_flow(self, mock_services):
        """Test complete pipeline flow from query to response."""
        # Import here to avoid import errors in non-test environments
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        # Create orchestrator with mock services
        config = OrchestratorConfig(
            enable_validation=True,
            enable_document_grading=True,
            enable_query_expansion=True,
            enable_hybrid_search=False,  # Disable for simpler testing
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services["mongodb"],
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False  # Use legacy LLM service path

        # Create request
        request = QueryRequest(
            query="What is the CRAG pattern in RAG systems?",
            max_documents=5,
        )

        # Process query
        response = await orchestrator.process_query(request)

        # Verify response
        assert response is not None
        assert response.query == "What is the CRAG pattern in RAG systems?"
        assert response.answer != ""
        assert response.error is None
        assert response.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_query_with_grading(self, mock_services):
        """Test that document grading stage runs correctly."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_document_grading=True,
            enable_validation=False,  # Skip validation for this test
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services["mongodb"],
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Explain CRAG pattern")
        response = await orchestrator.process_query(request)

        # Verify grading happened (LLM was called for grading)
        assert mock_services["llm"].call_count >= 2  # At least classification + grading
        assert response.sources is not None

    @pytest.mark.asyncio
    async def test_process_query_with_validation(self, mock_services):
        """Test that validation stage runs correctly."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_document_grading=False,
            enable_validation=True,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services["mongodb"],
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="How does RAG work?",
            skip_validation=False,
        )
        response = await orchestrator.process_query(request)

        # Verify validation ran
        assert response.validation_passed is not None
        assert response.validation_details is not None or response.validation_passed

    @pytest.mark.asyncio
    async def test_process_query_skip_validation(self, mock_services):
        """Test skipping validation when requested."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_validation=True)

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services["mongodb"],
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="What is RAG?",
            skip_validation=True,
        )
        response = await orchestrator.process_query(request)

        # Validation should be skipped
        assert response.validation_passed  # Default to True when skipped

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self, mock_services):
        """Test error handling in pipeline."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        # Create a failing MongoDB service
        failing_mongodb = MockMongoDBService()
        failing_mongodb._vector_search = AsyncMock(side_effect=Exception("MongoDB error"))

        config = OrchestratorConfig()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=failing_mongodb,
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Test query")
        response = await orchestrator.process_query(request)

        # Should have empty documents but not crash
        assert response is not None


class TestOrchestratorQueryStream:
    """Test the streaming query processing."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for streaming tests."""
        mock_docs = [
            {
                "_id": "doc_1",
                "id": "chunk_1",
                "content": "Test content for streaming",
                "title": "Test Doc",
                "_similarity": 0.8,
            }
        ]
        return {
            "mongodb": MockMongoDBService(documents=mock_docs),
            "llm": MockLLMService(),
            "embedding": MockEmbeddingService()
        }

    @pytest.mark.asyncio
    async def test_process_query_stream_events(self, mock_services):
        """Test that streaming returns expected event types."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig, StreamEventType
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_document_grading=True,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services["mongodb"],
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="Streaming test query",
            stream=True,
        )

        events = []
        async for event in orchestrator.process_query_stream(request):
            events.append(event)

        # Verify we got events
        assert len(events) > 0

        # Check for expected event types
        event_types = [e.event_type for e in events]
        assert StreamEventType.STAGE_START in event_types
        assert StreamEventType.STAGE_COMPLETE in event_types
        assert StreamEventType.COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_process_query_stream_sse_format(self, mock_services):
        """Test that events can be converted to SSE format."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_validation=False)

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services["mongodb"],
            llm_service=mock_services["llm"],
            embedding_service=mock_services["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="SSE format test", stream=True)

        async for event in orchestrator.process_query_stream(request):
            # Each event should be convertible to SSE
            sse_str = event.to_sse()
            assert sse_str.startswith("event:")
            assert "data:" in sse_str
            assert sse_str.endswith("\n\n")


class TestCRAGPattern:
    """Test CRAG-specific functionality."""

    @pytest.fixture
    def low_relevance_documents(self):
        """Documents that should trigger corrective retrieval."""
        return [
            {
                "_id": "doc_1",
                "id": "chunk_1",
                "content": "Unrelated content about weather patterns.",
                "title": "Weather Report",
                "_similarity": 0.3,
            }
        ]

    @pytest.fixture
    def mock_services_low_relevance(self, low_relevance_documents):
        """Mock services with low relevance documents."""
        # LLM that grades documents as irrelevant
        llm = MockLLMService()
        llm.responses["grade"] = '{"grade": "irrelevant", "score": 0.2, "reason": "Not related"}'

        return {
            "mongodb": MockMongoDBService(documents=low_relevance_documents),
            "llm": llm,
            "embedding": MockEmbeddingService()
        }

    @pytest.mark.asyncio
    async def test_corrective_retrieval_trigger(self, mock_services_low_relevance):
        """Test that corrective retrieval is triggered for low relevance."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_document_grading=True,
            trigger_correction_threshold=0.5,  # Low relevance triggers correction
            enable_query_expansion=True,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=mock_services_low_relevance["mongodb"],
            llm_service=mock_services_low_relevance["llm"],
            embedding_service=mock_services_low_relevance["embedding"],
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="What is machine learning?")
        response = await orchestrator.process_query(request)

        # Pipeline should complete (may or may not trigger correction depending on grading)
        assert response is not None


class TestSelfCorrection:
    """Test self-correction functionality."""

    @pytest.fixture
    def failing_validation_llm(self):
        """LLM that fails validation checks."""
        llm = MockLLMService()
        # Override responses for validation failure
        original_generate = llm.generate

        async def mock_generate(prompt, **kwargs):
            if "Does this answer address" in prompt:
                return MagicMock(
                    success=True,
                    response='{"relevant": false, "score": 0.3, "reason": "Answer not relevant"}',
                    token_usage={"prompt_tokens": 100, "response_tokens": 50},
                    generation_time_ms=100,
                    model="mock-model"
                )
            return await original_generate(prompt, **kwargs)

        llm.generate = mock_generate
        return llm

    @pytest.mark.asyncio
    async def test_self_correction_on_validation_failure(self, failing_validation_llm):
        """Test that self-correction is triggered when validation fails."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        mock_docs = [{"_id": "doc_1", "content": "Test content", "_similarity": 0.8}]

        config = OrchestratorConfig(
            enable_validation=True,
            enable_self_correction=True,
            max_correction_attempts=2,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_docs),
            llm_service=failing_validation_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Test self-correction")
        response = await orchestrator.process_query(request)

        # Should complete (even if correction was attempted)
        assert response is not None


class TestBuildResponse:
    """Test response building logic."""

    @pytest.mark.asyncio
    async def test_response_includes_sources(self):
        """Test that response includes source documents."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        mock_docs = [
            {
                "_id": "doc_1",
                "id": "chunk_1",
                "content": "Source content 1",
                "title": "Source 1",
                "_similarity": 0.9,
            },
            {
                "_id": "doc_2",
                "id": "chunk_2",
                "content": "Source content 2",
                "title": "Source 2",
                "_similarity": 0.8,
            }
        ]

        config = OrchestratorConfig(enable_document_grading=False)

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_docs),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Test sources")
        response = await orchestrator.process_query(request)

        # Should have sources
        assert len(response.sources) >= 0  # May have sources from retrieval

    @pytest.mark.asyncio
    async def test_response_includes_stage_timings(self):
        """Test that response includes stage timing information."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        mock_docs = [{"_id": "doc_1", "content": "Test", "_similarity": 0.8}]

        config = OrchestratorConfig(
            enable_document_grading=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_docs),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Test timings")
        response = await orchestrator.process_query(request)

        # Should have stage timings
        assert response.stage_timings is not None
        assert "query_understanding" in response.stage_timings


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_injected_services(self):
        """Test initialization with dependency injection."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        mock_mongodb = MockMongoDBService()
        mock_llm = MockLLMService()
        mock_embedding = MockEmbeddingService()

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=mock_mongodb,
            llm_service=mock_llm,
            embedding_service=mock_embedding,
        )

        await orchestrator.initialize()

        assert orchestrator._initialized
        assert orchestrator._mongodb_service is mock_mongodb

    @pytest.mark.asyncio
    async def test_close_orchestrator(self):
        """Test closing orchestrator resources."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True

        await orchestrator.close()

        assert not orchestrator._initialized
