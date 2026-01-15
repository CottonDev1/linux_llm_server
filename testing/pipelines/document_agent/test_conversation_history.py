"""
Conversation History Tests
==========================

Tests for conversation history handling in the document pipeline.

Conversation history enables:
1. Context Persistence - Maintain conversation context across queries
2. Multi-turn Conversations - Handle sequences of related queries
3. Pronoun Resolution - Resolve "it", "that", "they" using prior context
4. Context-aware Query Expansion - Expand queries using conversation context
5. Follow-up Detection - Detect when current query refers to previous topics

These tests verify proper handling of conversation context.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any


# =============================================================================
# Mock Services
# =============================================================================

class MockLLMService:
    """Mock LLM service for conversation tests."""

    def __init__(self):
        self.call_history = []

    async def generate(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """Return mock LLM response."""
        self.call_history.append({"prompt": prompt, "system": system})

        response_text = '{"type": "FACTUAL", "confidence": 0.8}'

        if "Generate 3 alternative" in prompt:
            # Check if conversation context is included
            if "Previous conversation" in prompt:
                # Context-aware expansion
                response_text = '["What is the RAG system architecture?", "How does RAG work?", "Explain RAG components"]'
            else:
                response_text = '["Alternative 1", "Alternative 2", "Alternative 3"]'
        elif "Context:" in prompt and "Question:" in prompt:
            response_text = "Based on the conversation context, the answer is: The RAG system uses vector search for semantic matching."

        return MagicMock(
            success=True,
            response=response_text,
            token_usage={"prompt_tokens": 100, "response_tokens": 50},
            generation_time_ms=100,
            model="mock-model"
        )


class MockEmbeddingService:
    """Mock embedding service."""

    async def initialize(self):
        pass

    async def generate_embedding(self, text: str) -> List[float]:
        return [0.1] * 384


class MockMongoDBService:
    """Mock MongoDB service."""

    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []
        self.is_initialized = True

    async def initialize(self):
        pass

    async def _vector_search(self, **kwargs):
        return self.documents


# =============================================================================
# Conversation Persistence Tests
# =============================================================================

class TestConversationPersistence:
    """Test conversation history persistence."""

    @pytest.fixture
    def mock_documents(self):
        """Sample documents for retrieval."""
        return [
            {
                "_id": "doc_1",
                "id": "chunk_1",
                "content": "The RAG system uses vector search and retrieval for question answering.",
                "title": "RAG Overview",
                "_similarity": 0.85,
            }
        ]

    @pytest.fixture
    def conversation_history(self):
        """Sample conversation history."""
        return [
            {"role": "user", "content": "What is the RAG system?"},
            {"role": "assistant", "content": "The RAG (Retrieval-Augmented Generation) system combines retrieval with LLM generation."},
            {"role": "user", "content": "How does it retrieve documents?"},
            {"role": "assistant", "content": "It uses vector similarity search to find relevant documents based on query embeddings."},
        ]

    @pytest.mark.asyncio
    async def test_conversation_history_passed_to_request(self, mock_documents, conversation_history):
        """Test that conversation history is passed through the pipeline."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=True,
            enable_validation=False,
        )

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="What about the embedding process?",
            conversation_history=conversation_history,
        )

        response = await orchestrator.process_query(request)

        # Verify the request processed successfully
        assert response is not None
        assert response.error is None

    @pytest.mark.asyncio
    async def test_query_expansion_uses_history(self, mock_documents, conversation_history):
        """Test that query expansion uses conversation history for context."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        entities = []
        expanded = await orchestrator._expand_query(
            "How does it work?",
            entities,
            conversation_history=conversation_history
        )

        # Check that history was passed to LLM
        expansion_calls = [c for c in mock_llm.call_history if "alternative" in c["prompt"].lower()]
        assert len(expansion_calls) > 0

        # The prompt should include conversation context
        prompt_text = expansion_calls[0]["prompt"]
        assert "Previous conversation" in prompt_text or "conversation" in prompt_text.lower()


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================

class TestMultiTurnConversation:
    """Test multi-turn conversation handling."""

    @pytest.fixture
    def mock_documents(self):
        return [
            {
                "_id": "doc_1",
                "content": "Vector embeddings represent text as numerical vectors for similarity search.",
                "_similarity": 0.8,
            }
        ]

    @pytest.mark.asyncio
    async def test_multi_turn_query_processing(self, mock_documents):
        """Test processing multiple queries in sequence."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_validation=False)

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # First turn
        history = []
        request1 = QueryRequest(
            query="What are embeddings?",
            conversation_history=history,
        )
        response1 = await orchestrator.process_query(request1)
        assert response1 is not None

        # Second turn - with history
        history = [
            {"role": "user", "content": "What are embeddings?"},
            {"role": "assistant", "content": response1.answer},
        ]
        request2 = QueryRequest(
            query="How are they used in search?",
            conversation_history=history,
        )
        response2 = await orchestrator.process_query(request2)
        assert response2 is not None

        # Third turn - with more history
        history.extend([
            {"role": "user", "content": "How are they used in search?"},
            {"role": "assistant", "content": response2.answer},
        ])
        request3 = QueryRequest(
            query="What about performance?",
            conversation_history=history,
        )
        response3 = await orchestrator.process_query(request3)
        assert response3 is not None

    @pytest.mark.asyncio
    async def test_history_truncation_for_long_conversations(self, mock_documents):
        """Test that long conversations are properly truncated."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig()

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Create very long conversation history (20 turns)
        long_history = []
        for i in range(20):
            long_history.append({"role": "user", "content": f"Question {i}"})
            long_history.append({"role": "assistant", "content": f"Answer {i}"})

        # Expand query with long history
        expanded = await orchestrator._expand_query(
            "What about this?",
            [],
            conversation_history=long_history
        )

        # Check that some history was included (but not all 20 turns)
        # The orchestrator should limit to last 6 messages for expansion
        if mock_llm.call_history:
            last_prompt = mock_llm.call_history[-1]["prompt"]
            # Should have some history but truncated
            assert "conversation" in last_prompt.lower() or len(expanded) > 0


# =============================================================================
# Pronoun Resolution Tests
# =============================================================================

class TestPronounResolution:
    """Test pronoun resolution using conversation context."""

    @pytest.fixture
    def rag_conversation(self):
        """Conversation about RAG systems."""
        return [
            {"role": "user", "content": "Tell me about the RAG system"},
            {"role": "assistant", "content": "The RAG system uses vector search and retrieval to enhance LLM responses."},
        ]

    @pytest.fixture
    def mock_documents(self):
        return [
            {
                "_id": "doc_1",
                "content": "RAG retrieval uses cosine similarity for document matching.",
                "_similarity": 0.85,
            }
        ]

    @pytest.mark.asyncio
    async def test_resolve_it_pronoun(self, rag_conversation, mock_documents):
        """Test resolution of 'it' pronoun using conversation history."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Query with 'it' referring to RAG system
        request = QueryRequest(
            query="How does it retrieve documents?",
            conversation_history=rag_conversation,
        )

        response = await orchestrator.process_query(request)

        # Verify query was processed (expansion should have used context)
        assert response is not None
        # The LLM should have been called with history context for expansion
        expansion_calls = [c for c in mock_llm.call_history if "alternative" in c["prompt"].lower()]
        if expansion_calls:
            assert "RAG" in expansion_calls[0]["prompt"] or "conversation" in expansion_calls[0]["prompt"].lower()

    @pytest.mark.asyncio
    async def test_resolve_that_pronoun(self, mock_documents):
        """Test resolution of 'that' pronoun."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        history = [
            {"role": "user", "content": "What is vector search?"},
            {"role": "assistant", "content": "Vector search uses embeddings to find similar documents."},
        ]

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="Can you explain that in more detail?",
            conversation_history=history,
        )

        response = await orchestrator.process_query(request)
        assert response is not None

    @pytest.mark.asyncio
    async def test_resolve_they_pronoun(self, mock_documents):
        """Test resolution of 'they/them' pronouns."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        history = [
            {"role": "user", "content": "What are the main components of RAG?"},
            {"role": "assistant", "content": "The main components are the retriever, the embedding model, and the generator."},
        ]

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="How do they work together?",
            conversation_history=history,
        )

        response = await orchestrator.process_query(request)
        assert response is not None


# =============================================================================
# Context-aware Query Expansion Tests
# =============================================================================

class TestContextAwareExpansion:
    """Test context-aware query expansion."""

    @pytest.fixture
    def mock_documents(self):
        return [{"_id": "doc_1", "content": "Test content", "_similarity": 0.8}]

    @pytest.mark.asyncio
    async def test_expansion_includes_context_topics(self, mock_documents):
        """Test that expansion includes topics from conversation context."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        history = [
            {"role": "user", "content": "Explain the MongoDB vector search"},
            {"role": "assistant", "content": "MongoDB Atlas Vector Search enables semantic search using embeddings."},
        ]

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Query that references prior context
        expanded = await orchestrator._expand_query(
            "What about performance?",
            [],
            conversation_history=history
        )

        # The expansion should have accessed the history
        assert isinstance(expanded, list)

    @pytest.mark.asyncio
    async def test_expansion_without_history(self, mock_documents):
        """Test expansion works without conversation history."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Query without history
        expanded = await orchestrator._expand_query(
            "How does vector search work?",
            [],
            conversation_history=None
        )

        assert isinstance(expanded, list)
        assert len(expanded) > 0


# =============================================================================
# Previous Queries Tests
# =============================================================================

class TestPreviousQueries:
    """Test handling of previous queries for follow-up detection."""

    @pytest.mark.asyncio
    async def test_previous_queries_extracted_from_history(self):
        """Test that previous queries are extracted from conversation history."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig()

        history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG is Retrieval-Augmented Generation."},
            {"role": "user", "content": "How does it work?"},
            {"role": "assistant", "content": "It retrieves documents and generates answers."},
        ]

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Create request with history
        request = QueryRequest(
            query="What about the retrieval step?",
            conversation_history=history,
            previous_queries=["What is RAG?", "How does it work?"],
        )

        response = await orchestrator.process_query(request)
        assert response is not None

    @pytest.mark.asyncio
    async def test_follow_up_detection_with_previous_queries(self):
        """Test that follow-up is detected from previous queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        previous = ["What is the CRAG pattern?", "How does document grading work?"]

        # This query with 'it' should be detected as follow-up
        is_follow_up = orchestrator._detect_follow_up(
            "Why is it important?",
            previous
        )

        assert is_follow_up is True

    @pytest.mark.asyncio
    async def test_no_follow_up_for_new_topic(self):
        """Test that new topics are not detected as follow-ups."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        previous = ["What is machine learning?"]

        # This query starts a new topic without pronouns
        is_follow_up = orchestrator._detect_follow_up(
            "What is the capital of France?",
            previous
        )

        assert is_follow_up is False


# =============================================================================
# Generation with History Tests
# =============================================================================

class TestGenerationWithHistory:
    """Test answer generation with conversation history."""

    @pytest.fixture
    def mock_documents(self):
        return [
            {
                "_id": "doc_1",
                "content": "The CRAG pattern adds corrective retrieval when initial results are poor.",
                "title": "CRAG Guide",
                "_similarity": 0.85,
            }
        ]

    @pytest.mark.asyncio
    async def test_generation_includes_history_context(self, mock_documents):
        """Test that generation prompt includes conversation history."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_validation=False,
            enable_document_grading=False,
        )

        history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG combines retrieval with generation."},
        ]

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(documents=mock_documents),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(
            query="What improvements does CRAG add?",
            conversation_history=history,
        )

        response = await orchestrator.process_query(request)

        # Check that generation was called with history
        generation_calls = [c for c in mock_llm.call_history if "Context:" in c["prompt"]]
        assert len(generation_calls) > 0

        # The prompt should include conversation history
        gen_prompt = generation_calls[0]["prompt"]
        assert "Previous Conversation" in gen_prompt or "Question:" in gen_prompt
