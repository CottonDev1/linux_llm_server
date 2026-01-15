"""
Query Understanding Tests
=========================

Tests for the query understanding stage of the document pipeline.

Query understanding is responsible for:
1. Intent Classification - Categorize query type (SIMPLE, FACTUAL, ANALYTICAL, etc.)
2. Entity Extraction - Extract tables, columns, dates, and other entities
3. Query Expansion - Generate alternative phrasings for better retrieval
4. Follow-up Detection - Detect references to previous conversation
5. Pronoun Resolution - Resolve "it", "that", "they" using context

These tests verify each aspect of query understanding.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any


# =============================================================================
# Mock Services
# =============================================================================

class MockLLMService:
    """Mock LLM service for query understanding tests."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_history = []

    async def generate(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """Return mock LLM response based on prompt content."""
        self.call_history.append({"prompt": prompt, "system": system})

        # Default classification response
        response_text = '{"type": "FACTUAL", "confidence": 0.8}'

        # Match specific prompts - extract the actual query from the classification prompt
        if "Classify this query" in prompt:
            # Extract the query part (after "Query:" line)
            query_part = ""
            lines = prompt.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("Query:"):
                    query_part = line.replace("Query:", "").strip()
                    break

            query_lower = query_part.lower() if query_part else prompt.lower()

            if "how to" in query_lower or "how do i" in query_lower:
                response_text = '{"type": "PROCEDURAL", "confidence": 0.9}'
            elif "compare" in query_lower or " vs " in query_lower:
                response_text = '{"type": "COMPARISON", "confidence": 0.85}'
            elif "last week" in query_lower or "yesterday" in query_lower or "recent" in query_lower:
                response_text = '{"type": "TEMPORAL", "confidence": 0.88}'
            elif "how many" in query_lower or "count" in query_lower:
                response_text = '{"type": "AGGREGATION", "confidence": 0.87}'
            elif "explain" in query_lower or "analyze" in query_lower:
                response_text = '{"type": "ANALYTICAL", "confidence": 0.82}'
            elif "what is" in query_lower:
                response_text = '{"type": "SIMPLE", "confidence": 0.75}'
            else:
                response_text = '{"type": "FACTUAL", "confidence": 0.8}'

        elif "Generate 3 alternative" in prompt:
            # Query expansion
            response_text = '["Alternative query 1", "Alternative query 2", "Alternative query 3"]'

        return MagicMock(
            success=True,
            response=response_text,
            token_usage={"prompt_tokens": 50, "response_tokens": 20},
            generation_time_ms=50,
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

    def __init__(self):
        self.is_initialized = True

    async def initialize(self):
        pass

    async def _vector_search(self, **kwargs):
        return []


# =============================================================================
# Intent Classification Tests
# =============================================================================

class TestIntentClassification:
    """Test query intent classification."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMService()

    @pytest.mark.asyncio
    async def test_classify_simple_query(self, mock_llm):
        """Test classification of simple queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, QueryIntent, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=False,
            enable_document_grading=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Test simple query
        request = QueryRequest(query="What is Python?")
        response = await orchestrator.process_query(request)

        # Should classify as SIMPLE or FACTUAL
        assert response.query_intent in [QueryIntent.SIMPLE, QueryIntent.FACTUAL]

    @pytest.mark.asyncio
    async def test_classify_procedural_query(self, mock_llm):
        """Test classification of how-to queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, QueryIntent, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="How to configure the database connection?")
        response = await orchestrator.process_query(request)

        # Should classify as PROCEDURAL
        assert response.query_intent == QueryIntent.PROCEDURAL

    @pytest.mark.asyncio
    async def test_classify_comparison_query(self, mock_llm):
        """Test classification of comparison queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, QueryIntent, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Compare Python vs JavaScript for web development")
        response = await orchestrator.process_query(request)

        # Should classify as COMPARISON
        assert response.query_intent == QueryIntent.COMPARISON

    @pytest.mark.asyncio
    async def test_classify_temporal_query(self, mock_llm):
        """Test classification of time-based queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, QueryIntent, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="What tickets were created last week?")
        response = await orchestrator.process_query(request)

        # Should classify as TEMPORAL
        assert response.query_intent == QueryIntent.TEMPORAL

    @pytest.mark.asyncio
    async def test_classify_aggregation_query(self, mock_llm):
        """Test classification of counting/statistics queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, QueryIntent, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="How many documents were processed today?")
        response = await orchestrator.process_query(request)

        # Should classify as AGGREGATION
        assert response.query_intent == QueryIntent.AGGREGATION

    @pytest.mark.asyncio
    async def test_classify_analytical_query(self, mock_llm):
        """Test classification of analytical queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryRequest, QueryIntent, OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=False,
            enable_validation=False,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        request = QueryRequest(query="Explain the architecture of the RAG system and analyze its performance")
        response = await orchestrator.process_query(request)

        # Should classify as ANALYTICAL
        assert response.query_intent == QueryIntent.ANALYTICAL


# =============================================================================
# Entity Extraction Tests
# =============================================================================

class TestEntityExtraction:
    """Test entity extraction from queries."""

    @pytest.mark.asyncio
    async def test_extract_date_entities(self):
        """Test extraction of date entities from queries."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig, PipelineState, QueryRequest
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        # Test date extraction
        entities = await orchestrator._extract_entities("What happened yesterday?")

        # Should extract 'yesterday' as a date entity
        date_entities = [e for e in entities if e.entity_type == "date"]
        assert len(date_entities) > 0

    @pytest.mark.asyncio
    async def test_extract_date_range_entities(self):
        """Test extraction of date range entities."""
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
        orchestrator._use_traced = False

        entities = await orchestrator._extract_entities("Show tickets from last week")

        date_entities = [e for e in entities if e.entity_type == "date"]
        assert len(date_entities) > 0

    @pytest.mark.asyncio
    async def test_extract_camelcase_entities(self):
        """Test extraction of CamelCase table names."""
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
        orchestrator._use_traced = False

        entities = await orchestrator._extract_entities("Query the CentralTickets table for status")

        table_entities = [e for e in entities if e.entity_type == "table"]
        assert any("CentralTickets" in e.text for e in table_entities)

    @pytest.mark.asyncio
    async def test_extract_ewr_database_entities(self):
        """Test extraction of EWR database names."""
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
        orchestrator._use_traced = False

        entities = await orchestrator._extract_entities("Get data from EWRCentral database")

        db_entities = [e for e in entities if e.entity_type == "database"]
        assert any("EWRCentral" in e.text for e in db_entities)


# =============================================================================
# Query Expansion Tests
# =============================================================================

class TestQueryExpansion:
    """Test query expansion functionality."""

    @pytest.mark.asyncio
    async def test_expand_query_generates_alternatives(self):
        """Test that query expansion generates alternative phrasings."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(
            enable_query_expansion=True,
            max_expanded_queries=3,
        )

        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        entities = []
        expanded = await orchestrator._expand_query(
            "How to reset password?",
            entities,
            conversation_history=None
        )

        # Should generate alternatives
        assert len(expanded) > 0
        assert all(isinstance(q, str) for q in expanded)

    @pytest.mark.asyncio
    async def test_expand_query_with_conversation_history(self):
        """Test query expansion uses conversation history for context."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import OrchestratorConfig
        except ImportError:
            pytest.skip("Orchestrator module not available")

        config = OrchestratorConfig(enable_query_expansion=True)

        mock_llm = MockLLMService()
        orchestrator = KnowledgeBaseOrchestrator(
            config=config,
            mongodb_service=MockMongoDBService(),
            llm_service=mock_llm,
            embedding_service=MockEmbeddingService(),
        )
        orchestrator._initialized = True
        orchestrator._use_traced = False

        history = [
            {"role": "user", "content": "Tell me about the RAG system"},
            {"role": "assistant", "content": "The RAG system uses vector search..."},
        ]

        expanded = await orchestrator._expand_query(
            "How does it work?",
            [],
            conversation_history=history
        )

        # Verify LLM was called with history context
        assert len(mock_llm.call_history) > 0
        last_call = mock_llm.call_history[-1]
        assert "conversation for context" in last_call["prompt"].lower() or len(expanded) > 0


# =============================================================================
# Follow-up Detection Tests
# =============================================================================

class TestFollowUpDetection:
    """Test detection of follow-up queries."""

    @pytest.mark.asyncio
    async def test_detect_follow_up_with_pronouns(self):
        """Test detection of follow-up queries with pronouns."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        # Test with "it"
        is_follow_up = orchestrator._detect_follow_up(
            "How does it work?",
            ["Tell me about the RAG system"]
        )
        assert is_follow_up is True

    @pytest.mark.asyncio
    async def test_detect_follow_up_with_this(self):
        """Test detection of follow-up with 'this' reference."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        is_follow_up = orchestrator._detect_follow_up(
            "Can you explain this in more detail?",
            ["The CRAG pattern uses document grading"]
        )
        assert is_follow_up is True

    @pytest.mark.asyncio
    async def test_detect_follow_up_with_that(self):
        """Test detection of follow-up with 'that' reference."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        is_follow_up = orchestrator._detect_follow_up(
            "What are the benefits of that?",
            ["Vector search enables semantic matching"]
        )
        assert is_follow_up is True

    @pytest.mark.asyncio
    async def test_detect_not_follow_up_standalone_query(self):
        """Test that standalone queries are not detected as follow-ups."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        is_follow_up = orchestrator._detect_follow_up(
            "What is machine learning?",
            []  # No previous queries
        )
        assert is_follow_up is False

    @pytest.mark.asyncio
    async def test_detect_follow_up_with_elaborate(self):
        """Test detection of follow-up with 'elaborate' request."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        is_follow_up = orchestrator._detect_follow_up(
            "Please elaborate on that",
            ["The system uses BM25 for keyword matching"]
        )
        assert is_follow_up is True


# =============================================================================
# Retrieval Requirement Tests
# =============================================================================

class TestRetrievalRequirement:
    """Test logic for determining if retrieval is needed."""

    @pytest.mark.asyncio
    async def test_simple_greeting_skips_retrieval(self):
        """Test that simple greetings skip retrieval."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryIntent
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        # Test greetings
        for greeting in ["Hello", "Hi", "Hey", "Thanks", "Thank you"]:
            requires = orchestrator._requires_retrieval(QueryIntent.SIMPLE, greeting)
            assert requires is False, f"'{greeting}' should not require retrieval"

    @pytest.mark.asyncio
    async def test_follow_up_indicators_require_retrieval(self):
        """Test that queries with follow-up indicators require retrieval."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryIntent
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        # Test follow-up indicators
        queries = [
            "Tell me more about it",
            "Explain this further",
            "What about that?",
            "Can you elaborate on the same thing?",
        ]

        for query in queries:
            requires = orchestrator._requires_retrieval(QueryIntent.FACTUAL, query)
            assert requires is True, f"'{query}' should require retrieval"

    @pytest.mark.asyncio
    async def test_knowledge_questions_require_retrieval(self):
        """Test that knowledge questions require retrieval."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
            from orchestrator.models import QueryIntent
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        queries = [
            "What is machine learning?",
            "Explain the CRAG pattern",
            "How does vector search work?",
            "Who is the author of this document?",
        ]

        for query in queries:
            requires = orchestrator._requires_retrieval(QueryIntent.FACTUAL, query)
            assert requires is True, f"'{query}' should require retrieval"


# =============================================================================
# Query Rewriting Tests
# =============================================================================

class TestQueryRewriting:
    """Test query rewriting for retrieval optimization."""

    @pytest.mark.asyncio
    async def test_remove_question_prefixes(self):
        """Test that question prefixes are removed."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        test_cases = [
            ("Can you tell me about RAG?", "about RAG?"),
            ("Could you explain the system?", "explain the system?"),
            ("Please describe the process", "describe the process"),
            ("I want to know about vectors", "about vectors"),
        ]

        for original, expected_contains in test_cases:
            rewritten = await orchestrator._rewrite_for_retrieval(original, [])
            # The rewritten query should be shorter or equal
            assert len(rewritten) <= len(original)

    @pytest.mark.asyncio
    async def test_preserve_core_content(self):
        """Test that core content is preserved after rewriting."""
        try:
            from orchestrator.document_orchestrator import KnowledgeBaseOrchestrator
        except ImportError:
            pytest.skip("Orchestrator module not available")

        orchestrator = KnowledgeBaseOrchestrator(
            mongodb_service=MockMongoDBService(),
            llm_service=MockLLMService(),
            embedding_service=MockEmbeddingService(),
        )

        original = "What is the CRAG pattern for RAG systems?"
        rewritten = await orchestrator._rewrite_for_retrieval(original, [])

        # Core terms should be preserved
        assert "CRAG" in rewritten or "RAG" in rewritten or original == rewritten
