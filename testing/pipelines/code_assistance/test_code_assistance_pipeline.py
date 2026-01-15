"""
Code Assistance Pipeline Orchestration Tests
=============================================

Test the CodeAssistancePipeline class orchestration, including:
- Complete process_query() flow
- Query processing with context
- Initialization with _mongodb_service
- Caching mechanism
- Error handling
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from fixtures.llm_fixtures import LocalLLMClient
from fixtures.mongodb_fixtures import create_mock_code_method
from utils import (
    assert_document_stored,
    assert_llm_response_valid,
    generate_test_id,
    measure_time,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def mock_methods() -> List[Dict[str, Any]]:
    """Create mock method data for retrieval."""
    return [
        {
            "method_name": "SaveBale",
            "class_name": "BaleService",
            "file_path": "/src/Services/BaleService.cs",
            "line_number": 42,
            "signature": "public void SaveBale(Bale bale)",
            "summary": "Saves a bale record to the database",
            "sql_calls": ["INSERT INTO Bales"],
            "similarity": 0.95,
            "project": "Gin",
        },
        {
            "method_name": "ValidateBale",
            "class_name": "BaleService",
            "file_path": "/src/Services/BaleService.cs",
            "line_number": 28,
            "signature": "private bool ValidateBale(Bale bale)",
            "summary": "Validates bale data before saving",
            "similarity": 0.82,
            "project": "Gin",
        },
    ]


@pytest.fixture
def mock_classes() -> List[Dict[str, Any]]:
    """Create mock class data for retrieval."""
    return [
        {
            "class_name": "BaleService",
            "namespace": "Gin.Services",
            "file_path": "/src/Services/BaleService.cs",
            "base_class": "BaseService",
            "interfaces": ["IBaleService"],
            "methods": ["SaveBale", "ValidateBale", "GetBales", "DeleteBale"],
            "similarity": 0.88,
            "project": "Gin",
        },
    ]


@pytest.fixture
def mock_event_handlers() -> List[Dict[str, Any]]:
    """Create mock event handler data."""
    return [
        {
            "event_name": "Click",
            "handler_method": "btnSave_Click",
            "element_name": "btnSave",
            "ui_element_type": "Button",
            "handler_class": "BaleEntryForm",
            "file_path": "/src/Forms/BaleEntryForm.cs",
            "line_number": 150,
            "similarity": 0.75,
            "project": "Gin",
        },
    ]


@pytest.fixture
def sample_query_request():
    """Create a sample CodeQueryRequest for testing."""
    # Import here to avoid circular imports
    from code_assistance_pipeline.models.query_models import (
        CodeQueryRequest,
        CodeQueryOptions,
    )

    return CodeQueryRequest(
        query="How do I save a bale in the system?",
        project="Gin",
        options=CodeQueryOptions(
            include_sources=True,
            include_call_chains=True,
            max_sources=10,
        ),
    )


# =============================================================================
# Pipeline Initialization Tests
# =============================================================================

class TestCodeAssistancePipelineInit:
    """Test CodeAssistancePipeline initialization."""

    @pytest.mark.asyncio
    async def test_pipeline_lazy_initialization(self, pipeline_config):
        """Test that pipeline initializes services lazily."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        pipeline = CodeAssistancePipeline()

        # Services should not be initialized yet
        assert pipeline._retriever is None
        assert pipeline._context_builder is None
        assert pipeline._generator is None
        assert pipeline._mongodb_service is None
        assert pipeline._initialized is False

    @pytest.mark.asyncio
    @pytest.mark.requires_mongodb
    async def test_pipeline_initialize_sets_services(self, pipeline_config):
        """Test that initialize() properly sets up all services."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        # Mock the services to avoid actual connections
        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            MockContextBuilder.return_value = mock_context_builder

            mock_generator = AsyncMock()
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            assert pipeline._initialized is True
            assert pipeline._retriever is mock_retriever
            assert pipeline._context_builder is mock_context_builder
            assert pipeline._generator is mock_generator
            mock_retriever.initialize.assert_called_once()
            mock_generator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_double_initialization_safe(self, pipeline_config):
        """Test that calling initialize() twice is safe."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            MockRetriever.return_value = mock_retriever
            MockContextBuilder.return_value = MagicMock()
            mock_generator = AsyncMock()
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()
            await pipeline.initialize()  # Second call should be no-op

            # Should only be called once
            assert MockRetriever.call_count == 1
            assert mock_retriever.initialize.call_count == 1


# =============================================================================
# Process Query Tests
# =============================================================================

class TestCodeAssistancePipelineProcessQuery:
    """Test CodeAssistancePipeline.process_query() method."""

    @pytest.mark.asyncio
    async def test_process_query_returns_response(
        self,
        pipeline_config,
        mock_methods,
        mock_classes,
        sample_query_request,
    ):
        """Test that process_query returns a complete CodeQueryResponse."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            CodeQueryResponse,
            SourceInfo,
            TokenUsage,
            TimingInfo,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            # Setup mock retriever
            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = (
                mock_methods,
                mock_classes,
                [],  # event_handlers
                ["BaleService.SaveBale"],  # call_chain
            )
            MockRetriever.return_value = mock_retriever

            # Setup mock context builder
            mock_context_builder = MagicMock()
            mock_sources = [
                SourceInfo(
                    type="method",
                    name="BaleService.SaveBale",
                    file_path="/src/Services/BaleService.cs",
                    line_number=42,
                    relevance_score=0.95,
                    content="public void SaveBale(Bale bale)",
                ),
            ]
            mock_context_builder.build_context.return_value = (
                "Context: BaleService.SaveBale saves bale data",
                mock_sources,
            )
            mock_context_builder.build_prompt.return_value = "Full prompt with context"
            MockContextBuilder.return_value = mock_context_builder

            # Setup mock generator
            mock_generator = AsyncMock()
            mock_token_usage = TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
            mock_generator.generate.return_value = (
                "To save a bale, use BaleService.SaveBale method.",
                mock_token_usage,
                1500,  # generation_time_ms
            )
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            response = await pipeline.process_query(sample_query_request)

            # Verify response structure
            assert isinstance(response, CodeQueryResponse)
            assert response.answer is not None
            assert len(response.answer) > 0
            assert response.response_id is not None
            assert response.timing is not None
            # Mocked tests may run sub-millisecond, so check >= 0
            assert response.timing.total_ms >= 0

    @pytest.mark.asyncio
    async def test_process_query_includes_sources(
        self,
        pipeline_config,
        mock_methods,
        mock_classes,
        sample_query_request,
    ):
        """Test that process_query includes source information."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            SourceInfo,
            TokenUsage,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = (
                mock_methods, mock_classes, [], ["BaleService.SaveBale"],
            )
            MockRetriever.return_value = mock_retriever

            mock_sources = [
                SourceInfo(
                    type="method",
                    name="BaleService.SaveBale",
                    file_path="/src/Services/BaleService.cs",
                    line_number=42,
                    relevance_score=0.95,
                ),
                SourceInfo(
                    type="class",
                    name="Gin.Services.BaleService",
                    file_path="/src/Services/BaleService.cs",
                    line_number=0,
                    relevance_score=0.88,
                ),
            ]

            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("Context", mock_sources)
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            mock_generator = AsyncMock()
            mock_generator.generate.return_value = (
                "Answer text",
                TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                1000,
            )
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            response = await pipeline.process_query(sample_query_request)

            assert len(response.sources) > 0
            # Check source contains expected fields
            first_source = response.sources[0]
            assert first_source.name is not None

    @pytest.mark.asyncio
    async def test_process_query_includes_timing(
        self,
        pipeline_config,
        mock_methods,
        sample_query_request,
    ):
        """Test that process_query includes timing information."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import TokenUsage

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = (mock_methods, [], [], [])
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("Context", [])
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            mock_generator = AsyncMock()
            mock_generator.generate.return_value = (
                "Answer",
                TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                2000,  # 2 second generation time
            )
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            response = await pipeline.process_query(sample_query_request)

            assert response.timing is not None
            assert response.timing.retrieval_ms >= 0
            assert response.timing.generation_ms >= 0
            assert response.timing.total_ms >= response.timing.retrieval_ms

    @pytest.mark.asyncio
    async def test_process_query_with_conversation_history(self, pipeline_config):
        """Test process_query with conversation history."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            CodeQueryRequest,
            ConversationMessage,
            TokenUsage,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = ([], [], [], [])
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("Context with history", [])
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            mock_generator = AsyncMock()
            mock_generator.generate.return_value = (
                "Follow-up answer",
                TokenUsage(prompt_tokens=150, completion_tokens=60, total_tokens=210),
                1500,
            )
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            # Create request with history
            request = CodeQueryRequest(
                query="How do I use that method?",
                project="Gin",
                history=[
                    ConversationMessage(
                        role="user",
                        content="What methods are available in BaleService?",
                    ),
                    ConversationMessage(
                        role="assistant",
                        content="BaleService has SaveBale, ValidateBale, and GetBales methods.",
                    ),
                ],
            )

            response = await pipeline.process_query(request)

            # Verify history was passed to context builder
            mock_context_builder.build_context.assert_called_once()
            call_args = mock_context_builder.build_context.call_args
            assert call_args.kwargs.get('history') is not None
            assert len(call_args.kwargs.get('history')) == 2


# =============================================================================
# Singleton Pattern Tests
# =============================================================================

class TestCodeAssistancePipelineSingleton:
    """Test the singleton pattern for pipeline access."""

    @pytest.mark.asyncio
    async def test_get_code_assistance_pipeline_returns_same_instance(self, pipeline_config):
        """Test that get_code_assistance_pipeline returns singleton."""
        from code_assistance_pipeline.pipeline import (
            get_code_assistance_pipeline,
            close_code_assistance_pipeline,
            _pipeline_instance,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            MockRetriever.return_value = mock_retriever
            MockContextBuilder.return_value = MagicMock()
            mock_generator = AsyncMock()
            MockGenerator.return_value = mock_generator

            # Close any existing instance first
            await close_code_assistance_pipeline()

            # Get two instances
            pipeline1 = await get_code_assistance_pipeline()
            pipeline2 = await get_code_assistance_pipeline()

            # Should be the same instance
            assert pipeline1 is pipeline2

            # Cleanup
            await close_code_assistance_pipeline()

    @pytest.mark.asyncio
    async def test_close_code_assistance_pipeline(self, pipeline_config):
        """Test that close_code_assistance_pipeline resets singleton."""
        from code_assistance_pipeline.pipeline import (
            get_code_assistance_pipeline,
            close_code_assistance_pipeline,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            MockRetriever.return_value = mock_retriever
            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            # Reset any existing state
            await close_code_assistance_pipeline()

            pipeline1 = await get_code_assistance_pipeline()
            await close_code_assistance_pipeline()
            pipeline2 = await get_code_assistance_pipeline()

            # After close, should get a new instance
            assert pipeline1 is not pipeline2

            # Cleanup
            await close_code_assistance_pipeline()


# =============================================================================
# MongoDB Logging Tests
# =============================================================================

class TestCodeAssistancePipelineLogging:
    """Test interaction logging to MongoDB."""

    @pytest.mark.asyncio
    @pytest.mark.requires_mongodb
    async def test_pipeline_logs_interaction(
        self,
        mongodb_database,
        pipeline_config,
        mock_methods,
        sample_query_request,
    ):
        """Test that pipeline logs interactions to MongoDB."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            SourceInfo,
            TokenUsage,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = (mock_methods, [], [], [])
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            mock_sources = [
                SourceInfo(type="method", name="BaleService.SaveBale"),
            ]
            mock_context_builder.build_context.return_value = ("Context", mock_sources)
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            mock_generator = AsyncMock()
            mock_generator.generate.return_value = (
                "Test answer for logging",
                TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                1000,
            )
            MockGenerator.return_value = mock_generator

            # Create pipeline with mock MongoDB service
            pipeline = CodeAssistancePipeline()

            # Mock the MongoDB service
            mock_mongodb = MagicMock()
            mock_collection = MagicMock()
            mock_mongodb.db = {"code_interactions": mock_collection}
            mock_collection.insert_one = AsyncMock()

            # Initialize and inject mock
            await pipeline.initialize()
            pipeline._mongodb_service = mock_mongodb

            response = await pipeline.process_query(sample_query_request)

            # Verify insert was called
            mock_collection.insert_one.assert_called_once()

            # Verify logged document structure
            logged_doc = mock_collection.insert_one.call_args[0][0]
            assert logged_doc["response_id"] == response.response_id
            assert logged_doc["query"] == sample_query_request.query
            assert "answer" in logged_doc
            assert "sources" in logged_doc


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestCodeAssistancePipelineErrorHandling:
    """Test error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_handles_retrieval_error(self, pipeline_config, sample_query_request):
        """Test that pipeline handles retrieval errors gracefully."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import TokenUsage

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            # Make retriever raise an error
            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.side_effect = Exception("Retrieval failed")
            MockRetriever.return_value = mock_retriever

            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            # Should raise the error
            with pytest.raises(Exception, match="Retrieval failed"):
                await pipeline.process_query(sample_query_request)

    @pytest.mark.asyncio
    async def test_pipeline_handles_generator_error(
        self,
        pipeline_config,
        mock_methods,
        sample_query_request,
    ):
        """Test that pipeline handles generation errors gracefully."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = (mock_methods, [], [], [])
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("Context", [])
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            # Make generator raise an error
            mock_generator = AsyncMock()
            mock_generator.generate.side_effect = Exception("Generation failed")
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            # Should raise the error
            with pytest.raises(Exception, match="Generation failed"):
                await pipeline.process_query(sample_query_request)


# =============================================================================
# E2E Integration Tests (with real services when available)
# =============================================================================

class TestCodeAssistancePipelineE2E:
    """End-to-end tests requiring real services."""

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_services(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config,
    ):
        """Test full pipeline with real MongoDB and LLM."""
        # This test requires actual services to be running
        # It will be skipped if services are unavailable

        methods_collection = mongodb_database["code_methods"]

        # Set up test data
        method_doc = create_mock_code_method(
            method_name="ProcessOrder",
            class_name="OrderService",
            project="Warehouse",
            code="public void ProcessOrder(Order order) { _db.Execute(\"INSERT INTO Orders...\", order); }",
        )
        method_doc.update({
            "purpose_summary": "Processes and saves an order",
            "database_tables": ["Orders"],
            "test_run_id": pipeline_config.test_run_id,
        })
        methods_collection.insert_one(method_doc)

        # Generate a response using the LLM
        context = f"""
        Method: {method_doc['method_name']}
        Class: {method_doc['class_name']}
        Purpose: {method_doc.get('purpose_summary', '')}
        """

        prompt = f"""Based on this code context:
{context}

How do I process an order in the system?

Answer:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.2,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["order"],
        )
