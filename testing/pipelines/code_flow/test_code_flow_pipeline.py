"""
Code Flow Pipeline Orchestration Tests
=======================================

Test the main CodeFlowPipeline class including:
- Pipeline initialization and lazy loading
- Full analyze() flow
- Streaming analyze_stream() SSE events
- Caching mechanism
- Error handling
"""

import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fixtures.shared_fixtures import mock_embedding_service, sse_consumer
from utils import generate_test_id


class TestCodeFlowPipelineInitialization:
    """Test pipeline initialization and lazy loading."""

    def test_pipeline_creation_defaults(self):
        """Test pipeline creates with default configuration."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline()

        assert pipeline.mongodb_uri == "mongodb://localhost:27017"
        assert pipeline.llm_endpoint == "http://localhost:8081"
        assert pipeline.cache_enabled is True
        assert pipeline._classifier is None  # Lazy
        assert pipeline._retrieval is None  # Lazy
        assert pipeline._chain_builder is None  # Lazy

    def test_pipeline_creation_custom_config(self):
        """Test pipeline creates with custom configuration."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline(
            mongodb_uri="mongodb://custom:27018",
            llm_endpoint="http://custom:8080",
            cache_enabled=False,
        )

        assert pipeline.mongodb_uri == "mongodb://custom:27018"
        assert pipeline.llm_endpoint == "http://custom:8080"
        assert pipeline.cache_enabled is False

    @pytest.mark.asyncio
    async def test_lazy_service_initialization(self, mock_mongodb_service, mock_llm_service):
        """Test services are initialized on first use."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline()
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        # Services should be None initially (except what we mocked)
        assert pipeline._classifier is None
        assert pipeline._retrieval is None
        assert pipeline._chain_builder is None

        # Trigger lazy initialization
        await pipeline._get_services()

        # Classifier should now be initialized
        assert pipeline._classifier is not None

    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_llm_service):
        """Test pipeline properly cleans up on close."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline()
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        # Mock the close method
        pipeline._llm_service.close = AsyncMock()

        await pipeline.close()

        pipeline._llm_service.close.assert_called_once()


class TestCodeFlowPipelineAnalyze:
    """Test the analyze() method for complete flow execution."""

    @pytest.mark.asyncio
    async def test_analyze_basic_query(
        self,
        mock_mongodb_service,
        mock_llm_service,
        code_flow_test_data,
        sample_bale_processing_scenario,
    ):
        """Test basic code flow analysis."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        # Configure mock MongoDB results
        mock_mongodb_service.set_search_results(
            "business-process", None,
            sample_bale_processing_scenario["business_processes"]
        )
        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_search_results(
            "ui-mapping", None,
            sample_bale_processing_scenario["ui_events"]
        )
        mock_mongodb_service.set_search_results(
            "relationship", "method-call",
            sample_bale_processing_scenario["call_relationships"]
        )

        # Create pipeline with mocks
        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        # Initialize classifier
        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        # Create request
        request = CodeFlowRequest(
            query="How are bales committed to purchase contracts?",
            project="Gin",
        )

        # Analyze
        response = await pipeline.analyze(request)

        # Assertions
        assert response.success is True
        assert response.query == request.query
        assert response.project == "Gin"
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.confidence > 0

    @pytest.mark.asyncio
    async def test_analyze_returns_sources(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
    ):
        """Test that analysis returns proper source attribution."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        # Configure mock results
        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="Find methods that handle bale processing",
            project="Gin",
        )

        response = await pipeline.analyze(request)

        assert response.success is True
        assert response.sources is not None

    @pytest.mark.asyncio
    async def test_analyze_with_project_filter(
        self,
        mock_mongodb_service,
        mock_llm_service,
        code_flow_test_data,
    ):
        """Test analysis respects project filter."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        # Create methods for different projects
        gin_method = code_flow_test_data.create_method_result(
            method_name="GinMethod",
            project="Gin",
        )
        warehouse_method = code_flow_test_data.create_method_result(
            method_name="WarehouseMethod",
            project="Warehouse",
        )

        mock_mongodb_service.set_search_results(
            "code", "method",
            [gin_method, warehouse_method]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        # Request for Gin project only
        request = CodeFlowRequest(
            query="How does data processing work?",
            project="Gin",
        )

        response = await pipeline.analyze(request)

        assert response.success is True
        assert response.project == "Gin"

    @pytest.mark.asyncio
    async def test_analyze_includes_call_chains(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
    ):
        """Test analysis includes call chains when requested."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_search_results(
            "ui-mapping", None,
            sample_bale_processing_scenario["ui_events"]
        )
        mock_mongodb_service.set_search_results(
            "relationship", "method-call",
            sample_bale_processing_scenario["call_relationships"]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="What happens when user clicks commit button?",
            project="Gin",
            include_call_graph=True,
        )

        response = await pipeline.analyze(request)

        assert response.success is True
        # Call chains should be present
        assert response.call_chains is not None

    @pytest.mark.asyncio
    async def test_analyze_without_call_chains(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
    ):
        """Test analysis can skip call chains."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="What methods handle validation?",
            project="Gin",
            include_call_graph=False,
        )

        response = await pipeline.analyze(request)

        assert response.success is True
        # Call chains should be empty when not requested
        assert len(response.call_chains) == 0


class TestCodeFlowPipelineStreaming:
    """Test the analyze_stream() method for SSE streaming."""

    @pytest.mark.asyncio
    async def test_stream_emits_expected_events(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
        sse_collector,
    ):
        """Test streaming emits all expected event types."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="How does bale processing work?",
            project="Gin",
            include_call_graph=False,
        )

        # Collect all events
        events = []
        async for event in pipeline.analyze_stream(request):
            events.append(event)

        # Verify event sequence
        event_types = [e.event for e in events]

        assert "status" in event_types  # Progress events
        assert "classification" in event_types
        assert "retrieval" in event_types
        assert "synthesis" in event_types
        assert "result" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_stream_classification_event_content(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
    ):
        """Test classification event contains expected data."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="How does the workflow process orders?",
            project="Warehouse",
        )

        classification_event = None
        async for event in pipeline.analyze_stream(request):
            if event.event == "classification":
                classification_event = event
                break

        assert classification_event is not None
        data = json.loads(classification_event.data)
        assert "type" in data
        assert "confidence" in data
        assert "patterns" in data

    @pytest.mark.asyncio
    async def test_stream_retrieval_event_content(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
    ):
        """Test retrieval event contains result counts."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_search_results(
            "code", "class",
            []
        )
        mock_mongodb_service.set_search_results(
            "ui-mapping", None,
            sample_bale_processing_scenario["ui_events"]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="How are bales processed?",
            project="Gin",
        )

        retrieval_event = None
        async for event in pipeline.analyze_stream(request):
            if event.event == "retrieval":
                retrieval_event = event
                break

        assert retrieval_event is not None
        data = json.loads(retrieval_event.data)
        assert "total_results" in data
        assert "methods" in data
        assert "classes" in data
        assert "ui_events" in data

    @pytest.mark.asyncio
    async def test_stream_error_handling(
        self,
        mock_mongodb_service,
        mock_llm_service,
    ):
        """Test stream handles errors gracefully."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        # Configure LLM to fail
        mock_llm_service.set_failure(True, "Test LLM failure")

        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="Test query",
            project="Test",
        )

        events = []
        async for event in pipeline.analyze_stream(request):
            events.append(event)

        # Should still complete with some result (fallback)
        event_types = [e.event for e in events]
        assert "done" in event_types or "error" in event_types

    @pytest.mark.asyncio
    async def test_stream_progress_events(
        self,
        mock_mongodb_service,
        mock_llm_service,
    ):
        """Test stream includes progress events with step counts."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="What methods exist?",
            project="Test",
            include_call_graph=False,
        )

        progress_events = []
        async for event in pipeline.analyze_stream(request):
            if event.event == "status":
                progress_events.append(json.loads(event.data))

        # Should have multiple progress updates
        assert len(progress_events) >= 3

        # Check progress event structure
        for progress in progress_events:
            assert "stage" in progress
            assert "step" in progress
            assert "totalSteps" in progress
            assert "elapsed" in progress


class TestCodeFlowPipelineCaching:
    """Test the caching mechanism."""

    @pytest.mark.asyncio
    async def test_cache_returns_cached_response(
        self,
        mock_mongodb_service,
        mock_llm_service,
        code_flow_test_data,
    ):
        """Test that identical queries return cached responses."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            [code_flow_test_data.create_method_result("TestMethod")]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=True)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="What methods handle data?",
            project="Test",
            include_call_graph=False,
        )

        # First call - not cached
        response1 = await pipeline.analyze(request)
        assert response1.cached is False

        # Second call - should be cached
        response2 = await pipeline.analyze(request)
        assert response2.cached is True

        # Responses should be equivalent
        assert response1.query == response2.query

    @pytest.mark.asyncio
    async def test_cache_key_includes_project(
        self,
        mock_mongodb_service,
        mock_llm_service,
        code_flow_test_data,
    ):
        """Test that different projects have different cache entries."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            [code_flow_test_data.create_method_result("TestMethod")]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=True)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        # Same query, different projects
        request1 = CodeFlowRequest(
            query="What methods handle data?",
            project="Project1",
            include_call_graph=False,
        )
        request2 = CodeFlowRequest(
            query="What methods handle data?",
            project="Project2",
            include_call_graph=False,
        )

        response1 = await pipeline.analyze(request1)
        response2 = await pipeline.analyze(request2)

        # Both should not be cached (different projects)
        assert response1.cached is False
        assert response2.cached is False

    @pytest.mark.asyncio
    async def test_cache_disabled(
        self,
        mock_mongodb_service,
        mock_llm_service,
        code_flow_test_data,
    ):
        """Test that caching can be disabled."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "code", "method",
            [code_flow_test_data.create_method_result("TestMethod")]
        )
        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="What methods handle data?",
            project="Test",
            include_call_graph=False,
        )

        # Both calls should not be cached
        response1 = await pipeline.analyze(request)
        response2 = await pipeline.analyze(request)

        assert response1.cached is False
        assert response2.cached is False

    def test_clear_cache(self, mock_mongodb_service, mock_llm_service):
        """Test cache clearing."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline(cache_enabled=True)

        # Manually populate cache
        pipeline._cache["test_key"] = (MagicMock(), time.time())

        assert len(pipeline._cache) == 1

        pipeline.clear_cache()

        assert len(pipeline._cache) == 0


class TestCodeFlowPipelineFallback:
    """Test fallback behavior when LLM fails."""

    @pytest.mark.asyncio
    async def test_fallback_answer_on_llm_failure(
        self,
        mock_mongodb_service,
        mock_llm_service,
        sample_bale_processing_scenario,
    ):
        """Test fallback answer generation when LLM fails."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_search_results(
            "business-process", None,
            sample_bale_processing_scenario["business_processes"]
        )
        mock_mongodb_service.set_search_results(
            "code", "method",
            sample_bale_processing_scenario["methods"]
        )
        mock_mongodb_service.set_default_results([])

        # Configure LLM to fail
        mock_llm_service.set_failure(True, "LLM unavailable")

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="How are bales committed?",
            project="Gin",
        )

        # Should still return a response with fallback answer
        response = await pipeline.analyze(request)

        assert response.success is True
        assert response.answer is not None
        assert len(response.answer) > 0
        # Fallback should include method names from results
        assert "Bale" in response.answer or "Method" in response.answer


class TestCodeFlowPipelineProcessingTime:
    """Test processing time tracking."""

    @pytest.mark.asyncio
    async def test_processing_time_tracked(
        self,
        mock_mongodb_service,
        mock_llm_service,
    ):
        """Test that processing time is tracked."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline
        from code_flow_pipeline.models.query_models import CodeFlowRequest

        mock_mongodb_service.set_default_results([])

        pipeline = CodeFlowPipeline(cache_enabled=False)
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._llm_service = mock_llm_service
        pipeline._use_traced = False

        from code_flow_pipeline.services.query_classifier import get_query_classifier
        pipeline._classifier = get_query_classifier()

        request = CodeFlowRequest(
            query="What methods exist?",
            project="Test",
        )

        response = await pipeline.analyze(request)

        assert response.processing_time > 0
        assert isinstance(response.processing_time, float)
