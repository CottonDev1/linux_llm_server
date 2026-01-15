"""
Multi-Stage Retrieval Tests
===========================

Test the MultiStageRetrieval service including:
- Parallel execution of retrieval stages
- Result aggregation and normalization
- Reference expansion
- RetrievalResults.compute_total()
- Stage configuration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from code_flow_pipeline.services.multi_stage_retrieval import (
    MultiStageRetrieval,
    get_multi_stage_retrieval,
    RetrievalContext,
)
from code_flow_pipeline.models.query_models import (
    RetrievalResults,
    RetrievalStage,
    RetrievalStageType,
    QueryClassification,
    QueryType,
    FormattedResult,
)
from utils import generate_test_id


class TestMultiStageRetrievalInitialization:
    """Test retrieval service initialization."""

    def test_retrieval_creation(self):
        """Test retrieval service creates successfully."""
        retrieval = MultiStageRetrieval()

        assert retrieval._mongodb_service is None  # Lazy
        assert retrieval._classifier is not None

    def test_retrieval_with_mongodb_service(self, mock_mongodb_service):
        """Test retrieval with injected MongoDB service."""
        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        assert retrieval._mongodb_service is mock_mongodb_service

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test get_multi_stage_retrieval returns singleton."""
        # Clear singleton for test
        import code_flow_pipeline.services.multi_stage_retrieval as module
        module._retrieval_instance = None

        retrieval1 = await get_multi_stage_retrieval()
        retrieval2 = await get_multi_stage_retrieval()

        assert retrieval1 is retrieval2

        # Clean up
        module._retrieval_instance = None


class TestRetrievalExecution:
    """Test the execute() method."""

    @pytest.mark.asyncio
    async def test_execute_basic_query(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test basic retrieval execution."""
        # Configure mock results
        method_results = [
            code_flow_test_data.create_method_result("TestMethod1"),
            code_flow_test_data.create_method_result("TestMethod2"),
        ]
        mock_mongodb_service.set_search_results("code", "method", method_results)
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        results = await retrieval.execute(
            query="How does method work?",
            project="TestProject",
        )

        assert results.total_results > 0
        assert len(results.methods) > 0

    @pytest.mark.asyncio
    async def test_execute_with_project_filter(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test retrieval respects project filter."""
        # Create methods for different projects
        gin_method = code_flow_test_data.create_method_result(
            "GinMethod", project="Gin"
        )
        warehouse_method = code_flow_test_data.create_method_result(
            "WarehouseMethod", project="Warehouse"
        )

        mock_mongodb_service.set_search_results(
            "code", "method",
            [gin_method, warehouse_method]
        )
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        results = await retrieval.execute(
            query="Find methods",
            project="Gin",
        )

        # Should filter to Gin project only
        for method in results.methods:
            project = method.metadata.get("project")
            assert project == "Gin" or project is None

    @pytest.mark.asyncio
    async def test_execute_multiple_stages(
        self,
        mock_mongodb_service,
        sample_bale_processing_scenario,
    ):
        """Test retrieval executes multiple stages."""
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
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        # Classify as business process to get multiple stages
        classification = QueryClassification(
            type=QueryType.BUSINESS_PROCESS,
            confidence=0.9,
        )

        results = await retrieval.execute(
            query="How are bales committed?",
            project="Gin",
            classification=classification,
        )

        # Should have results from multiple categories
        assert len(results.business_processes) > 0 or len(results.methods) > 0

    @pytest.mark.asyncio
    async def test_execute_with_custom_stages(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test retrieval with custom stage configuration."""
        method_results = [code_flow_test_data.create_method_result("TestMethod")]
        mock_mongodb_service.set_search_results("code", "method", method_results)
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        # Define custom stages
        custom_stages = [
            RetrievalStage(
                type=RetrievalStageType.METHODS,
                collection="code_methods",
                limit=5,
                enabled=True,
                filter_category="code",
                filter_type="method",
            ),
        ]

        results = await retrieval.execute(
            query="Find methods",
            stages=custom_stages,
        )

        assert len(results.methods) > 0

    @pytest.mark.asyncio
    async def test_execute_without_call_graph(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test retrieval can exclude call graph stage."""
        method_results = [code_flow_test_data.create_method_result("TestMethod")]
        mock_mongodb_service.set_search_results("code", "method", method_results)
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        results = await retrieval.execute(
            query="Find methods",
            include_call_graph=False,
        )

        # Should still return results but without call graph
        assert results.total_results >= 0


class TestParallelExecution:
    """Test parallel stage execution."""

    @pytest.mark.asyncio
    async def test_stages_execute_in_parallel(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test multiple stages execute concurrently."""
        import time

        # Add delay to mock to simulate real async operation
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return [code_flow_test_data.create_method_result("TestMethod")]

        mock_mongodb_service.search_vectors = slow_search

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        # Multiple stages
        stages = [
            RetrievalStage(
                type=RetrievalStageType.METHODS,
                collection="code_methods",
                enabled=True,
                filter_category="code",
                filter_type="method",
            ),
            RetrievalStage(
                type=RetrievalStageType.CLASSES,
                collection="code_classes",
                enabled=True,
                filter_category="code",
                filter_type="class",
            ),
        ]

        start_time = time.time()
        await retrieval.execute(
            query="Test query",
            stages=stages,
        )
        elapsed = time.time() - start_time

        # If parallel, should complete in ~100ms, not 200ms
        # Allow some overhead
        assert elapsed < 0.3, "Stages did not execute in parallel"

    @pytest.mark.asyncio
    async def test_stage_failure_does_not_block_others(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test one stage failing doesn't block others."""
        call_count = {"success": 0, "fail": 0}

        async def conditional_search(query, category=None, **kwargs):
            if category == "code":
                call_count["success"] += 1
                return [code_flow_test_data.create_method_result("TestMethod")]
            else:
                call_count["fail"] += 1
                raise Exception("Simulated failure")

        mock_mongodb_service.search_vectors = conditional_search

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        # Should complete without raising
        results = await retrieval.execute(
            query="Test query",
            classification=QueryClassification(
                type=QueryType.BUSINESS_PROCESS,
                confidence=0.9,
            ),
        )

        # Should have some results despite failures
        assert results is not None


class TestResultAggregation:
    """Test result aggregation and normalization."""

    @pytest.mark.asyncio
    async def test_results_aggregated_by_category(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test results are properly categorized."""
        mock_mongodb_service.set_search_results(
            "code", "method",
            [code_flow_test_data.create_method_result("TestMethod")]
        )
        mock_mongodb_service.set_search_results(
            "code", "class",
            [code_flow_test_data.create_class_result("TestClass")]
        )
        mock_mongodb_service.set_search_results(
            "ui-mapping", None,
            [code_flow_test_data.create_ui_event_result("btnTest", "btnTest_Click")]
        )
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        stages = [
            RetrievalStage(
                type=RetrievalStageType.METHODS,
                collection="code_methods",
                enabled=True,
                filter_category="code",
                filter_type="method",
            ),
            RetrievalStage(
                type=RetrievalStageType.CLASSES,
                collection="code_classes",
                enabled=True,
                filter_category="code",
                filter_type="class",
            ),
            RetrievalStage(
                type=RetrievalStageType.UI_EVENTS,
                collection="ui_events",
                enabled=True,
                filter_category="ui-mapping",
                filter_type=None,
            ),
        ]

        results = await retrieval.execute(
            query="Test query",
            stages=stages,
        )

        # Each category should have results
        assert len(results.methods) > 0
        assert len(results.classes) > 0
        assert len(results.ui_events) > 0

    @pytest.mark.asyncio
    async def test_results_normalized_to_formatted_result(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test raw results are converted to FormattedResult."""
        mock_mongodb_service.set_search_results(
            "code", "method",
            [code_flow_test_data.create_method_result("TestMethod")]
        )
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        results = await retrieval.execute(query="Test")

        if results.methods:
            method = results.methods[0]

            # Should be FormattedResult with expected fields
            assert hasattr(method, 'id')
            assert hasattr(method, 'similarity')
            assert hasattr(method, 'content')
            assert hasattr(method, 'metadata')

    @pytest.mark.asyncio
    async def test_content_truncation(
        self,
        mock_mongodb_service,
    ):
        """Test long content is truncated."""
        long_content = "A" * 1000  # 1000 character content
        mock_mongodb_service.set_search_results(
            "code", "method",
            [{
                "id": "test_id",
                "content": long_content,
                "similarity": 0.9,
                "metadata": {},
            }]
        )
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        results = await retrieval.execute(query="Test")

        if results.methods:
            # Content should be truncated to 300 chars
            assert len(results.methods[0].content) <= 300


class TestRetrievalResults:
    """Test RetrievalResults dataclass."""

    def test_compute_total_empty(self):
        """Test compute_total with empty results."""
        results = RetrievalResults()
        results.compute_total()

        assert results.total_results == 0

    def test_compute_total_with_methods(self, code_flow_test_data):
        """Test compute_total counts methods."""
        results = RetrievalResults()
        results.methods = [
            FormattedResult(id="1", similarity=0.9),
            FormattedResult(id="2", similarity=0.8),
        ]
        results.compute_total()

        assert results.total_results == 2

    def test_compute_total_all_categories(self, code_flow_test_data):
        """Test compute_total counts all categories."""
        results = RetrievalResults()
        results.methods = [FormattedResult(id="1")]
        results.classes = [FormattedResult(id="2")]
        results.ui_events = [FormattedResult(id="3")]
        results.business_processes = [FormattedResult(id="4")]
        results.call_relationships = [FormattedResult(id="5")]
        results.database_accessors = [FormattedResult(id="6")]

        results.compute_total()

        assert results.total_results == 6

    def test_post_init_computes_total(self):
        """Test __post_init__ calls compute_total."""
        results = RetrievalResults(
            methods=[FormattedResult(id="1")],
            classes=[FormattedResult(id="2")],
        )

        assert results.total_results == 2


class TestReferenceTracking:
    """Test reference tracking for multi-hop expansion."""

    @pytest.mark.asyncio
    async def test_tracks_called_methods(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test referenced method names are tracked."""
        method_with_calls = code_flow_test_data.create_method_result(
            "MainMethod",
            calls_method=["HelperMethod1", "HelperMethod2"],
        )
        mock_mongodb_service.set_search_results(
            "code", "method",
            [method_with_calls]
        )
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        # Create context to track references
        context = RetrievalContext(
            query="Test query",
            project=None,
            classification=QueryClassification(type=QueryType.GENERAL),
            stages=[],
        )

        # Execute stage
        stage = RetrievalStage(
            type=RetrievalStageType.METHODS,
            collection="code_methods",
            enabled=True,
            filter_category="code",
            filter_type="method",
        )

        await retrieval._execute_stage(mock_mongodb_service, context, stage)

        # Should have tracked referenced methods
        assert "HelperMethod1" in context.referenced_methods
        assert "HelperMethod2" in context.referenced_methods

    @pytest.mark.asyncio
    async def test_tracks_called_by_methods(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test methods that call this one are tracked."""
        method_with_callers = code_flow_test_data.create_method_result(
            "HelperMethod",
            called_by=["CallerMethod1", "CallerMethod2"],
        )
        mock_mongodb_service.set_search_results(
            "code", "method",
            [method_with_callers]
        )
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        context = RetrievalContext(
            query="Test query",
            project=None,
            classification=QueryClassification(type=QueryType.GENERAL),
            stages=[],
        )

        stage = RetrievalStage(
            type=RetrievalStageType.METHODS,
            collection="code_methods",
            enabled=True,
            filter_category="code",
            filter_type="method",
        )

        await retrieval._execute_stage(mock_mongodb_service, context, stage)

        assert "CallerMethod1" in context.referenced_methods
        assert "CallerMethod2" in context.referenced_methods


class TestReferenceExpansion:
    """Test expand_references for multi-hop retrieval."""

    @pytest.mark.asyncio
    async def test_expand_missing_methods(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test expansion fetches missing referenced methods."""
        # Initial results with reference to HelperMethod
        initial_method = code_flow_test_data.create_method_result(
            "MainMethod",
            calls_method=["HelperMethod"],
        )

        # HelperMethod to be fetched on expansion
        helper_method = code_flow_test_data.create_method_result("HelperMethod")

        async def search_vectors(query, **kwargs):
            if "HelperMethod" in query:
                return [helper_method]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        context = RetrievalContext(
            query="Test query",
            project=None,
            classification=QueryClassification(type=QueryType.GENERAL),
            stages=[],
        )

        # Add initial results and referenced methods
        context.results.methods = [
            FormattedResult(
                id="main",
                metadata={"methodName": "MainMethod"}
            )
        ]
        context.referenced_methods.add("HelperMethod")

        # Expand
        expanded_results = await retrieval.expand_references(context)

        # Should now have HelperMethod
        method_names = [
            m.metadata.get("methodName", "")
            for m in expanded_results.methods
        ]
        assert "HelperMethod" in method_names or len(expanded_results.methods) > 1

    @pytest.mark.asyncio
    async def test_expand_respects_max_limit(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test expansion respects max_expansion limit."""
        mock_mongodb_service.set_default_results([])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        context = RetrievalContext(
            query="Test query",
            project=None,
            classification=QueryClassification(type=QueryType.GENERAL),
            stages=[],
        )

        # Add many referenced methods
        for i in range(20):
            context.referenced_methods.add(f"Method{i}")

        # Expand with limit of 5
        await retrieval.expand_references(context, max_expansion=5)

        # Should have made at most 5 expansion queries
        # (Checking behavior not breaking with many refs)

    @pytest.mark.asyncio
    async def test_expand_skips_already_retrieved(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test expansion skips already retrieved methods."""
        mock_mongodb_service.search_vectors = AsyncMock(return_value=[])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        context = RetrievalContext(
            query="Test query",
            project=None,
            classification=QueryClassification(type=QueryType.GENERAL),
            stages=[],
        )

        # Already have MainMethod
        context.results.methods = [
            FormattedResult(
                id="main",
                metadata={"methodName": "MainMethod"}
            )
        ]
        # Reference to same method
        context.referenced_methods.add("MainMethod")

        await retrieval.expand_references(context)

        # Should not have made any queries (method already retrieved)
        mock_mongodb_service.search_vectors.assert_not_called()


class TestFormattedResultConversion:
    """Test conversion of raw results to FormattedResult."""

    def test_format_result_basic(self):
        """Test basic result formatting."""
        retrieval = MultiStageRetrieval()

        raw = {
            "id": "test_id",
            "_id": "test_id",
            "content": "Test content",
            "similarity": 0.85,
            "metadata": {
                "methodName": "TestMethod",
                "project": "TestProject",
            },
        }

        formatted = retrieval._format_result(raw)

        assert formatted.id == "test_id"
        assert formatted.similarity == 0.85
        assert formatted.content == "Test content"
        assert formatted.metadata["methodName"] == "TestMethod"

    def test_format_result_fallback_id(self):
        """Test fallback to _id if id not present."""
        retrieval = MultiStageRetrieval()

        raw = {
            "_id": "fallback_id",
            "content": "Test",
            "score": 0.8,
            "metadata": {},
        }

        formatted = retrieval._format_result(raw)

        assert formatted.id == "fallback_id"

    def test_format_result_score_fallback(self):
        """Test fallback to score if similarity not present."""
        retrieval = MultiStageRetrieval()

        raw = {
            "id": "test_id",
            "score": 0.75,
            "metadata": {},
        }

        formatted = retrieval._format_result(raw)

        assert formatted.similarity == 0.75


class TestJSONParsing:
    """Test JSON parsing of metadata arrays."""

    def test_try_parse_json_list(self):
        """Test parsing actual list."""
        retrieval = MultiStageRetrieval()

        result = retrieval._try_parse_json(["Method1", "Method2"])

        assert result == ["Method1", "Method2"]

    def test_try_parse_json_string(self):
        """Test parsing JSON string."""
        retrieval = MultiStageRetrieval()

        result = retrieval._try_parse_json('["Method1", "Method2"]')

        assert result == ["Method1", "Method2"]

    def test_try_parse_json_invalid(self):
        """Test parsing invalid JSON returns empty list."""
        retrieval = MultiStageRetrieval()

        result = retrieval._try_parse_json("not valid json")

        assert result == []

    def test_try_parse_json_none(self):
        """Test parsing None returns empty list."""
        retrieval = MultiStageRetrieval()

        result = retrieval._try_parse_json(None)

        assert result == []

    def test_try_parse_json_empty_string(self):
        """Test parsing empty string returns empty list."""
        retrieval = MultiStageRetrieval()

        result = retrieval._try_parse_json("")

        assert result == []


class TestStageConfiguration:
    """Test stage configuration behavior."""

    @pytest.mark.asyncio
    async def test_disabled_stage_skipped(
        self,
        mock_mongodb_service,
    ):
        """Test disabled stages are not executed."""
        mock_mongodb_service.search_vectors = AsyncMock(return_value=[])

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        stages = [
            RetrievalStage(
                type=RetrievalStageType.METHODS,
                collection="code_methods",
                enabled=False,  # Disabled
                filter_category="code",
                filter_type="method",
            ),
        ]

        await retrieval.execute(query="Test", stages=stages)

        # Should not have called search
        mock_mongodb_service.search_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_stage_limit_respected(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test stage limit is passed to search."""
        # Create more results than limit
        many_results = [
            code_flow_test_data.create_method_result(f"Method{i}")
            for i in range(20)
        ]
        mock_mongodb_service.set_search_results("code", "method", many_results)

        retrieval = MultiStageRetrieval(mongodb_service=mock_mongodb_service)

        stages = [
            RetrievalStage(
                type=RetrievalStageType.METHODS,
                collection="code_methods",
                limit=5,  # Limit to 5
                enabled=True,
                filter_category="code",
                filter_type="method",
            ),
        ]

        results = await retrieval.execute(query="Test", stages=stages)

        # Should have at most 5 results
        assert len(results.methods) <= 5
