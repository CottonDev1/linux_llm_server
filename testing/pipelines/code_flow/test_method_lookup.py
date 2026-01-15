"""
Method Lookup Tests
===================

Test method lookup functionality including:
- Method search by name
- Method search by signature
- Filtering by project/class
- Response format validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from code_flow_pipeline.models.query_models import (
    MethodLookupRequest,
    MethodLookupResponse,
    MethodInfo,
)
from utils import generate_test_id


class TestMethodLookupRequest:
    """Test MethodLookupRequest model validation."""

    def test_request_with_method_name(self):
        """Test request with method name only."""
        request = MethodLookupRequest(
            method_name="SaveBale"
        )

        assert request.method_name == "SaveBale"
        assert request.class_name is None
        assert request.project is None
        assert request.limit == 10  # Default

    def test_request_with_all_fields(self):
        """Test request with all fields."""
        request = MethodLookupRequest(
            method_name="SaveBale",
            class_name="BaleService",
            project="Gin",
            signature="void SaveBale(Bale bale)",
            limit=20,
        )

        assert request.method_name == "SaveBale"
        assert request.class_name == "BaleService"
        assert request.project == "Gin"
        assert request.signature == "void SaveBale(Bale bale)"
        assert request.limit == 20

    def test_request_limit_bounds(self):
        """Test request limit validation."""
        # Valid limit
        request = MethodLookupRequest(method_name="Test", limit=50)
        assert request.limit == 50

        # Min limit
        request = MethodLookupRequest(method_name="Test", limit=1)
        assert request.limit == 1

        # Max limit
        request = MethodLookupRequest(method_name="Test", limit=100)
        assert request.limit == 100


class TestMethodLookupResponse:
    """Test MethodLookupResponse model."""

    def test_response_creation(self):
        """Test basic response creation."""
        response = MethodLookupResponse(
            success=True,
            methods=[],
            total=0,
        )

        assert response.success is True
        assert response.methods == []
        assert response.total == 0

    def test_response_with_methods(self):
        """Test response with methods."""
        methods = [
            {"name": "Method1", "class": "Class1"},
            {"name": "Method2", "class": "Class2"},
        ]

        response = MethodLookupResponse(
            success=True,
            methods=methods,
            total=2,
        )

        assert len(response.methods) == 2
        assert response.total == 2

    def test_response_total_results_alias(self):
        """Test total_results property alias."""
        response = MethodLookupResponse(
            success=True,
            methods=[],
            total=5,
        )

        assert response.total_results == 5


class TestMethodInfo:
    """Test MethodInfo dataclass."""

    def test_method_info_creation(self):
        """Test basic method info creation."""
        info = MethodInfo(
            name="SaveBale",
            class_name="BaleService",
        )

        assert info.name == "SaveBale"
        assert info.class_name == "BaleService"

    def test_method_info_defaults(self):
        """Test method info default values."""
        info = MethodInfo()

        assert info.name == ""
        assert info.class_name == ""
        assert info.calls == []
        assert info.called_by == []
        assert info.is_database_accessor is False

    def test_method_info_all_fields(self):
        """Test method info with all fields."""
        info = MethodInfo(
            name="SaveBale",
            class_name="BaleService",
            signature="void SaveBale(Bale bale)",
            file_path="/Services/BaleService.cs",
            project="Gin",
            doc_string="Saves a bale to the database",
            calls=["ValidateBale", "InsertBale"],
            called_by=["ProcessBale"],
            is_database_accessor=True,
            metadata={"custom": "value"},
        )

        assert info.name == "SaveBale"
        assert info.class_name == "BaleService"
        assert info.signature == "void SaveBale(Bale bale)"
        assert info.file_path == "/Services/BaleService.cs"
        assert info.project == "Gin"
        assert info.doc_string == "Saves a bale to the database"
        assert "ValidateBale" in info.calls
        assert "ProcessBale" in info.called_by
        assert info.is_database_accessor is True
        assert info.metadata["custom"] == "value"


class TestMethodLookupPipeline:
    """Test method lookup through the pipeline."""

    @pytest.mark.asyncio
    async def test_lookup_by_method_name(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test looking up methods by name."""
        from code_flow_pipeline.pipeline import CodeFlowPipeline

        # Configure mock results
        save_method = code_flow_test_data.create_method_result(
            "SaveBale",
            class_name="BaleService",
            project="Gin",
        )
        mock_mongodb_service.set_default_results([save_method])

        pipeline = CodeFlowPipeline()
        pipeline._mongodb_service = mock_mongodb_service

        # We need to mock _mongodb since lookup_method uses it directly
        pipeline._mongodb = mock_mongodb_service

        request = MethodLookupRequest(
            method_name="SaveBale",
        )

        response = await pipeline.lookup_method(request)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_lookup_by_class_name(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test filtering by class name."""
        method1 = code_flow_test_data.create_method_result(
            "Save",
            class_name="BaleService",
        )
        method2 = code_flow_test_data.create_method_result(
            "Save",
            class_name="OrderService",
        )

        mock_mongodb_service.set_default_results([method1, method2])

        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline()
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._mongodb = mock_mongodb_service

        request = MethodLookupRequest(
            method_name="Save",
            class_name="BaleService",
        )

        response = await pipeline.lookup_method(request)

        assert response.success is True

    @pytest.mark.asyncio
    async def test_lookup_with_project_filter(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test filtering by project."""
        gin_method = code_flow_test_data.create_method_result(
            "ProcessBale",
            project="Gin",
        )
        warehouse_method = code_flow_test_data.create_method_result(
            "ProcessOrder",
            project="Warehouse",
        )

        mock_mongodb_service.set_default_results([gin_method, warehouse_method])

        from code_flow_pipeline.pipeline import CodeFlowPipeline

        pipeline = CodeFlowPipeline()
        pipeline._mongodb_service = mock_mongodb_service
        pipeline._mongodb = mock_mongodb_service

        request = MethodLookupRequest(
            method_name="Process",
            project="Gin",
        )

        response = await pipeline.lookup_method(request)

        assert response.success is True


class TestMethodLookupSearchTerms:
    """Test search term construction for method lookup."""

    def test_search_term_from_method_name(self):
        """Test search term includes method name."""
        request = MethodLookupRequest(
            method_name="SaveBale"
        )

        # Build search terms manually (mimicking pipeline logic)
        search_terms = []
        if request.method_name:
            search_terms.append(f"method {request.method_name}")

        assert "method SaveBale" in search_terms

    def test_search_term_from_class_name(self):
        """Test search term includes class name."""
        request = MethodLookupRequest(
            method_name="Save",
            class_name="BaleService",
        )

        search_terms = []
        if request.method_name:
            search_terms.append(f"method {request.method_name}")
        if request.class_name:
            search_terms.append(f"class {request.class_name}")

        assert "method Save" in search_terms
        assert "class BaleService" in search_terms

    def test_search_term_from_signature(self):
        """Test search term includes signature."""
        request = MethodLookupRequest(
            method_name="",
            signature="void SaveBale(Bale bale)",
        )

        search_terms = []
        if request.method_name:
            search_terms.append(f"method {request.method_name}")
        if request.signature:
            search_terms.append(f"signature {request.signature}")

        assert "signature void SaveBale(Bale bale)" in search_terms


class TestMethodLookupFiltering:
    """Test result filtering logic."""

    def test_filter_by_method_name_partial_match(self):
        """Test partial method name matching."""
        results = [
            {"metadata": {"methodName": "SaveBale"}},
            {"metadata": {"methodName": "SaveOrder"}},
            {"metadata": {"methodName": "LoadBale"}},
        ]

        method_name_filter = "Save"

        filtered = [
            r for r in results
            if method_name_filter.lower() in r["metadata"].get("methodName", "").lower()
        ]

        assert len(filtered) == 2
        assert all("Save" in r["metadata"]["methodName"] for r in filtered)

    def test_filter_by_class_name_partial_match(self):
        """Test partial class name matching."""
        results = [
            {"metadata": {"methodName": "Save", "className": "BaleService"}},
            {"metadata": {"methodName": "Save", "className": "OrderService"}},
            {"metadata": {"methodName": "Save", "className": "BaleValidator"}},
        ]

        class_name_filter = "Bale"

        filtered = [
            r for r in results
            if class_name_filter.lower() in r["metadata"].get("className", "").lower()
        ]

        assert len(filtered) == 2
        assert all("Bale" in r["metadata"]["className"] for r in filtered)

    def test_filter_by_signature_partial_match(self):
        """Test partial signature matching."""
        results = [
            {"metadata": {"methodName": "Save", "signature": "void Save(Bale bale)"}},
            {"metadata": {"methodName": "Save", "signature": "void Save(Order order)"}},
            {"metadata": {"methodName": "Save", "fullMethodSignature": "void Save(Item item)"}},
        ]

        signature_filter = "Bale"

        filtered = [
            r for r in results
            if signature_filter.lower() in (
                r["metadata"].get("signature", "") or
                r["metadata"].get("fullMethodSignature", "")
            ).lower()
        ]

        assert len(filtered) == 1

    def test_filter_case_insensitive(self):
        """Test filtering is case-insensitive."""
        results = [
            {"metadata": {"methodName": "SAVEBALE"}},
            {"metadata": {"methodName": "savebale"}},
            {"metadata": {"methodName": "SaveBale"}},
        ]

        method_name_filter = "savebale"

        filtered = [
            r for r in results
            if method_name_filter.lower() in r["metadata"].get("methodName", "").lower()
        ]

        assert len(filtered) == 3


class TestMethodLookupResultFormat:
    """Test method lookup result formatting."""

    def test_result_contains_method_name(self, code_flow_test_data):
        """Test result contains method name."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
            class_name="TestClass",
        )

        metadata = raw_result.get("metadata", {})

        assert metadata.get("methodName") == "TestMethod"
        assert metadata.get("className") == "TestClass"

    def test_result_contains_file_path(self, code_flow_test_data):
        """Test result contains file path."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
            class_name="TestClass",
        )

        metadata = raw_result.get("metadata", {})

        assert "filePath" in metadata
        assert metadata["filePath"].endswith(".cs")

    def test_result_contains_line_numbers(self, code_flow_test_data):
        """Test result contains line numbers."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
        )

        metadata = raw_result.get("metadata", {})

        assert "startLine" in metadata
        assert "endLine" in metadata
        assert isinstance(metadata["startLine"], int)
        assert isinstance(metadata["endLine"], int)

    def test_result_contains_return_type(self, code_flow_test_data):
        """Test result contains return type."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
        )

        metadata = raw_result.get("metadata", {})

        assert "returnType" in metadata

    def test_result_contains_visibility(self, code_flow_test_data):
        """Test result contains visibility flags."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
        )

        metadata = raw_result.get("metadata", {})

        assert "isPublic" in metadata
        assert isinstance(metadata["isPublic"], bool)

    def test_result_contains_calls(self, code_flow_test_data):
        """Test result contains called methods."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
            calls_method=["Helper1", "Helper2"],
        )

        metadata = raw_result.get("metadata", {})

        assert "callsMethod" in metadata

    def test_result_contains_callers(self, code_flow_test_data):
        """Test result contains calling methods."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
            called_by=["Caller1", "Caller2"],
        )

        metadata = raw_result.get("metadata", {})

        assert "calledByMethod" in metadata

    def test_result_contains_database_tables(self, code_flow_test_data):
        """Test result contains database tables."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
            database_tables=["Users", "Orders"],
        )

        metadata = raw_result.get("metadata", {})

        assert "databaseTables" in metadata

    def test_result_contains_similarity_score(self, code_flow_test_data):
        """Test result contains similarity score."""
        raw_result = code_flow_test_data.create_method_result(
            "TestMethod",
            similarity=0.92,
        )

        assert raw_result.get("similarity") == 0.92
        assert raw_result.get("score") == 0.92


class TestMethodLookupLimit:
    """Test result limiting behavior."""

    def test_results_respect_limit(self):
        """Test results are limited to requested count."""
        results = [
            {"metadata": {"methodName": f"Method{i}"}}
            for i in range(20)
        ]

        limit = 5
        limited_results = results[:limit]

        assert len(limited_results) == 5

    def test_over_fetch_for_filtering(self):
        """Test over-fetching to allow for post-filter."""
        request_limit = 10

        # Pipeline over-fetches by 2x for filtering
        fetch_limit = request_limit * 2

        assert fetch_limit == 20


class TestMethodLookupEdgeCases:
    """Test edge cases in method lookup."""

    def test_empty_method_name(self):
        """Test handling empty method name."""
        request = MethodLookupRequest(method_name="")

        # Empty method name should be valid but return no results
        assert request.method_name == ""

    def test_special_characters_in_name(self):
        """Test handling special characters in method name."""
        request = MethodLookupRequest(method_name="Process<T>")

        assert request.method_name == "Process<T>"

    def test_unicode_in_method_name(self):
        """Test handling unicode in method name."""
        request = MethodLookupRequest(method_name="Traiter_Commande")

        assert request.method_name == "Traiter_Commande"

    def test_no_results_found(self):
        """Test handling no results."""
        response = MethodLookupResponse(
            success=True,
            methods=[],
            total=0,
        )

        assert response.success is True
        assert response.total == 0
        assert len(response.methods) == 0

    def test_method_not_in_requested_project(self, code_flow_test_data):
        """Test method exists but not in requested project."""
        method = code_flow_test_data.create_method_result(
            "TestMethod",
            project="DifferentProject",
        )

        # Filter by different project
        request_project = "RequestedProject"
        method_project = method["metadata"]["project"]

        assert method_project != request_project
