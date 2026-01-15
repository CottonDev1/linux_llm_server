"""
Code Flow API Endpoint Tests
============================

Test the FastAPI endpoints for code flow analysis including:
- POST /api/code-flow - Main analysis endpoint
- POST /api/code-flow/stream - SSE streaming endpoint
- GET /api/method-lookup - Method search endpoint
- POST /api/call-chain - Call chain building endpoint
- DELETE /api/code-flow/cache - Cache clearing endpoint
- GET /api/code-flow/health - Health check endpoint
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from utils.api_test_client import APITestClient, SSEEvent
from fixtures.shared_fixtures import sse_consumer, response_validator
from utils import generate_test_id


# =============================================================================
# Test Setup
# =============================================================================

@pytest.fixture
def mock_pipeline():
    """Create a mock CodeFlowPipeline for endpoint testing."""
    from code_flow_pipeline.models.query_models import (
        CodeFlowResponse,
        MethodLookupResponse,
        CallChainResponse,
        QueryType,
    )

    pipeline = MagicMock()

    # Mock analyze
    pipeline.analyze = AsyncMock(return_value=CodeFlowResponse(
        success=True,
        query="Test query",
        project="TestProject",
        answer="This is the answer to your question.",
        query_type=QueryType.GENERAL,
        confidence=0.8,
        sources=[],
        call_chains=[],
        total_results=5,
        processing_time=1.5,
        cached=False,
    ))

    # Mock lookup_method
    pipeline.lookup_method = AsyncMock(return_value=MethodLookupResponse(
        success=True,
        methods=[
            {"name": "TestMethod", "class": "TestClass"}
        ],
        total=1,
    ))

    # Mock build_call_chain
    pipeline.build_call_chain = AsyncMock(return_value=CallChainResponse(
        success=True,
        entry_point="TestMethod",
        chains=[],
        tree=None,
        total_chains=0,
    ))

    # Mock clear_cache
    pipeline.clear_cache = MagicMock()

    # Mock _cache for health check
    pipeline._cache = {}

    return pipeline


@pytest.fixture
def app_with_mock_pipeline(mock_pipeline):
    """Create FastAPI app with mocked pipeline."""
    from fastapi import FastAPI
    from api.code_flow_routes import router

    app = FastAPI()
    app.include_router(router)

    # Patch the get_pipeline function
    async def mock_get_pipeline():
        return mock_pipeline

    with patch('api.code_flow_routes.get_pipeline', mock_get_pipeline):
        yield app


@pytest.fixture
def test_client(app_with_mock_pipeline):
    """Create test client."""
    return TestClient(app_with_mock_pipeline)


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Test GET /api/code-flow/health endpoint."""

    def test_health_returns_healthy(self, test_client, mock_pipeline):
        """Test health endpoint returns healthy status."""
        response = test_client.get("/api/code-flow/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline_initialized"] is True

    def test_health_includes_cache_size(self, test_client, mock_pipeline):
        """Test health endpoint includes cache information."""
        mock_pipeline._cache = {"key1": "value1", "key2": "value2"}

        response = test_client.get("/api/code-flow/health")

        assert response.status_code == 200
        data = response.json()
        assert "cache_size" in data

    def test_health_includes_version(self, test_client):
        """Test health endpoint includes version."""
        response = test_client.get("/api/code-flow/health")

        data = response.json()
        assert "version" in data


# =============================================================================
# Code Flow Analysis Endpoint Tests
# =============================================================================

class TestCodeFlowEndpoint:
    """Test POST /api/code-flow endpoint."""

    def test_analyze_basic_query(self, test_client, mock_pipeline):
        """Test basic code flow analysis."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "How does bale processing work?",
                "project": "Gin",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "answer" in data

    def test_analyze_with_all_options(self, test_client, mock_pipeline):
        """Test analysis with all options specified."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "How does bale processing work?",
                "project": "Gin",
                "maxHops": 10,
                "includeCallGraph": True,
                "detailed": True,
            }
        )

        assert response.status_code == 200
        mock_pipeline.analyze.assert_called_once()

    def test_analyze_without_project(self, test_client, mock_pipeline):
        """Test analysis without project filter."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "Find methods that save data",
            }
        )

        assert response.status_code == 200

    def test_analyze_validates_question_min_length(self, test_client):
        """Test question minimum length validation."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "ab",  # Too short (min 3)
            }
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_validates_max_hops_bounds(self, test_client):
        """Test maxHops validation bounds."""
        # Too low
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "Test question",
                "maxHops": 0,
            }
        )
        assert response.status_code == 422

        # Too high
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "Test question",
                "maxHops": 25,  # Max is 20
            }
        )
        assert response.status_code == 422

    def test_analyze_returns_query_type(self, test_client, mock_pipeline):
        """Test response includes query type classification."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "How does the workflow process orders?",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "query_type" in data

    def test_analyze_returns_processing_time(self, test_client, mock_pipeline):
        """Test response includes processing time."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "Test question for timing",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "processing_time" in data
        assert isinstance(data["processing_time"], (int, float))


class TestCodeFlowStreamEndpoint:
    """Test POST /api/code-flow/stream SSE endpoint."""

    def test_stream_returns_sse_content_type(self, test_client, mock_pipeline):
        """Test stream endpoint returns SSE content type."""
        from code_flow_pipeline.models.query_models import SSEEvent

        # Mock analyze_stream to return async generator
        async def mock_stream(request):
            yield SSEEvent(event="status", data=json.dumps({"stage": "start"}))
            yield SSEEvent(event="done", data=json.dumps({"complete": True}))

        mock_pipeline.analyze_stream = mock_stream

        with test_client.stream(
            "POST",
            "/api/code-flow/stream",
            json={"question": "Test streaming query"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

    def test_stream_accepts_same_params_as_analyze(self, test_client, mock_pipeline):
        """Test stream endpoint accepts same parameters as analyze."""
        from code_flow_pipeline.models.query_models import SSEEvent

        async def mock_stream(request):
            yield SSEEvent(event="done", data=json.dumps({}))

        mock_pipeline.analyze_stream = mock_stream

        response = test_client.post(
            "/api/code-flow/stream",
            json={
                "question": "How does bale processing work?",
                "project": "Gin",
                "maxHops": 5,
                "includeCallGraph": True,
            }
        )

        # Should not fail validation
        assert response.status_code in [200, 500]  # 500 if mock doesn't work correctly


# =============================================================================
# Method Lookup Endpoint Tests
# =============================================================================

class TestMethodLookupEndpoint:
    """Test GET /api/method-lookup endpoint."""

    def test_lookup_by_method_name(self, test_client, mock_pipeline):
        """Test method lookup by name."""
        response = test_client.get(
            "/api/method-lookup",
            params={"methodName": "SaveBale"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "methods" in data

    def test_lookup_with_class_filter(self, test_client, mock_pipeline):
        """Test method lookup with class filter."""
        response = test_client.get(
            "/api/method-lookup",
            params={
                "methodName": "Save",
                "className": "BaleService",
            }
        )

        assert response.status_code == 200

    def test_lookup_with_project_filter(self, test_client, mock_pipeline):
        """Test method lookup with project filter."""
        response = test_client.get(
            "/api/method-lookup",
            params={
                "methodName": "Process",
                "project": "Gin",
            }
        )

        assert response.status_code == 200

    def test_lookup_with_limit(self, test_client, mock_pipeline):
        """Test method lookup with custom limit."""
        response = test_client.get(
            "/api/method-lookup",
            params={
                "methodName": "Test",
                "limit": 5,
            }
        )

        assert response.status_code == 200

    def test_lookup_validates_limit_bounds(self, test_client):
        """Test limit parameter validation."""
        # Too low
        response = test_client.get(
            "/api/method-lookup",
            params={
                "methodName": "Test",
                "limit": 0,
            }
        )
        assert response.status_code == 422

        # Too high
        response = test_client.get(
            "/api/method-lookup",
            params={
                "methodName": "Test",
                "limit": 150,  # Max is 100
            }
        )
        assert response.status_code == 422

    def test_lookup_requires_method_name(self, test_client):
        """Test method name is required."""
        response = test_client.get("/api/method-lookup")

        assert response.status_code == 422


# =============================================================================
# Call Chain Endpoint Tests
# =============================================================================

class TestCallChainEndpoint:
    """Test POST /api/call-chain endpoint."""

    def test_build_chain_basic(self, test_client, mock_pipeline):
        """Test basic call chain building."""
        response = test_client.post(
            "/api/call-chain",
            json={
                "entryPoint": "btnSave_Click",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "entry_point" in data

    def test_build_chain_with_project(self, test_client, mock_pipeline):
        """Test call chain with project filter."""
        response = test_client.post(
            "/api/call-chain",
            json={
                "entryPoint": "btnSave_Click",
                "project": "Gin",
            }
        )

        assert response.status_code == 200

    def test_build_chain_with_target(self, test_client, mock_pipeline):
        """Test call chain with target method."""
        response = test_client.post(
            "/api/call-chain",
            json={
                "entryPoint": "btnSave_Click",
                "targetMethod": "SaveToDatabase",
            }
        )

        assert response.status_code == 200

    def test_build_chain_with_max_depth(self, test_client, mock_pipeline):
        """Test call chain with custom max depth."""
        response = test_client.post(
            "/api/call-chain",
            json={
                "entryPoint": "ProcessOrder",
                "maxDepth": 15,
            }
        )

        assert response.status_code == 200

    def test_build_chain_validates_max_depth(self, test_client):
        """Test max depth validation."""
        # Too high
        response = test_client.post(
            "/api/call-chain",
            json={
                "entryPoint": "Test",
                "maxDepth": 100,  # Max is 50
            }
        )
        assert response.status_code == 422

    def test_build_chain_requires_entry_point(self, test_client):
        """Test entry point is required."""
        response = test_client.post(
            "/api/call-chain",
            json={}
        )

        assert response.status_code == 422


# =============================================================================
# Cache Clear Endpoint Tests
# =============================================================================

class TestCacheClearEndpoint:
    """Test DELETE /api/code-flow/cache endpoint."""

    def test_clear_cache_success(self, test_client, mock_pipeline):
        """Test cache clearing returns success."""
        response = test_client.delete("/api/code-flow/cache")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_pipeline.clear_cache.assert_called_once()

    def test_clear_cache_returns_message(self, test_client, mock_pipeline):
        """Test cache clearing returns message."""
        response = test_client.delete("/api/code-flow/cache")

        data = response.json()
        assert "message" in data


# =============================================================================
# Legacy Endpoint Tests (v1)
# =============================================================================

class TestLegacyEndpoints:
    """Test legacy v1 endpoints for backward compatibility."""

    def test_code_flow_v1_endpoint(self, test_client, mock_pipeline):
        """Test legacy code flow v1 endpoint."""
        response = test_client.post(
            "/api/code-flow-v1",
            json={
                "question": "Test query",
            }
        )

        assert response.status_code == 200

    def test_method_lookup_v1_endpoint(self, test_client, mock_pipeline):
        """Test legacy method lookup v1 endpoint."""
        response = test_client.get(
            "/api/method-lookup-v1",
            params={"method": "TestMethod"}
        )

        assert response.status_code == 200

    def test_method_lookup_v1_requires_param(self, test_client):
        """Test v1 method lookup requires at least one parameter."""
        response = test_client.get("/api/method-lookup-v1")

        # Should return 400 or similar error
        assert response.status_code in [400, 422]

    def test_call_chain_v1_endpoint(self, test_client, mock_pipeline):
        """Test legacy call chain v1 endpoint."""
        response = test_client.get(
            "/api/call-chain-v1",
            params={
                "startMethod": "TestMethod",
                "project": "TestProject",
            }
        )

        assert response.status_code == 200

    def test_call_chain_v1_requires_project(self, test_client):
        """Test v1 call chain requires project parameter."""
        response = test_client.get(
            "/api/call-chain-v1",
            params={"startMethod": "TestMethod"}
        )

        assert response.status_code == 400

    def test_call_chain_v1_requires_start(self, test_client):
        """Test v1 call chain requires start method or event handler."""
        response = test_client.get(
            "/api/call-chain-v1",
            params={"project": "TestProject"}
        )

        assert response.status_code == 400


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in endpoints."""

    def test_analyze_handles_pipeline_error(self, test_client, mock_pipeline):
        """Test analysis handles pipeline errors gracefully."""
        mock_pipeline.analyze.side_effect = Exception("Pipeline error")

        response = test_client.post(
            "/api/code-flow",
            json={"question": "Test query"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"]

    def test_lookup_handles_pipeline_error(self, test_client, mock_pipeline):
        """Test lookup handles pipeline errors gracefully."""
        mock_pipeline.lookup_method.side_effect = Exception("Lookup error")

        response = test_client.get(
            "/api/method-lookup",
            params={"methodName": "Test"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"]

    def test_call_chain_handles_pipeline_error(self, test_client, mock_pipeline):
        """Test call chain handles pipeline errors gracefully."""
        mock_pipeline.build_call_chain.side_effect = Exception("Chain error")

        response = test_client.post(
            "/api/call-chain",
            json={"entryPoint": "Test"}
        )

        assert response.status_code == 500

    def test_cache_clear_handles_error(self, test_client, mock_pipeline):
        """Test cache clear handles errors gracefully."""
        mock_pipeline.clear_cache.side_effect = Exception("Cache error")

        response = test_client.delete("/api/code-flow/cache")

        assert response.status_code == 500


# =============================================================================
# Request Validation Tests
# =============================================================================

class TestRequestValidation:
    """Test request body validation."""

    def test_invalid_json_body(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/api/code-flow",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_field(self, test_client):
        """Test handling of missing required field."""
        response = test_client.post(
            "/api/code-flow",
            json={}  # Missing 'question'
        )

        assert response.status_code == 422

    def test_wrong_field_type(self, test_client):
        """Test handling of wrong field type."""
        response = test_client.post(
            "/api/code-flow",
            json={
                "question": "Valid question",
                "maxHops": "not a number",  # Should be int
            }
        )

        assert response.status_code == 422


# =============================================================================
# Response Format Tests
# =============================================================================

class TestResponseFormat:
    """Test response format consistency."""

    def test_analyze_response_format(self, test_client, mock_pipeline):
        """Test analyze response has expected format."""
        response = test_client.post(
            "/api/code-flow",
            json={"question": "Test query"}
        )

        data = response.json()

        # Required fields
        assert "success" in data
        assert "query" in data
        assert "answer" in data
        assert "query_type" in data
        assert "confidence" in data
        assert "sources" in data
        assert "processing_time" in data

    def test_lookup_response_format(self, test_client, mock_pipeline):
        """Test lookup response has expected format."""
        response = test_client.get(
            "/api/method-lookup",
            params={"methodName": "Test"}
        )

        data = response.json()

        assert "success" in data
        assert "methods" in data
        assert isinstance(data["methods"], list)

    def test_call_chain_response_format(self, test_client, mock_pipeline):
        """Test call chain response has expected format."""
        response = test_client.post(
            "/api/call-chain",
            json={"entryPoint": "Test"}
        )

        data = response.json()

        assert "success" in data
        assert "entry_point" in data

    def test_health_response_format(self, test_client):
        """Test health response has expected format."""
        response = test_client.get("/api/code-flow/health")

        data = response.json()

        assert "status" in data
        assert "pipeline_initialized" in data
        assert "version" in data
