"""
Code Assistance API Endpoint Tests
==================================

Test the FastAPI endpoints for the code assistance pipeline including:
- POST /api/code/query endpoint
- POST /api/code/query/stream SSE endpoint
- POST /api/code/feedback endpoint
- GET /api/code/stats endpoint
- GET /api/code/health endpoint
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from utils import generate_test_id
from utils.api_test_client import APITestClient, APIResponse, SSEEvent


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_query_payload() -> Dict[str, Any]:
    """Create a sample query payload."""
    return {
        "query": "How do I save a bale in the system?",
        "project": "Gin",
        "options": {
            "include_sources": True,
            "include_call_chains": True,
            "max_sources": 10,
        },
    }


@pytest.fixture
def sample_feedback_payload() -> Dict[str, Any]:
    """Create a sample feedback payload."""
    return {
        "response_id": f"resp_{generate_test_id()}",
        "rating": 5,
        "feedback": "Very helpful explanation!",
        "was_helpful": True,
        "is_helpful": True,
    }


@pytest.fixture
def mock_query_response():
    """Create a mock query response."""
    from code_assistance_pipeline.models.query_models import (
        CodeQueryResponse,
        SourceInfo,
        SourceType,
        TokenUsage,
        TimingInfo,
    )

    return CodeQueryResponse(
        success=True,
        response_id="resp_test_123",
        query="How do I save a bale?",
        answer="To save a bale, use the BaleService.SaveBale method.",
        sources=[
            SourceInfo(
                type=SourceType.METHOD,
                name="BaleService.SaveBale",
                file_path="/src/Services/BaleService.cs",
                line_number=42,
                relevance_score=0.95,
            ),
        ],
        call_chain=["btnSave_Click", "ValidateBale", "SaveBale"],
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        timing=TimingInfo(
            retrieval_ms=50,
            generation_ms=1500,
            total_ms=1550,
        ),
    )


@pytest.fixture
def mock_stats_response():
    """Create a mock stats response."""
    from code_assistance_pipeline.models.query_models import CodeStatsResponse

    return CodeStatsResponse(
        success=True,
        total_methods=1500,
        total_classes=300,
        total_interfaces=100,
        total_event_handlers=200,
        projects=["Gin", "Warehouse", "Central"],
        code_entities={
            "methods": 1500,
            "classes": 300,
            "interfaces": 100,
            "event_handlers": 200,
        },
        timestamp="2024-01-15T12:00:00Z",
    )


# =============================================================================
# Router Creation Tests
# =============================================================================

class TestCodeRoutesCreation:
    """Test route creation and registration."""

    def test_create_code_routes_returns_router(self, pipeline_config):
        """Test that create_code_routes returns a router."""
        from code_assistance_pipeline.routes import create_code_routes
        from fastapi import APIRouter

        router = create_code_routes()

        assert isinstance(router, APIRouter)

    def test_router_has_query_endpoint(self, pipeline_config):
        """Test that router has /query endpoint."""
        from code_assistance_pipeline.routes import create_code_routes

        router = create_code_routes()
        routes = [r.path for r in router.routes]

        assert "/query" in routes

    def test_router_has_stream_endpoint(self, pipeline_config):
        """Test that router has /query/stream endpoint."""
        from code_assistance_pipeline.routes import create_code_routes

        router = create_code_routes()
        routes = [r.path for r in router.routes]

        assert "/query/stream" in routes

    def test_router_has_feedback_endpoint(self, pipeline_config):
        """Test that router has /feedback endpoint."""
        from code_assistance_pipeline.routes import create_code_routes

        router = create_code_routes()
        routes = [r.path for r in router.routes]

        assert "/feedback" in routes

    def test_router_has_stats_endpoint(self, pipeline_config):
        """Test that router has /stats endpoint."""
        from code_assistance_pipeline.routes import create_code_routes

        router = create_code_routes()
        routes = [r.path for r in router.routes]

        assert "/stats" in routes

    def test_router_has_health_endpoint(self, pipeline_config):
        """Test that router has /health endpoint."""
        from code_assistance_pipeline.routes import create_code_routes

        router = create_code_routes()
        routes = [r.path for r in router.routes]

        assert "/health" in routes


# =============================================================================
# Query Endpoint Tests
# =============================================================================

class TestQueryEndpoint:
    """Test POST /api/code/query endpoint."""

    @pytest.mark.asyncio
    async def test_query_endpoint_success(
        self,
        pipeline_config,
        sample_query_payload,
        mock_query_response,
    ):
        """Test successful query request."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_query.return_value = mock_query_response
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.post(
                    "/api/code/query",
                    json=sample_query_payload,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "answer" in data
                assert "sources" in data

    @pytest.mark.asyncio
    async def test_query_endpoint_validation_error(self, pipeline_config):
        """Test query endpoint with invalid payload."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        app = FastAPI()
        register_code_routes(app)

        with TestClient(app) as client:
            # Missing required 'query' field
            response = client.post(
                "/api/code/query",
                json={"project": "Gin"},  # Missing query
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_endpoint_includes_sources(
        self,
        pipeline_config,
        sample_query_payload,
        mock_query_response,
    ):
        """Test that query response includes sources."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_query.return_value = mock_query_response
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.post(
                    "/api/code/query",
                    json=sample_query_payload,
                )

                data = response.json()
                assert "sources" in data
                assert len(data["sources"]) > 0
                assert data["sources"][0]["name"] == "BaleService.SaveBale"

    @pytest.mark.asyncio
    async def test_query_endpoint_includes_timing(
        self,
        pipeline_config,
        sample_query_payload,
        mock_query_response,
    ):
        """Test that query response includes timing info."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_query.return_value = mock_query_response
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.post(
                    "/api/code/query",
                    json=sample_query_payload,
                )

                data = response.json()
                assert "timing" in data
                assert "total_ms" in data["timing"]

    @pytest.mark.asyncio
    async def test_query_endpoint_handles_pipeline_error(
        self, pipeline_config, sample_query_payload
    ):
        """Test query endpoint handles pipeline errors."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_query.side_effect = Exception("Pipeline error")
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.post(
                    "/api/code/query",
                    json=sample_query_payload,
                )

                assert response.status_code == 500


# =============================================================================
# Feedback Endpoint Tests
# =============================================================================

class TestFeedbackEndpoint:
    """Test POST /api/code/feedback endpoint."""

    @pytest.mark.asyncio
    async def test_feedback_endpoint_success(
        self, pipeline_config, sample_feedback_payload
    ):
        """Test successful feedback submission."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import CodeFeedbackResponse

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.submit_feedback.return_value = CodeFeedbackResponse(
                success=True,
                feedback_id="fb_test_123",
            )
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.post(
                    "/api/code/feedback",
                    json=sample_feedback_payload,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_endpoint_with_error_category(self, pipeline_config):
        """Test feedback submission with error category."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import CodeFeedbackResponse

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.submit_feedback.return_value = CodeFeedbackResponse(
                success=True,
                feedback_id="fb_test_123",
            )
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                payload = {
                    "response_id": "resp_123",
                    "rating": 2,
                    "is_helpful": False,
                    "error_category": "wrong_method",
                    "expected_methods": ["BaleService.ValidateBale"],
                }

                response = client.post(
                    "/api/code/feedback",
                    json=payload,
                )

                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_feedback_endpoint_validation_error(self, pipeline_config):
        """Test feedback endpoint with invalid payload."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        app = FastAPI()
        register_code_routes(app)

        with TestClient(app) as client:
            # Missing required response_id
            response = client.post(
                "/api/code/feedback",
                json={"rating": 5},
            )

            assert response.status_code == 422


# =============================================================================
# Stats Endpoint Tests
# =============================================================================

class TestStatsEndpoint:
    """Test GET /api/code/stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_endpoint_success(
        self, pipeline_config, mock_stats_response
    ):
        """Test successful stats retrieval."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_stats.return_value = mock_stats_response
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.get("/api/code/stats")

                assert response.status_code == 200
                data = response.json()
                assert "code_entities" in data or "total_methods" in data

    @pytest.mark.asyncio
    async def test_stats_endpoint_returns_entity_counts(
        self, pipeline_config, mock_stats_response
    ):
        """Test that stats includes entity counts."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_stats.return_value = mock_stats_response
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.get("/api/code/stats")

                data = response.json()
                # Check for code_entities dict with entity counts
                if "code_entities" in data:
                    entities = data["code_entities"]
                    assert "methods" in entities or "total_methods" in data


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Test GET /api/code/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(self, pipeline_config):
        """Test health endpoint when all components are healthy."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import CodeStatsResponse

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_stats.return_value = CodeStatsResponse(
                success=True,
                code_entities={"methods": 100},
            )
            mock_get_pipeline.return_value = mock_pipeline

            with patch('code_assistance_pipeline.routes.ResponseGenerator') as MockGenerator:
                mock_generator = AsyncMock()
                mock_generator.health_check.return_value = {
                    "healthy": True,
                    "endpoint": "http://localhost:8082",
                }
                MockGenerator.return_value = mock_generator

                app = FastAPI()
                register_code_routes(app)

                with TestClient(app) as client:
                    response = client.get("/api/code/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_endpoint_includes_components(self, pipeline_config):
        """Test that health response includes component status."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import CodeStatsResponse

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_stats.return_value = CodeStatsResponse(
                success=True,
                code_entities={"methods": 100},
            )
            mock_get_pipeline.return_value = mock_pipeline

            with patch('code_assistance_pipeline.routes.ResponseGenerator') as MockGenerator:
                mock_generator = AsyncMock()
                mock_generator.health_check.return_value = {"healthy": True}
                MockGenerator.return_value = mock_generator

                app = FastAPI()
                register_code_routes(app)

                with TestClient(app) as client:
                    response = client.get("/api/code/health")

                    data = response.json()
                    if data["healthy"]:
                        assert "components" in data

    @pytest.mark.asyncio
    async def test_health_endpoint_unhealthy(self, pipeline_config):
        """Test health endpoint when a component is unhealthy."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_get_pipeline.side_effect = Exception("Connection failed")

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.get("/api/code/health")

                data = response.json()
                assert data["healthy"] is False
                assert "error" in data


# =============================================================================
# Request Model Validation Tests
# =============================================================================

class TestRequestValidation:
    """Test request model validation."""

    @pytest.mark.asyncio
    async def test_query_options_defaults(self, pipeline_config):
        """Test that query options have correct defaults."""
        from code_assistance_pipeline.models.query_models import CodeQueryOptions

        options = CodeQueryOptions()

        assert options.include_sources is True
        assert options.include_call_chains is True
        assert options.max_sources == 10
        assert options.stream is False

    @pytest.mark.asyncio
    async def test_query_options_max_sources_validation(self, pipeline_config):
        """Test that max_sources has bounds validation."""
        from code_assistance_pipeline.models.query_models import CodeQueryOptions
        from pydantic import ValidationError

        # Should fail for values outside bounds
        with pytest.raises(ValidationError):
            CodeQueryOptions(max_sources=0)

        with pytest.raises(ValidationError):
            CodeQueryOptions(max_sources=100)

        # Should succeed for valid values
        options = CodeQueryOptions(max_sources=25)
        assert options.max_sources == 25

    @pytest.mark.asyncio
    async def test_query_request_with_history(self, pipeline_config):
        """Test query request with conversation history."""
        from code_assistance_pipeline.models.query_models import (
            CodeQueryRequest,
            ConversationMessage,
        )

        request = CodeQueryRequest(
            query="How do I use that method?",
            project="Gin",
            history=[
                ConversationMessage(
                    role="user",
                    content="What is BaleService?",
                ),
                ConversationMessage(
                    role="assistant",
                    content="BaleService handles bale operations.",
                ),
            ],
        )

        assert len(request.history) == 2
        assert request.history[0].role == "user"


# =============================================================================
# Response Model Tests
# =============================================================================

class TestResponseModels:
    """Test response model structure."""

    def test_code_query_response_structure(self, pipeline_config):
        """Test CodeQueryResponse structure."""
        from code_assistance_pipeline.models.query_models import (
            CodeQueryResponse,
            SourceInfo,
            SourceType,
            TokenUsage,
            TimingInfo,
        )

        response = CodeQueryResponse(
            success=True,
            response_id="test_123",
            query="Test query",
            answer="Test answer",
            sources=[
                SourceInfo(
                    type=SourceType.METHOD,
                    name="Test.Method",
                ),
            ],
            call_chain=["A", "B", "C"],
            token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
            timing=TimingInfo(
                retrieval_ms=50,
                generation_ms=1000,
                total_ms=1050,
            ),
        )

        # Verify serialization
        data = response.model_dump()

        assert data["success"] is True
        assert data["response_id"] == "test_123"
        assert len(data["sources"]) == 1
        assert len(data["call_chain"]) == 3
        assert data["token_usage"]["total_tokens"] == 150

    def test_code_stats_response_structure(self, pipeline_config):
        """Test CodeStatsResponse structure."""
        from code_assistance_pipeline.models.query_models import CodeStatsResponse

        response = CodeStatsResponse(
            success=True,
            total_methods=1000,
            total_classes=200,
            code_entities={
                "methods": 1000,
                "classes": 200,
            },
            timestamp="2024-01-15T12:00:00Z",
        )

        data = response.model_dump()

        assert data["success"] is True
        assert data["code_entities"]["methods"] == 1000


# =============================================================================
# E2E API Tests (with actual HTTP client)
# =============================================================================

class TestEndpointsE2E:
    """End-to-end API tests."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_query_endpoint_e2e(self, pipeline_config):
        """Test query endpoint end-to-end with mock pipeline."""
        from fastapi import FastAPI
        from httpx import AsyncClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import (
            CodeQueryResponse,
            TokenUsage,
            TimingInfo,
        )

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.process_query.return_value = CodeQueryResponse(
                success=True,
                response_id="resp_e2e_test",
                query="Test query",
                answer="Test answer from E2E test",
                sources=[],
                token_usage=TokenUsage(
                    prompt_tokens=50,
                    completion_tokens=25,
                    total_tokens=75,
                ),
                timing=TimingInfo(
                    retrieval_ms=10,
                    generation_ms=100,
                    total_ms=110,
                ),
            )
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/code/query",
                    json={
                        "query": "Test query",
                        "project": "Test",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["answer"] == "Test answer from E2E test"
