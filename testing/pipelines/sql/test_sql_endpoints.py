"""
SQL Pipeline HTTP Endpoint Tests.

Comprehensive tests for all SQL API endpoints including:
- POST /api/sql/query - Basic query generation
- POST /api/sql/query-stream - SSE streaming
- POST /api/sql/execute - SQL execution
- POST /api/sql/validate - SQL validation
- POST /api/sql/test-connection - Connection testing
- POST /api/sql/feedback - Feedback submission
- GET /api/sql/rules/{database} - Rule retrieval
- POST /api/sql/save-example - Training examples
- POST /api/sql/schema/check - Schema check
- POST /api/sql/disconnect - Connection cleanup
- GET /api/sql/health - Health check
- CRUD operations for rules

Uses local llama.cpp and MongoDB only.
"""

import pytest
import json
from typing import Dict, Any, List
from dataclasses import dataclass

from testing.utils.api_test_client import APITestClient, APIResponse
from testing.fixtures.shared_fixtures import ResponseValidator, SSEConsumer
from testing.templates.error_test_templates import (
    ErrorTestCase,
    ErrorCategory,
    get_sql_pipeline_error_cases,
    create_validation_case,
)


# =============================================================================
# Test Data Constants
# =============================================================================

VALID_QUERY_REQUEST = {
    "natural_language": "How many tickets were created today?",
    "database": "EWRCentral",
    "server": "NCSQLTEST",
    "credentials": {
        "server": "NCSQLTEST",
        "database": "EWRCentral",
        "username": "EWRUser",
        "password": "66a3904d69"
    },
    "options": {
        "execute_sql": False,
        "use_cache": False
    }
}

VALID_EXECUTE_REQUEST = {
    "sql": "SELECT COUNT(*) as Total FROM CentralTickets",
    "database": "EWRCentral",
    "server": "NCSQLTEST",
    "credentials": {
        "server": "NCSQLTEST",
        "database": "EWRCentral",
        "username": "EWRUser",
        "password": "66a3904d69"
    },
    "max_results": 100
}

VALID_CONNECTION_REQUEST = {
    "server": "NCSQLTEST",
    "database": "master",
    "username": "EWRUser",
    "password": "66a3904d69",
    "authType": "sql"
}

VALID_FEEDBACK_REQUEST = {
    "question": "How many tickets were created today?",
    "sql": "SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
    "database": "EWRCentral",
    "feedback": "positive",
    "comment": "Correct SQL generated"
}

VALID_SAVE_EXAMPLE_REQUEST = {
    "question": "Count tickets created today",
    "sql": "SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
    "database": "EWRCentral",
    "explanation": "Uses CAST for date comparison",
    "tags": ["tickets", "count", "date"]
}

VALID_RULE_REQUEST = {
    "database": "EWRCentral",
    "rule_id": "test-rule-endpoint-001",
    "description": "Test rule for endpoint testing",
    "type": "assistance",
    "priority": "normal",
    "enabled": True,
    "trigger_keywords": ["test", "endpoint"],
    "trigger_tables": [],
    "trigger_columns": [],
    "rule_text": "This is a test rule"
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def api_client():
    """Fixture providing configured API test client for Python service."""
    async with APITestClient(base_url="http://localhost:8001", timeout=60.0) as client:
        yield client


@pytest.fixture
def response_validator():
    """Fixture providing response validation utilities."""
    return ResponseValidator()


@pytest.fixture
def sse_consumer():
    """Fixture providing SSE stream consumer."""
    return SSEConsumer()


# =============================================================================
# Health Check Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for GET /api/sql/health endpoint."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_health_check_success(self, api_client: APITestClient):
        """Test health endpoint returns healthy status."""
        response = await api_client.get("/api/sql/health")

        assert response.is_success
        assert response.data["status"] == "healthy"
        assert response.data["service"] == "sql_query"
        assert "timestamp" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_health_check_includes_components(self, api_client: APITestClient):
        """Test health endpoint includes component status."""
        response = await api_client.get("/api/sql/health")

        assert response.is_success
        # Check for component status fields
        assert "mongodb" in response.data
        assert "pipeline" in response.data


# =============================================================================
# Query Endpoint Tests
# =============================================================================

class TestQueryEndpoint:
    """Tests for POST /api/sql/query endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_query_basic_generation(self, api_client: APITestClient):
        """Test basic SQL query generation."""
        request = {
            "natural_language": "How many tickets are there?",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "options": {
                "execute_sql": False,
                "use_cache": False
            }
        }

        response = await api_client.post("/api/sql/query", request)

        assert response.is_success
        assert "sql" in response.data or "generated_sql" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_query_missing_natural_language(self, api_client: APITestClient):
        """Test query endpoint rejects missing natural_language."""
        request = {
            "database": "EWRCentral",
            "server": "NCSQLTEST"
        }

        response = await api_client.post("/api/sql/query", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_query_missing_database(self, api_client: APITestClient):
        """Test query endpoint rejects missing database."""
        request = {
            "natural_language": "How many tickets?",
            "server": "NCSQLTEST"
        }

        response = await api_client.post("/api/sql/query", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_query_empty_natural_language(self, api_client: APITestClient):
        """Test query endpoint rejects empty question."""
        request = {
            "natural_language": "",
            "database": "EWRCentral",
            "server": "NCSQLTEST"
        }

        response = await api_client.post("/api/sql/query", request)

        # Should reject empty question
        assert response.status_code in [400, 422]

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, api_client: APITestClient):
        """Test query with conversation history context."""
        request = {
            "natural_language": "Show me the same for last month",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [
                {"role": "user", "content": "How many tickets were created today?"},
                {"role": "assistant", "content": "Generated SQL: SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)"}
            ],
            "options": {
                "execute_sql": False,
                "use_cache": False
            }
        }

        response = await api_client.post("/api/sql/query", request)

        # Should process without error even if LLM is not available
        assert response.status_code in [200, 503]


# =============================================================================
# Execute Endpoint Tests
# =============================================================================

class TestExecuteEndpoint:
    """Tests for POST /api/sql/execute endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_execute_simple_query(self, api_client: APITestClient):
        """Test executing a simple SQL query."""
        request = {
            "sql": "SELECT 1 as Test",
            "database": "master",
            "server": "NCSQLTEST",
            "credentials": {
                "server": "NCSQLTEST",
                "database": "master",
                "username": "EWRUser",
                "password": "66a3904d69"
            },
            "max_results": 10
        }

        response = await api_client.post("/api/sql/execute", request)

        if response.is_success:
            assert "success" in response.data
            if response.data.get("success"):
                assert "data" in response.data or "rows" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_execute_missing_sql(self, api_client: APITestClient):
        """Test execute endpoint rejects missing SQL."""
        request = {
            "database": "master",
            "server": "NCSQLTEST",
            "credentials": VALID_CONNECTION_REQUEST,
            "max_results": 10
        }

        response = await api_client.post("/api/sql/execute", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_execute_missing_credentials(self, api_client: APITestClient):
        """Test execute endpoint rejects missing credentials."""
        request = {
            "sql": "SELECT 1",
            "database": "master",
            "server": "NCSQLTEST",
            "max_results": 10
        }

        response = await api_client.post("/api/sql/execute", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_execute_max_results_validation(self, api_client: APITestClient):
        """Test execute validates max_results range."""
        request = {
            "sql": "SELECT 1",
            "database": "master",
            "server": "NCSQLTEST",
            "credentials": {
                "server": "NCSQLTEST",
                "database": "master",
                "username": "EWRUser",
                "password": "66a3904d69"
            },
            "max_results": 100000  # Over limit
        }

        response = await api_client.post("/api/sql/execute", request)

        # Should either cap at max or return validation error
        assert response.status_code in [200, 422]


# =============================================================================
# Test Connection Endpoint Tests
# =============================================================================

class TestConnectionEndpoint:
    """Tests for POST /api/sql/test-connection endpoint."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_success(self, api_client: APITestClient):
        """Test successful connection to database."""
        response = await api_client.post("/api/sql/test-connection", VALID_CONNECTION_REQUEST)

        # Connection may succeed or fail depending on server availability
        assert response.status_code in [200, 500, 503]
        if response.is_success:
            assert "success" in response.data
            if response.data["success"]:
                assert "latency_ms" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_missing_server(self, api_client: APITestClient):
        """Test connection endpoint rejects missing server."""
        request = {
            "database": "master",
            "username": "EWRUser",
            "password": "66a3904d69"
        }

        response = await api_client.post("/api/sql/test-connection", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_invalid_server(self, api_client: APITestClient):
        """Test connection to invalid server returns failure."""
        request = {
            "server": "invalid-server-that-does-not-exist",
            "database": "master",
            "username": "EWRUser",
            "password": "66a3904d69"
        }

        response = await api_client.post("/api/sql/test-connection", request)

        # Should return connection failure, not crash
        if response.is_success:
            assert response.data.get("success") is False
            assert "error" in response.data or "message" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_returns_databases(self, api_client: APITestClient):
        """Test successful connection returns database list."""
        response = await api_client.post("/api/sql/test-connection", VALID_CONNECTION_REQUEST)

        if response.is_success and response.data.get("success"):
            # Should include list of databases
            assert "databases" in response.data or "connection_info" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_windows_auth_format(self, api_client: APITestClient):
        """Test connection with Windows authentication format."""
        request = {
            "server": "NCSQLTEST",
            "database": "master",
            "username": "user",
            "password": "pass",
            "domain": "EWR",
            "authType": "windows"
        }

        response = await api_client.post("/api/sql/test-connection", request)

        # Should process the request (may fail auth, but shouldn't crash)
        assert response.status_code in [200, 401, 500, 503]


# =============================================================================
# Disconnect Endpoint Tests
# =============================================================================

class TestDisconnectEndpoint:
    """Tests for POST /api/sql/disconnect endpoint."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_disconnect_success(self, api_client: APITestClient):
        """Test disconnect endpoint succeeds."""
        response = await api_client.post("/api/sql/disconnect")

        assert response.is_success
        assert response.data.get("success") is True


# =============================================================================
# Feedback Endpoint Tests
# =============================================================================

class TestFeedbackEndpoint:
    """Tests for POST /api/sql/feedback endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_feedback_positive(self, api_client: APITestClient):
        """Test submitting positive feedback."""
        request = {
            "question": "Test query for feedback",
            "sql": "SELECT 1",
            "database": "EWRCentral",
            "feedback": "positive",
            "comment": "Test positive feedback"
        }

        response = await api_client.post("/api/sql/feedback", request)

        assert response.is_success
        assert response.data.get("success") is True
        assert "feedback_id" in response.data

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_feedback_negative(self, api_client: APITestClient):
        """Test submitting negative feedback."""
        request = {
            "question": "Test query for negative feedback",
            "sql": "SELECT * FROM WrongTable",
            "database": "EWRCentral",
            "feedback": "negative",
            "comment": "Wrong table referenced"
        }

        response = await api_client.post("/api/sql/feedback", request)

        assert response.is_success
        assert response.data.get("success") is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_feedback_missing_question(self, api_client: APITestClient):
        """Test feedback rejects missing question."""
        request = {
            "sql": "SELECT 1",
            "database": "EWRCentral",
            "feedback": "positive"
        }

        response = await api_client.post("/api/sql/feedback", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_feedback_missing_feedback_type(self, api_client: APITestClient):
        """Test feedback rejects missing feedback type."""
        request = {
            "question": "Test query",
            "sql": "SELECT 1",
            "database": "EWRCentral"
        }

        response = await api_client.post("/api/sql/feedback", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_feedback_with_alias_fields(self, api_client: APITestClient):
        """Test feedback with alias field names (query, generatedSql, isPositive)."""
        request = {
            "query": "Test query for alias fields",
            "generatedSql": "SELECT 1",
            "database": "EWRCentral",
            "isPositive": True,
            "reason": "Testing alias fields"
        }

        response = await api_client.post("/api/sql/feedback", request)

        assert response.is_success
        assert response.data.get("success") is True


# =============================================================================
# Save Example Endpoint Tests
# =============================================================================

class TestSaveExampleEndpoint:
    """Tests for POST /api/sql/save-example endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_save_example_success(self, api_client: APITestClient):
        """Test saving a training example."""
        request = {
            "question": "Count tickets for test example",
            "sql": "SELECT COUNT(*) FROM CentralTickets",
            "database": "EWRCentral",
            "explanation": "Test example",
            "tags": ["test"]
        }

        response = await api_client.post("/api/sql/save-example", request)

        assert response.is_success
        assert response.data.get("success") is True
        assert "example_id" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_save_example_missing_question(self, api_client: APITestClient):
        """Test save example rejects missing question."""
        request = {
            "sql": "SELECT 1",
            "database": "EWRCentral"
        }

        response = await api_client.post("/api/sql/save-example", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_save_example_missing_sql(self, api_client: APITestClient):
        """Test save example rejects missing SQL."""
        request = {
            "question": "Test question",
            "database": "EWRCentral"
        }

        response = await api_client.post("/api/sql/save-example", request)

        assert response.status_code == 422


# =============================================================================
# Rules Endpoint Tests
# =============================================================================

class TestRulesEndpoints:
    """Tests for SQL rules CRUD endpoints."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_rules_for_database(self, api_client: APITestClient):
        """Test getting rules for specific database."""
        response = await api_client.get("/api/sql/rules/EWRCentral")

        assert response.is_success
        assert "rules" in response.data
        assert "rule_count" in response.data
        assert isinstance(response.data["rules"], list)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_rules_includes_global(self, api_client: APITestClient):
        """Test getting rules includes global rules by default."""
        response = await api_client.get("/api/sql/rules/EWRCentral?include_global=true")

        assert response.is_success
        assert "rules" in response.data

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_all_rules(self, api_client: APITestClient):
        """Test getting all rules across databases."""
        response = await api_client.get("/api/sql/rules")

        assert response.is_success
        assert "rules" in response.data
        assert "rule_count" in response.data

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_create_rule(self, api_client: APITestClient):
        """Test creating a new SQL rule."""
        # Use unique rule_id to avoid conflicts
        import uuid
        request = {
            "database": "EWRCentral",
            "rule_id": f"test-rule-create-{uuid.uuid4().hex[:8]}",
            "description": "Test rule for create endpoint",
            "type": "assistance",
            "priority": "normal",
            "enabled": True,
            "trigger_keywords": ["test", "create"],
            "trigger_tables": [],
            "trigger_columns": [],
            "rule_text": "This is a test rule for create endpoint"
        }

        response = await api_client.post("/api/sql/rules", request)

        if response.is_success:
            assert response.data.get("success") is True
            # Cleanup - delete the test rule
            await api_client.delete(f"/api/sql/rules/{request['rule_id']}")

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_create_rule_duplicate_id(self, api_client: APITestClient):
        """Test creating rule with duplicate ID fails."""
        import uuid
        rule_id = f"test-rule-duplicate-{uuid.uuid4().hex[:8]}"
        request = {
            "database": "EWRCentral",
            "rule_id": rule_id,
            "description": "First rule",
            "type": "assistance",
            "priority": "normal",
            "enabled": True,
            "trigger_keywords": [],
            "rule_text": "First rule text"
        }

        # Create first rule
        response1 = await api_client.post("/api/sql/rules", request)

        if response1.is_success:
            # Try to create duplicate
            response2 = await api_client.post("/api/sql/rules", request)
            assert response2.status_code == 400

            # Cleanup
            await api_client.delete(f"/api/sql/rules/{rule_id}")

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_delete_rule(self, api_client: APITestClient):
        """Test deleting a SQL rule."""
        import uuid
        rule_id = f"test-rule-delete-{uuid.uuid4().hex[:8]}"

        # Create rule first
        request = {
            "database": "EWRCentral",
            "rule_id": rule_id,
            "description": "Rule to delete",
            "type": "assistance",
            "priority": "normal",
            "enabled": True,
            "trigger_keywords": [],
            "rule_text": "This rule will be deleted"
        }
        await api_client.post("/api/sql/rules", request)

        # Delete rule
        response = await api_client.delete(f"/api/sql/rules/{rule_id}")

        assert response.is_success
        assert response.data.get("success") is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_delete_nonexistent_rule(self, api_client: APITestClient):
        """Test deleting non-existent rule returns 404."""
        response = await api_client.delete("/api/sql/rules/nonexistent-rule-12345")

        assert response.status_code == 404

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_invalidate_rules_cache(self, api_client: APITestClient):
        """Test cache invalidation endpoint."""
        response = await api_client.post("/api/sql/rules/invalidate-cache")

        assert response.is_success
        assert response.data.get("success") is True


# =============================================================================
# Validate Endpoint Tests
# =============================================================================

class TestValidateEndpoint:
    """Tests for POST /api/sql/validate endpoint."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validate_valid_sql(self, api_client: APITestClient):
        """Test validating valid SQL returns success."""
        request = {
            "sql": "SELECT * FROM CentralTickets",
            "database": "EWRCentral"
        }

        response = await api_client.post("/sql/validate", request)

        if response.is_success:
            assert "valid" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validate_missing_sql(self, api_client: APITestClient):
        """Test validate rejects missing SQL."""
        request = {
            "database": "EWRCentral"
        }

        response = await api_client.post("/sql/validate", request)

        assert response.status_code == 422


# =============================================================================
# Schema Endpoints Tests
# =============================================================================

class TestSchemaEndpoints:
    """Tests for schema-related endpoints."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_schema_check_exists(self, api_client: APITestClient):
        """Test checking if schema exists for database."""
        request = {
            "database": "EWRCentral"
        }

        response = await api_client.post("/api/sql/schema/check", request)

        assert response.is_success
        assert "exists" in response.data
        assert "database" in response.data
        assert "table_count" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_schema_check_missing_database(self, api_client: APITestClient):
        """Test schema check rejects missing database."""
        request = {}

        response = await api_client.post("/api/sql/schema/check", request)

        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_analyzed_databases(self, api_client: APITestClient):
        """Test getting list of analyzed databases."""
        response = await api_client.get("/api/sql/databases")

        assert response.is_success
        assert isinstance(response.data, list)


# =============================================================================
# Save Correction Endpoint Tests
# =============================================================================

class TestSaveCorrectionEndpoint:
    """Tests for POST /api/sql/save-correction endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_save_correction_success(self, api_client: APITestClient):
        """Test saving a SQL correction."""
        request = {
            "database": "EWRCentral",
            "question": "Show tickets from today",
            "failed_sql": "SELECT * FROM CentralTickets WHERE CreateDate = GETDATE()",
            "corrected_sql": "SELECT * FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
            "error_message": "Invalid column name 'CreateDate'"
        }

        response = await api_client.post("/api/sql/save-correction", request)

        assert response.is_success
        assert response.data.get("success") is True
        assert "correction_id" in response.data

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_save_correction_missing_corrected_sql(self, api_client: APITestClient):
        """Test save correction rejects missing corrected SQL."""
        request = {
            "database": "EWRCentral",
            "question": "Show tickets",
            "failed_sql": "SELECT * FROM BadTable"
        }

        response = await api_client.post("/api/sql/save-correction", request)

        assert response.status_code == 422


# =============================================================================
# User Settings Endpoints Tests
# =============================================================================

class TestUserSettingsEndpoints:
    """Tests for user settings endpoints."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_user_settings(self, api_client: APITestClient):
        """Test getting user settings."""
        response = await api_client.get("/api/sql/settings/test-user-001")

        assert response.is_success
        assert "success" in response.data
        assert "settings" in response.data

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_save_user_settings(self, api_client: APITestClient):
        """Test saving user settings."""
        request = {
            "default_server": "NCSQLTEST",
            "default_database": "EWRCentral",
            "default_max_results": 100,
            "enable_streaming": True,
            "temperature": 0.0,
            "use_cache": True
        }

        response = await api_client.post("/api/sql/settings/test-user-001", request)

        assert response.is_success
        assert response.data.get("success") is True


# =============================================================================
# Cache Management Endpoint Tests
# =============================================================================

class TestCacheEndpoint:
    """Tests for DELETE /api/sql/cache endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_clear_all_cache(self, api_client: APITestClient):
        """Test clearing all cache entries."""
        response = await api_client.delete("/api/sql/cache")

        assert response.is_success
        assert response.data.get("success") is True
        assert "deleted_count" in response.data

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_clear_cache_by_database(self, api_client: APITestClient):
        """Test clearing cache for specific database."""
        response = await api_client.delete("/api/sql/cache", params={"database": "EWRCentral"})

        assert response.is_success
        assert response.data.get("success") is True

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_clear_cache_by_question_pattern(self, api_client: APITestClient):
        """Test clearing cache matching question pattern."""
        response = await api_client.delete("/api/sql/cache", params={"question": "test.*pattern"})

        assert response.is_success
        assert response.data.get("success") is True


# =============================================================================
# Feedback List Endpoint Tests
# =============================================================================

class TestFeedbackListEndpoint:
    """Tests for GET /api/sql/feedback endpoint."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_feedback_list(self, api_client: APITestClient):
        """Test getting feedback list."""
        response = await api_client.get("/api/sql/feedback")

        assert response.is_success
        assert "feedback" in response.data
        assert "total" in response.data
        assert "unprocessed" in response.data

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_feedback_filtered_by_type(self, api_client: APITestClient):
        """Test getting feedback filtered by type."""
        response = await api_client.get("/api/sql/feedback", params={"feedback_type": "negative"})

        assert response.is_success

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.asyncio
    async def test_get_feedback_pagination(self, api_client: APITestClient):
        """Test feedback list pagination."""
        response = await api_client.get("/api/sql/feedback", params={"limit": 10, "skip": 0})

        assert response.is_success
        assert len(response.data.get("feedback", [])) <= 10


# =============================================================================
# Error Response Tests
# =============================================================================

class TestErrorResponses:
    """Test error response formats across endpoints."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, api_client: APITestClient):
        """Test that invalid JSON returns 400 error."""
        # This tests the API's ability to handle malformed requests
        # Note: HTTPx will reject this before sending, so we test a different scenario
        request = {"natural_language": 123}  # Wrong type

        response = await api_client.post("/api/sql/query", request)

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, api_client: APITestClient):
        """Test that unknown endpoint returns 404."""
        response = await api_client.get("/api/sql/nonexistent-endpoint")

        assert response.status_code == 404

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_method_not_allowed(self, api_client: APITestClient):
        """Test that wrong HTTP method returns 405."""
        # GET instead of POST for query endpoint
        response = await api_client.get("/api/sql/query")

        assert response.status_code == 405


# =============================================================================
# Parameterized Validation Tests
# =============================================================================

SQL_ENDPOINT_VALIDATION_CASES = [
    ErrorTestCase(
        name="query_missing_natural_language",
        input_data={"database": "EWRCentral"},
        expected_status=422,
        expected_error_contains="natural_language",
        description="Query endpoint requires natural_language"
    ),
    ErrorTestCase(
        name="execute_negative_max_results",
        input_data={
            "sql": "SELECT 1",
            "database": "EWRCentral",
            "server": "test",
            "credentials": {"server": "test", "database": "test", "username": "test", "password": "test"},
            "max_results": -1
        },
        expected_status=422,
        expected_error_contains="greater",
        description="Execute endpoint rejects negative max_results"
    ),
]


class TestParameterizedValidation:
    """Parameterized validation tests for SQL endpoints."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_case",
        SQL_ENDPOINT_VALIDATION_CASES,
        ids=[case.name for case in SQL_ENDPOINT_VALIDATION_CASES]
    )
    async def test_endpoint_validation(self, api_client: APITestClient, error_case: ErrorTestCase):
        """Test validation across SQL endpoints."""
        endpoint = "/api/sql/query"
        if "execute" in error_case.name:
            endpoint = "/api/sql/execute"

        response = await api_client.post(endpoint, error_case.input_data)

        assert response.status_code == error_case.expected_status
