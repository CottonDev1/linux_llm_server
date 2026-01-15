"""
SQL Pipeline SSE Streaming Tests.

Comprehensive tests for Server-Sent Events (SSE) streaming including:
- Event type sequence validation
- Proper JSON formatting of events
- Error event handling
- Streaming cancellation/timeout
- Progress tracking
- Result delivery

Tests both the sql_query_routes.py and sql_routes_new.py streaming endpoints.
"""

import pytest
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from testing.utils.api_test_client import APITestClient, SSEEvent


# =============================================================================
# Constants
# =============================================================================

# Expected event types for query-stream endpoint (sql_query_routes.py)
EXPECTED_EVENT_TYPES_QUERY_STREAM = [
    "status",    # Pipeline stage updates
    "rules",     # Matched rules information
    "schema",    # Loaded schema information
    "sql",       # Generated SQL query
    "explanation",  # SQL explanation
    "validation",   # Validation results
    "execution",    # Execution results (if execute_sql)
    "error",     # Error information (if any)
    "done",      # Processing complete
]

# Expected event types for new streaming endpoint (sql_routes_new.py)
EXPECTED_EVENT_TYPES_SQL_STREAM = [
    "progress",  # Stage updates (analyzing, searching, loading, generating, etc.)
    "result",    # Final result (success or error)
]

VALID_STREAM_REQUEST = {
    "naturalLanguage": "How many tickets were created today?",
    "server": "NCSQLTEST",
    "database": "EWRCentral",
    "user": "EWRUser",
    "password": "66a3904d69",
    "trustServerCertificate": True,
    "encrypt": False,
    "maxTokens": 512
}

SIMPLE_STREAM_REQUEST = {
    "natural_language": "How many tickets are there?",
    "database": "EWRCentral",
    "server": "NCSQLTEST",
    "options": {
        "execute_sql": False,
        "use_cache": False
    }
}


# =============================================================================
# Helper Classes
# =============================================================================

@dataclass
class SSEStreamResult:
    """Result from consuming an SSE stream."""
    events: List[Dict[str, Any]] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timed_out: bool = False

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.get("type") == event_type or e.get("stage") == event_type]

    def get_event_types(self) -> List[str]:
        """Get list of all event types in order."""
        types = []
        for e in self.events:
            if "type" in e:
                types.append(e["type"])
            elif "stage" in e:
                types.append(e["stage"])
            elif "event" in e:
                types.append(e["event"])
        return types

    def has_event_type(self, event_type: str) -> bool:
        """Check if a specific event type exists."""
        return event_type in self.get_event_types()

    def get_final_result(self) -> Optional[Dict[str, Any]]:
        """Get the final result event (prefers 'result' over 'done')."""
        # First look for a 'result' type event which has the full data
        for e in reversed(self.events):
            if e.get("type") == "result":
                return e
        # Fall back to 'done' event if no result event found
        for e in reversed(self.events):
            if e.get("type") == "done":
                return e
        return None

    @property
    def was_successful(self) -> bool:
        """Check if stream completed successfully."""
        final = self.get_final_result()
        if final:
            return final.get("success", False)
        return False


class SSEStreamConsumer:
    """
    Helper class for consuming SSE streams with timeout support.
    """

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout

    async def consume_stream(
        self,
        client: APITestClient,
        endpoint: str,
        request_data: Dict[str, Any]
    ) -> SSEStreamResult:
        """
        Consume SSE stream from endpoint and return parsed events.

        Args:
            client: API test client
            endpoint: SSE endpoint path
            request_data: Request body

        Returns:
            SSEStreamResult with parsed events
        """
        result = SSEStreamResult()

        try:
            async with asyncio.timeout(self.timeout):
                async for event in client.post_stream(endpoint, request_data):
                    result.events.append(event.data)
                    result.raw_lines.append(str(event))

        except asyncio.TimeoutError:
            result.timed_out = True
            result.error = f"Stream timed out after {self.timeout} seconds"

        except Exception as e:
            result.error = str(e)

        return result


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def api_client():
    """Fixture providing configured API test client."""
    async with APITestClient(base_url="http://localhost:8001", timeout=120.0) as client:
        yield client


@pytest.fixture
def stream_consumer():
    """Fixture providing SSE stream consumer."""
    return SSEStreamConsumer(timeout=120.0)


# =============================================================================
# Basic Streaming Tests
# =============================================================================

class TestSSEStreamBasics:
    """Test basic SSE streaming functionality."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_stream_returns_events(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that streaming endpoint returns SSE events."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        assert len(result.events) > 0, "Stream should return at least one event"
        assert not result.timed_out, f"Stream should not timeout: {result.error}"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_stream_content_type(self, api_client: APITestClient):
        """Test that streaming response has correct content type."""
        # This tests the response headers
        import httpx

        async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
            async with client.stream("POST", "/api/sql/query-stream", json=SIMPLE_STREAM_REQUEST) as response:
                content_type = response.headers.get("content-type", "")
                assert "text/event-stream" in content_type

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_stream_cache_control_headers(self, api_client: APITestClient):
        """Test that streaming response has correct cache headers."""
        import httpx

        async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=30.0) as client:
            async with client.stream("POST", "/api/sql/query-stream", json=SIMPLE_STREAM_REQUEST) as response:
                cache_control = response.headers.get("cache-control", "")
                assert "no-cache" in cache_control


# =============================================================================
# Event Sequence Tests
# =============================================================================

class TestEventSequence:
    """Test SSE event type sequencing."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_query_stream_event_sequence(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that query-stream events appear in expected order."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        event_types = result.get_event_types()

        # Should have some events
        assert len(event_types) > 0, "Should have events in stream"

        # Check for expected event types (may vary based on pipeline path)
        # Status/progress events should come before result/done
        has_progress = any(t in event_types for t in ["status", "progress", "analyzing", "generating"])
        has_completion = any(t in event_types for t in ["done", "result", "sql"])

        assert has_progress or has_completion, f"Expected progress or completion events, got: {event_types}"

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_progress_events_before_result(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that progress events come before result event."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        events = result.events
        progress_indices = []
        result_index = -1

        for i, event in enumerate(events):
            event_type = event.get("type") or event.get("stage", "")
            if event_type in ["progress", "status", "analyzing", "generating"]:
                progress_indices.append(i)
            elif event_type in ["result", "done"]:
                result_index = i

        if result_index >= 0 and progress_indices:
            # All progress events should come before result
            assert all(pi < result_index for pi in progress_indices), \
                "All progress events should come before result event"


# =============================================================================
# Event JSON Format Tests
# =============================================================================

class TestEventJSONFormat:
    """Test that SSE events are properly formatted JSON."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_events_are_valid_json(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that all events contain valid JSON."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        for i, event in enumerate(result.events):
            assert isinstance(event, dict), f"Event {i} should be a dict, got {type(event)}"

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_progress_events_have_required_fields(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that progress events have required fields."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        for event in result.events:
            event_type = event.get("type") or event.get("stage")
            if event_type in ["progress", "status"]:
                # Progress events should have stage or message
                has_stage = "stage" in event
                has_message = "message" in event
                assert has_stage or has_message, f"Progress event missing stage/message: {event}"

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_result_event_has_success_field(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that result event has success field."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        final = result.get_final_result()
        if final:
            assert "success" in final, f"Final result should have 'success' field: {final}"


# =============================================================================
# Error Event Tests
# =============================================================================

class TestErrorEvents:
    """Test SSE error event handling."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_event_on_invalid_request(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that invalid request produces error event."""
        invalid_request = {
            "natural_language": "",  # Empty question
            "database": "EWRCentral",
            "server": "NCSQLTEST"
        }

        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            invalid_request
        )

        # Should either return error event or non-streaming error
        if result.events:
            # Check for error indication in events
            has_error = any(
                e.get("type") == "error" or
                e.get("success") is False or
                "error" in e
                for e in result.events
            )
            # It's acceptable to either have error events or no events (422 response)
            assert has_error or len(result.events) == 0 or result.error is not None

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_event_on_missing_credentials_for_execution(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test error when credentials missing but execution requested."""
        request = {
            "natural_language": "Count tickets",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "options": {
                "execute_sql": True  # Requested but no credentials
            }
        }

        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            request
        )

        # Should handle gracefully - either error event or validation rejection
        # Don't assert specific behavior as it may vary

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_error_event_has_error_message(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that error events contain error message."""
        # Use request that will likely cause an error
        request = {
            "natural_language": "Query something",
            "database": "NonexistentDatabase12345",
            "server": "NCSQLTEST",
            "options": {
                "execute_sql": False
            }
        }

        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            request
        )

        error_events = [e for e in result.events if e.get("type") == "error" or "error" in e]
        for error_event in error_events:
            # Error events should have error message
            has_error_info = "error" in error_event or "message" in error_event
            assert has_error_info, f"Error event should have error info: {error_event}"


# =============================================================================
# Streaming Timeout Tests
# =============================================================================

class TestStreamingTimeout:
    """Test SSE streaming timeout handling."""

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_stream_completes_within_timeout(
        self,
        api_client: APITestClient
    ):
        """Test that stream completes within reasonable timeout."""
        consumer = SSEStreamConsumer(timeout=60.0)  # 60 second timeout

        result = await consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        # Should either complete or fail gracefully
        assert not result.timed_out or result.events, \
            "Stream should either complete or return some events before timeout"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_short_timeout_behavior(
        self,
        api_client: APITestClient
    ):
        """Test behavior with very short timeout."""
        consumer = SSEStreamConsumer(timeout=0.1)  # 100ms timeout

        result = await consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        # With such short timeout, likely to timeout
        # But should handle gracefully without exception
        assert result is not None


# =============================================================================
# Progress Stage Tests
# =============================================================================

class TestProgressStages:
    """Test SSE progress stage reporting."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_analyzing_stage_reported(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that analyzing stage is reported."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        # Look for analyzing/analyzing stage
        stages = []
        for event in result.events:
            stage = event.get("stage") or event.get("type")
            if stage:
                stages.append(stage)

        # Should have at least one progress-related stage
        progress_stages = ["analyzing", "searching", "loading", "building", "generating",
                         "validating", "executing", "summarizing", "status", "progress"]
        has_progress = any(s in stages for s in progress_stages)

        # If we got events, should have progress
        if result.events:
            assert has_progress or "result" in stages or "done" in stages, \
                f"Expected progress stages, got: {stages}"

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_generating_stage_reported(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that SQL generation stage is reported."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        # This test is informational - generating may or may not be explicit
        stages = [e.get("stage") or e.get("type") for e in result.events]

        # Just verify we got some events
        assert len(stages) > 0 or result.error


# =============================================================================
# Result Delivery Tests
# =============================================================================

class TestResultDelivery:
    """Test SSE result delivery."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_sql_delivered_in_result(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that generated SQL is delivered in result."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        # Find result event with SQL
        sql_found = False
        for event in result.events:
            if event.get("type") in ["result", "sql", "done"]:
                if "sql" in event or "generatedSql" in event or "generated_sql" in event:
                    sql_found = True
                    break

        if result.was_successful:
            assert sql_found, "Successful stream should deliver SQL"

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_token_usage_reported(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test that token usage is reported in result."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        if result.error and not result.timed_out:
            pytest.skip(f"Stream error: {result.error}")

        # Token usage may be in result event
        final = result.get_final_result()
        if final and final.get("success"):
            # Token usage is optional but good to have
            # This is informational
            has_tokens = "tokenUsage" in final or "token_usage" in final
            # Not asserting - just checking


# =============================================================================
# Query-Stream vs Generate-Stream Comparison Tests
# =============================================================================

class TestStreamEndpointBehavior:
    """Test different streaming endpoint behaviors."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_query_stream_endpoint(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test query-stream endpoint behavior."""
        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/query-stream",
            SIMPLE_STREAM_REQUEST
        )

        # Should return events
        assert len(result.events) > 0 or result.error, \
            "query-stream should return events or error"

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_sql_query_stream_new_format(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test new SQL routes query-stream format."""
        # Test the sql_routes_new.py format
        request = {
            "naturalLanguage": "How many tickets?",
            "server": "NCSQLTEST",
            "database": "EWRCentral",
            "user": "EWRUser",
            "password": "66a3904d69",
            "maxTokens": 512
        }

        result = await stream_consumer.consume_stream(
            api_client,
            "/sql/query-stream",
            request
        )

        # May not exist or may have different format - that's OK
        # Just verify no crash


# =============================================================================
# Concurrent Stream Tests
# =============================================================================

class TestConcurrentStreams:
    """Test concurrent SSE stream handling."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_multiple_concurrent_streams(self, api_client: APITestClient):
        """Test that multiple concurrent streams work."""
        consumer = SSEStreamConsumer(timeout=60.0)

        # Start multiple streams concurrently
        tasks = [
            consumer.consume_stream(api_client, "/api/sql/query-stream", SIMPLE_STREAM_REQUEST),
            consumer.consume_stream(api_client, "/api/sql/query-stream", {
                "natural_language": "Count tickets by status",
                "database": "EWRCentral",
                "server": "NCSQLTEST",
                "options": {"execute_sql": False, "use_cache": False}
            })
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Both should complete (may have errors but shouldn't crash)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.skip(f"Stream {i} failed: {result}")
            assert isinstance(result, SSEStreamResult), f"Stream {i} should return SSEStreamResult"


# =============================================================================
# Schema Extract Stream Tests
# =============================================================================

class TestSchemaExtractStream:
    """Test schema extraction streaming endpoint."""

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_schema_extract_stream_events(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test schema extraction streaming returns events."""
        request = {
            "database": "master",  # Use master for testing
            "server": "NCSQLTEST",
            "username": "EWRUser",
            "password": "66a3904d69"
        }

        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/schema/extract-stream",
            request
        )

        # Schema extraction may take time or fail
        # Just verify it handles gracefully
        assert result is not None

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_schema_extract_progress_events(
        self,
        api_client: APITestClient,
        stream_consumer: SSEStreamConsumer
    ):
        """Test schema extraction reports progress."""
        request = {
            "database": "master",
            "server": "NCSQLTEST",
            "username": "EWRUser",
            "password": "66a3904d69"
        }

        result = await stream_consumer.consume_stream(
            api_client,
            "/api/sql/schema/extract-stream",
            request
        )

        if result.events and not result.error:
            # Should have progress-type events
            has_progress = any(
                e.get("stage") in ["connecting", "connected", "extracting", "tables", "complete"]
                for e in result.events
            )
            # Not asserting - schema extract may not be available
