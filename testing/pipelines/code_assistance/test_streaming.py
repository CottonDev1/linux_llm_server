"""
SSE Streaming Tests
===================

Test Server-Sent Events (SSE) streaming for code assistance including:
- SSEEvent.to_sse() method
- Event sequence
- Token streaming
- Error events
- Stream endpoint integration
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, AsyncGenerator

from utils import generate_test_id
from fixtures.shared_fixtures import SSEConsumer


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_streaming_events() -> List[Dict[str, Any]]:
    """Create sample streaming events sequence."""
    return [
        {"event": "status", "data": {"status": "retrieving", "message": "Searching codebase..."}},
        {"event": "sources", "data": {
            "status": "sources",
            "sources": [{"name": "BaleService.SaveBale", "type": "method"}],
            "call_chain": ["btnSave_Click", "SaveBale"],
            "response_id": "resp_123",
        }},
        {"event": "status", "data": {"status": "generating", "message": "Generating response..."}},
        {"event": "streaming", "data": {"status": "streaming", "token": "To"}},
        {"event": "streaming", "data": {"status": "streaming", "token": " save"}},
        {"event": "streaming", "data": {"status": "streaming", "token": " a"}},
        {"event": "streaming", "data": {"status": "streaming", "token": " bale"}},
        {"event": "complete", "data": {
            "status": "complete",
            "answer": "To save a bale",
            "response_id": "resp_123",
            "timing": {"total_ms": 1500},
        }},
    ]


@pytest.fixture
def sample_error_events() -> List[Dict[str, Any]]:
    """Create sample error events."""
    return [
        {"event": "status", "data": {"status": "retrieving", "message": "Searching codebase..."}},
        {"event": "error", "data": {"status": "error", "error": "Connection timeout"}},
    ]


# =============================================================================
# SSEEvent Model Tests
# =============================================================================

class TestSSEEventModel:
    """Test SSEEvent Pydantic model."""

    def test_create_basic_event(self, pipeline_config):
        """Test creating a basic SSE event."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="message",
            data={"content": "test"},
        )

        assert event.event == "message"
        assert event.data == {"content": "test"}

    def test_create_event_with_all_fields(self, pipeline_config):
        """Test creating event with all optional fields."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="status",
            data={"status": "generating"},
            id="evt_123",
            retry=3000,
        )

        assert event.event == "status"
        assert event.id == "evt_123"
        assert event.retry == 3000

    def test_event_defaults(self, pipeline_config):
        """Test SSEEvent default values."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent()

        assert event.event == "message"  # Default
        assert event.data is None
        assert event.id is None
        assert event.retry is None


# =============================================================================
# to_sse() Method Tests
# =============================================================================

class TestSSEEventToSSE:
    """Test SSEEvent.to_sse() conversion method."""

    def test_to_sse_basic_event(self, pipeline_config):
        """Test converting basic event to SSE format."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="status",
            data={"status": "generating"},
        )

        sse_str = event.to_sse()

        assert "event: status" in sse_str
        assert "data:" in sse_str
        assert "generating" in sse_str
        assert sse_str.endswith("\n\n")  # SSE terminator

    def test_to_sse_with_id(self, pipeline_config):
        """Test SSE format includes event ID."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="streaming",
            data={"token": "test"},
            id="evt_001",
        )

        sse_str = event.to_sse()

        assert "id: evt_001" in sse_str

    def test_to_sse_with_retry(self, pipeline_config):
        """Test SSE format includes retry directive."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="status",
            data={"status": "reconnecting"},
            retry=5000,
        )

        sse_str = event.to_sse()

        assert "retry: 5000" in sse_str

    def test_to_sse_json_data(self, pipeline_config):
        """Test that data is properly JSON serialized."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="sources",
            data={
                "sources": [{"name": "Test", "type": "method"}],
                "count": 5,
                "nested": {"key": "value"},
            },
        )

        sse_str = event.to_sse()

        # Extract data line and verify JSON
        lines = sse_str.split("\n")
        data_line = next(l for l in lines if l.startswith("data:"))
        data_json = data_line[6:]  # Remove "data: " prefix

        parsed = json.loads(data_json)
        assert parsed["count"] == 5
        assert len(parsed["sources"]) == 1

    def test_to_sse_string_data(self, pipeline_config):
        """Test SSE format with plain string data."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="message",
            data="Simple string message",
        )

        sse_str = event.to_sse()

        assert "data: Simple string message" in sse_str

    def test_to_sse_multiline_data(self, pipeline_config):
        """Test SSE format handles multiline data correctly."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="message",
            data="Line 1\nLine 2\nLine 3",
        )

        sse_str = event.to_sse()

        # Each line should have its own data: prefix
        assert "data: Line 1" in sse_str
        assert "data: Line 2" in sse_str
        assert "data: Line 3" in sse_str

    def test_to_sse_null_data(self, pipeline_config):
        """Test SSE format with null data."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="heartbeat",
            data=None,
        )

        sse_str = event.to_sse()

        # Should still have event type
        assert "event: heartbeat" in sse_str


# =============================================================================
# Event Sequence Tests
# =============================================================================

class TestEventSequence:
    """Test correct SSE event sequencing."""

    def test_standard_sequence_order(
        self, pipeline_config, sample_streaming_events
    ):
        """Test that events follow the standard sequence."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        events = [
            SSEEvent(event=e["event"], data=e["data"])
            for e in sample_streaming_events
        ]

        event_types = [e.event for e in events]

        # Verify expected sequence
        assert event_types[0] == "status"  # Start with retrieving
        assert "sources" in event_types  # Sources after retrieval
        assert event_types.count("streaming") >= 1  # At least one token
        assert event_types[-1] == "complete"  # End with complete

    def test_sources_event_precedes_streaming(
        self, pipeline_config, sample_streaming_events
    ):
        """Test that sources event comes before streaming tokens."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        events = [
            SSEEvent(event=e["event"], data=e["data"])
            for e in sample_streaming_events
        ]

        sources_idx = next(
            i for i, e in enumerate(events) if e.event == "sources"
        )
        first_streaming_idx = next(
            i for i, e in enumerate(events) if e.event == "streaming"
        )

        assert sources_idx < first_streaming_idx

    def test_error_terminates_sequence(
        self, pipeline_config, sample_error_events
    ):
        """Test that error event terminates the sequence."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        events = [
            SSEEvent(event=e["event"], data=e["data"])
            for e in sample_error_events
        ]

        # Error should be last event
        assert events[-1].event == "error"

        # Should not have complete event
        event_types = [e.event for e in events]
        assert "complete" not in event_types


# =============================================================================
# Token Streaming Tests
# =============================================================================

class TestTokenStreaming:
    """Test token-by-token streaming behavior."""

    def test_streaming_event_contains_token(self, pipeline_config):
        """Test that streaming events contain token data."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="streaming",
            data={"status": "streaming", "token": "Hello"},
        )

        assert event.data["token"] == "Hello"

    def test_multiple_tokens_form_complete_text(
        self, pipeline_config, sample_streaming_events
    ):
        """Test that streamed tokens form complete text."""
        streaming_events = [
            e for e in sample_streaming_events if e["event"] == "streaming"
        ]

        tokens = [e["data"]["token"] for e in streaming_events]
        full_text = "".join(tokens)

        assert full_text == "To save a bale"

    def test_complete_event_has_full_answer(
        self, pipeline_config, sample_streaming_events
    ):
        """Test that complete event contains full answer."""
        complete_event = next(
            e for e in sample_streaming_events if e["event"] == "complete"
        )

        assert complete_event["data"]["answer"] == "To save a bale"

    @pytest.mark.asyncio
    async def test_stream_consumer_collects_tokens(self, pipeline_config):
        """Test that SSE consumer can collect streaming tokens."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        # Simulate stream of events
        events = [
            SSEEvent(event="streaming", data={"token": "Hello"}),
            SSEEvent(event="streaming", data={"token": " "}),
            SSEEvent(event="streaming", data={"token": "World"}),
        ]

        tokens = []
        for event in events:
            if event.event == "streaming" and event.data:
                tokens.append(event.data.get("token", ""))

        full_text = "".join(tokens)
        assert full_text == "Hello World"


# =============================================================================
# Error Event Tests
# =============================================================================

class TestErrorEvents:
    """Test error event handling in streams."""

    def test_error_event_structure(self, pipeline_config):
        """Test error event has correct structure."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="error",
            data={"status": "error", "error": "Connection timeout"},
        )

        assert event.event == "error"
        assert event.data["status"] == "error"
        assert "error" in event.data

    def test_error_to_sse_format(self, pipeline_config):
        """Test error event SSE format."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        event = SSEEvent(
            event="error",
            data={"status": "error", "error": "LLM service unavailable"},
        )

        sse_str = event.to_sse()

        assert "event: error" in sse_str
        assert "LLM service unavailable" in sse_str

    def test_different_error_types(self, pipeline_config):
        """Test various error types can be represented."""
        from code_assistance_pipeline.models.query_models import SSEEvent

        error_types = [
            "Connection timeout",
            "MongoDB connection failed",
            "LLM generation error",
            "Rate limit exceeded",
        ]

        for error_msg in error_types:
            event = SSEEvent(
                event="error",
                data={"status": "error", "error": error_msg},
            )

            sse_str = event.to_sse()
            assert error_msg in sse_str


# =============================================================================
# Pipeline Stream Integration Tests
# =============================================================================

class TestPipelineStreaming:
    """Test streaming through the pipeline."""

    @pytest.mark.asyncio
    async def test_process_query_stream_yields_events(self, pipeline_config):
        """Test that process_query_stream yields SSE events."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            CodeQueryRequest,
            SSEEvent,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            # Setup mock retriever
            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = (
                [{"method_name": "SaveBale", "class_name": "BaleService"}],
                [],
                [],
                [],
            )
            mock_retriever.to_source_info.return_value = MagicMock()
            MockRetriever.return_value = mock_retriever

            # Setup mock context builder
            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("Context", [])
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            # Setup mock generator with streaming
            mock_generator = AsyncMock()

            async def mock_stream(*args, **kwargs):
                yield SSEEvent(event="streaming", data={"token": "Test"})
                yield SSEEvent(event="complete", data={"answer": "Test"})

            mock_generator.generate_stream_sse = mock_stream
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            request = CodeQueryRequest(query="Test query")

            events = []
            async for event in pipeline.process_query_stream(request):
                events.append(event)

            # Should have events
            assert len(events) > 0
            # All should be SSEEvent instances
            assert all(isinstance(e, SSEEvent) for e in events)

    @pytest.mark.asyncio
    async def test_stream_includes_status_events(self, pipeline_config):
        """Test that stream includes status events."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            CodeQueryRequest,
            SSEEvent,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.return_value = ([], [], [], [])
            mock_retriever.to_source_info.return_value = MagicMock()
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("", [])
            mock_context_builder.build_prompt.return_value = ""
            MockContextBuilder.return_value = mock_context_builder

            async def mock_stream(*args, **kwargs):
                yield SSEEvent(event="complete", data={"answer": "Done"})

            mock_generator = AsyncMock()
            mock_generator.generate_stream_sse = mock_stream
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            request = CodeQueryRequest(query="Test")

            events = []
            async for event in pipeline.process_query_stream(request):
                events.append(event)

            # Should have status event at start
            event_types = [e.event for e in events]
            assert "status" in event_types

    @pytest.mark.asyncio
    async def test_stream_includes_sources_event(self, pipeline_config):
        """Test that stream includes sources event."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            CodeQueryRequest,
            SSEEvent,
            SourceInfo,
            SourceType,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            mock_retriever = MagicMock()
            mock_retriever.initialize = AsyncMock()
            mock_retriever.retrieve_comprehensive = AsyncMock(return_value=(
                [{"method_name": "SaveBale", "class_name": "BaleService"}],
                [],
                [],
                ["BaleService.SaveBale"],
            ))
            mock_retriever.to_source_info.return_value = SourceInfo(
                type=SourceType.METHOD,
                name="BaleService.SaveBale",
            )
            MockRetriever.return_value = mock_retriever

            mock_context_builder = MagicMock()
            mock_context_builder.build_context.return_value = ("Context", [])
            mock_context_builder.build_prompt.return_value = "Prompt"
            MockContextBuilder.return_value = mock_context_builder

            async def mock_stream(*args, **kwargs):
                yield SSEEvent(event="complete", data={"answer": "Done"})

            mock_generator = AsyncMock()
            mock_generator.generate_stream_sse = mock_stream
            MockGenerator.return_value = mock_generator

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            request = CodeQueryRequest(query="Test")

            events = []
            async for event in pipeline.process_query_stream(request):
                events.append(event)

            event_types = [e.event for e in events]
            assert "sources" in event_types

    @pytest.mark.asyncio
    async def test_stream_handles_error_gracefully(self, pipeline_config):
        """Test that stream handles errors with error event."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import (
            CodeQueryRequest,
            SSEEvent,
        )

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            # Make retriever fail
            mock_retriever = AsyncMock()
            mock_retriever.retrieve_comprehensive.side_effect = Exception("Retrieval failed")
            MockRetriever.return_value = mock_retriever

            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            request = CodeQueryRequest(query="Test")

            events = []
            async for event in pipeline.process_query_stream(request):
                events.append(event)

            # Should have error event
            event_types = [e.event for e in events]
            assert "error" in event_types


# =============================================================================
# HTTP Stream Endpoint Tests
# =============================================================================

class TestStreamEndpoint:
    """Test streaming endpoint integration."""

    @pytest.mark.asyncio
    async def test_stream_endpoint_content_type(self, pipeline_config):
        """Test that stream endpoint returns correct content type."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import SSEEvent

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()

            async def mock_stream(request):
                yield SSEEvent(event="status", data={"status": "generating"})
                yield SSEEvent(event="complete", data={"answer": "Done"})

            mock_pipeline.process_query_stream = mock_stream
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/api/code/query/stream",
                    json={"query": "Test query"},
                ) as response:
                    content_type = response.headers.get("content-type", "")
                    assert "text/event-stream" in content_type

    @pytest.mark.asyncio
    async def test_stream_endpoint_returns_events(self, pipeline_config):
        """Test that stream endpoint returns SSE formatted events."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from code_assistance_pipeline.routes import register_code_routes
        from code_assistance_pipeline.models.query_models import SSEEvent

        with patch('code_assistance_pipeline.routes.get_code_assistance_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()

            async def mock_stream(request):
                yield SSEEvent(event="status", data={"status": "generating"})
                yield SSEEvent(event="streaming", data={"token": "Test"})
                yield SSEEvent(event="complete", data={"answer": "Test answer"})

            mock_pipeline.process_query_stream = mock_stream
            mock_get_pipeline.return_value = mock_pipeline

            app = FastAPI()
            register_code_routes(app)

            with TestClient(app) as client:
                response = client.post(
                    "/api/code/query/stream",
                    json={"query": "Test query"},
                )

                content = response.text

                # Should contain SSE formatted events
                assert "event:" in content
                assert "data:" in content


# =============================================================================
# SSE Consumer Fixture Tests
# =============================================================================

class TestSSEConsumerFixture:
    """Test SSE consumer fixture functionality."""

    def test_consume_sync_parses_events(self, sse_consumer):
        """Test that consumer parses SSE lines."""
        lines = [
            'event: status',
            'data: {"status": "generating"}',
            '',
            'event: streaming',
            'data: {"token": "Hello"}',
            '',
        ]

        events = sse_consumer.consume_sync(lines)

        assert len(events) == 2

    def test_consumer_handles_json_data(self, sse_consumer):
        """Test that consumer parses JSON data."""
        lines = [
            'data: {"key": "value", "count": 5}',
            '',
        ]

        events = sse_consumer.consume_sync(lines)

        assert len(events) == 1
        assert events[0]["key"] == "value"
        assert events[0]["count"] == 5

    def test_consumer_handles_non_json_data(self, sse_consumer):
        """Test that consumer handles non-JSON data."""
        lines = [
            'data: Plain text message',
            '',
        ]

        events = sse_consumer.consume_sync(lines)

        assert len(events) == 1
        assert "raw_data" in events[0]

    def test_consumer_counts_events(self, sse_consumer):
        """Test event counting."""
        lines = [
            'data: {"a": 1}',
            '',
            'data: {"b": 2}',
            '',
            'data: {"c": 3}',
            '',
        ]

        sse_consumer.consume_sync(lines)

        assert sse_consumer.get_event_count() == 3
