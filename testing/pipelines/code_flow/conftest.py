"""
Code Flow Pipeline Test Configuration
======================================

Pytest configuration and fixtures for Code Flow Pipeline E2E tests.
Provides mock services, test data, and helper fixtures.
"""

import pytest
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import shared fixtures
from fixtures.shared_fixtures import (
    mock_embedding_service,
    mock_vector_search,
    sse_consumer,
    token_assertions,
    test_document_generator,
    response_validator,
)
from utils import generate_test_id


# =============================================================================
# Mock MongoDB Service
# =============================================================================

class MockMongoDBService:
    """
    Mock MongoDB service for testing code flow pipeline.

    Provides controllable search results without requiring actual database.
    """

    def __init__(self):
        self.is_initialized = True
        self._search_results: Dict[str, List[Dict[str, Any]]] = {}
        self._default_results: List[Dict[str, Any]] = []

    def set_search_results(
        self,
        category: str,
        doc_type: Optional[str],
        results: List[Dict[str, Any]]
    ):
        """Configure search results for a specific category/type."""
        key = f"{category}:{doc_type or 'all'}"
        self._search_results[key] = results

    def set_default_results(self, results: List[Dict[str, Any]]):
        """Set default results when no specific results are configured."""
        self._default_results = results

    async def initialize(self):
        """Mock initialization."""
        self.is_initialized = True

    async def search_vectors(
        self,
        query: str,
        project: Optional[str] = None,
        category: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Return configured search results."""
        key = f"{category}:{doc_type or 'all'}"
        results = self._search_results.get(key, self._default_results)

        # Filter by project if specified
        if project:
            results = [r for r in results if r.get("metadata", {}).get("project") == project or not r.get("metadata", {}).get("project")]

        return results[:limit]

    def clear(self):
        """Clear all configured results."""
        self._search_results.clear()
        self._default_results.clear()


@pytest.fixture
def mock_mongodb_service():
    """Provides a mock MongoDB service."""
    return MockMongoDBService()


# =============================================================================
# Mock LLM Service
# =============================================================================

class MockLLMService:
    """
    Mock LLM service for testing code flow pipeline.

    Provides controllable LLM responses without actual model calls.
    """

    def __init__(self):
        self._response_text = "This is a mock LLM response for code flow analysis."
        self._should_fail = False
        self._error_message = "Mock LLM error"

    def set_response(self, text: str):
        """Configure the response text."""
        self._response_text = text

    def set_failure(self, should_fail: bool = True, error_message: str = "Mock LLM error"):
        """Configure the service to fail."""
        self._should_fail = should_fail
        self._error_message = error_message

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.5,
        use_cache: bool = True,
        **kwargs
    ):
        """Return a mock LLM response."""
        if self._should_fail:
            return MagicMock(success=False, error=self._error_message)

        return MagicMock(success=True, response=self._response_text)

    async def close(self):
        """Mock close."""
        pass


@pytest.fixture
def mock_llm_service():
    """Provides a mock LLM service."""
    return MockLLMService()


# =============================================================================
# Test Data Factories
# =============================================================================

@dataclass
class CodeFlowTestData:
    """Factory for generating code flow test data."""
    test_run_id: str = field(default_factory=lambda: f"test_{generate_test_id()}")

    def create_method_result(
        self,
        method_name: str,
        class_name: str = "TestClass",
        project: str = "TestProject",
        similarity: float = 0.85,
        calls_method: Optional[List[str]] = None,
        called_by: Optional[List[str]] = None,
        database_tables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a mock method search result."""
        return {
            "id": f"method_{generate_test_id()}",
            "_id": f"method_{generate_test_id()}",
            "content": f"Implementation of {method_name} in {class_name}",
            "similarity": similarity,
            "score": similarity,
            "metadata": {
                "methodName": method_name,
                "className": class_name,
                "project": project,
                "purposeSummary": f"Handles {method_name} logic",
                "filePath": f"/src/{class_name}.cs",
                "startLine": 100,
                "endLine": 150,
                "returnType": "void",
                "isPublic": True,
                "isAsync": False,
                "callsMethod": json.dumps(calls_method or []),
                "calledByMethod": json.dumps(called_by or []),
                "databaseTables": json.dumps(database_tables or []),
                "category": "code",
                "type": "method",
            }
        }

    def create_class_result(
        self,
        class_name: str,
        project: str = "TestProject",
        similarity: float = 0.80,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a mock class search result."""
        return {
            "id": f"class_{generate_test_id()}",
            "_id": f"class_{generate_test_id()}",
            "content": f"Class {class_name} implementation",
            "similarity": similarity,
            "score": similarity,
            "metadata": {
                "className": class_name,
                "project": project,
                "namespace": "TestNamespace",
                "methods": json.dumps(methods or ["Method1", "Method2"]),
                "category": "code",
                "type": "class",
            }
        }

    def create_ui_event_result(
        self,
        control_name: str,
        handler_method: str,
        project: str = "TestProject",
        similarity: float = 0.90,
    ) -> Dict[str, Any]:
        """Create a mock UI event search result."""
        return {
            "id": f"ui_event_{generate_test_id()}",
            "_id": f"ui_event_{generate_test_id()}",
            "content": f"UI event handler for {control_name}",
            "similarity": similarity,
            "score": similarity,
            "metadata": {
                "controlName": control_name,
                "handlerMethod": handler_method,
                "eventType": "click",
                "project": project,
                "formName": "TestForm",
                "category": "ui-mapping",
                "type": "ui_event",
            }
        }

    def create_business_process_result(
        self,
        title: str,
        content: str,
        project: str = "TestProject",
        similarity: float = 0.88,
    ) -> Dict[str, Any]:
        """Create a mock business process search result."""
        return {
            "id": f"bp_{generate_test_id()}",
            "_id": f"bp_{generate_test_id()}",
            "content": content,
            "similarity": similarity,
            "score": similarity,
            "metadata": {
                "title": title,
                "project": project,
                "category": "business-process",
                "type": "documentation",
            }
        }

    def create_call_relationship_result(
        self,
        caller_method: str,
        callee_method: str,
        project: str = "TestProject",
        similarity: float = 0.75,
    ) -> Dict[str, Any]:
        """Create a mock call relationship search result."""
        return {
            "id": f"rel_{generate_test_id()}",
            "_id": f"rel_{generate_test_id()}",
            "content": f"{caller_method} calls {callee_method}",
            "similarity": similarity,
            "score": similarity,
            "metadata": {
                "callerMethod": caller_method,
                "calleeMethod": callee_method,
                "callType": "direct",
                "project": project,
                "category": "relationship",
                "type": "method-call",
            }
        }


@pytest.fixture
def code_flow_test_data():
    """Provides test data factory for code flow tests."""
    return CodeFlowTestData()


# =============================================================================
# Sample Test Scenarios
# =============================================================================

@pytest.fixture
def sample_bale_processing_scenario(code_flow_test_data: CodeFlowTestData) -> Dict[str, List[Dict[str, Any]]]:
    """
    Provides a complete test scenario for bale processing flow.

    This simulates a typical business process query with:
    - Business process documentation
    - UI event handlers
    - Method implementations
    - Database accessors
    """
    return {
        "business_processes": [
            code_flow_test_data.create_business_process_result(
                title="Bale Commitment Process",
                content="""The bale commitment process allows users to commit bales to purchase contracts.
                1. User selects bales from the bale grid
                2. User clicks 'Commit to Purchase' button
                3. System validates bale eligibility
                4. System updates bale status and creates commitment records
                5. System generates commitment report""",
                project="Gin",
            ),
        ],
        "methods": [
            code_flow_test_data.create_method_result(
                method_name="CommitBalesToPurchase",
                class_name="BaleCommitmentService",
                project="Gin",
                similarity=0.92,
                calls_method=["ValidateBaleEligibility", "CreateCommitmentRecord", "UpdateBaleStatus"],
                database_tables=["Bales", "PurchaseCommitments"],
            ),
            code_flow_test_data.create_method_result(
                method_name="ValidateBaleEligibility",
                class_name="BaleValidationService",
                project="Gin",
                similarity=0.88,
                called_by=["CommitBalesToPurchase"],
            ),
            code_flow_test_data.create_method_result(
                method_name="UpdateBaleStatus",
                class_name="BaleService",
                project="Gin",
                similarity=0.85,
                called_by=["CommitBalesToPurchase"],
                database_tables=["Bales"],
            ),
        ],
        "ui_events": [
            code_flow_test_data.create_ui_event_result(
                control_name="btnCommitToPurchase",
                handler_method="btnCommitToPurchase_Click",
                project="Gin",
            ),
        ],
        "call_relationships": [
            code_flow_test_data.create_call_relationship_result(
                caller_method="btnCommitToPurchase_Click",
                callee_method="CommitBalesToPurchase",
                project="Gin",
            ),
            code_flow_test_data.create_call_relationship_result(
                caller_method="CommitBalesToPurchase",
                callee_method="ValidateBaleEligibility",
                project="Gin",
            ),
            code_flow_test_data.create_call_relationship_result(
                caller_method="CommitBalesToPurchase",
                callee_method="UpdateBaleStatus",
                project="Gin",
            ),
        ],
    }


@pytest.fixture
def sample_order_processing_scenario(code_flow_test_data: CodeFlowTestData) -> Dict[str, List[Dict[str, Any]]]:
    """
    Provides a test scenario for order processing flow.
    """
    return {
        "business_processes": [
            code_flow_test_data.create_business_process_result(
                title="Order Processing",
                content="Order processing workflow handles customer orders from entry to fulfillment.",
                project="Warehouse",
            ),
        ],
        "methods": [
            code_flow_test_data.create_method_result(
                method_name="ProcessOrder",
                class_name="OrderService",
                project="Warehouse",
                calls_method=["ValidateOrder", "CalculateTotal", "SaveOrder"],
                database_tables=["Orders", "OrderItems"],
            ),
            code_flow_test_data.create_method_result(
                method_name="ValidateOrder",
                class_name="OrderValidationService",
                project="Warehouse",
                called_by=["ProcessOrder"],
            ),
        ],
        "ui_events": [
            code_flow_test_data.create_ui_event_result(
                control_name="btnProcessOrder",
                handler_method="btnProcessOrder_Click",
                project="Warehouse",
            ),
        ],
        "call_relationships": [
            code_flow_test_data.create_call_relationship_result(
                caller_method="btnProcessOrder_Click",
                callee_method="ProcessOrder",
                project="Warehouse",
            ),
        ],
    }


# =============================================================================
# Pipeline Component Fixtures
# =============================================================================

@pytest.fixture
def mock_query_classifier():
    """Provides a mock query classifier."""
    from code_flow_pipeline.services.query_classifier import QueryClassifier
    return QueryClassifier()


@pytest.fixture
async def mock_code_flow_pipeline(mock_mongodb_service, mock_llm_service):
    """
    Provides a CodeFlowPipeline with mocked dependencies.

    Use this fixture for testing pipeline orchestration without
    actual MongoDB or LLM calls.
    """
    from code_flow_pipeline.pipeline import CodeFlowPipeline

    pipeline = CodeFlowPipeline()

    # Inject mock services
    pipeline._mongodb_service = mock_mongodb_service
    pipeline._llm_service = mock_llm_service
    pipeline._use_traced = False  # Use legacy LLM service

    # Initialize the classifier
    from code_flow_pipeline.services.query_classifier import get_query_classifier
    pipeline._classifier = get_query_classifier()

    return pipeline


# =============================================================================
# SSE Stream Helpers
# =============================================================================

@dataclass
class SSECollector:
    """Collects SSE events from async generator."""
    events: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    async def collect(self, stream: AsyncIterator) -> None:
        """Collect all events from an SSE stream."""
        async for event in stream:
            if hasattr(event, 'event') and event.event == 'error':
                self.errors.append(event)
            else:
                self.events.append(event)

    def get_by_event_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type."""
        return [e for e in self.events if getattr(e, 'event', '') == event_type]

    def has_event_type(self, event_type: str) -> bool:
        """Check if an event type exists."""
        return any(getattr(e, 'event', '') == event_type for e in self.events)

    @property
    def event_types(self) -> List[str]:
        """Get list of all event types in order."""
        return [getattr(e, 'event', 'unknown') for e in self.events]


@pytest.fixture
def sse_collector():
    """Provides an SSE event collector."""
    return SSECollector()


# =============================================================================
# Async Helpers
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
