"""
API Test Client Wrapper
=======================

Provides a unified interface for testing all API endpoints with:
- Authentication handling
- SSE streaming support
- Token usage capture
- Response validation
- Error handling helpers
"""

import httpx
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field


@dataclass
class APIResponse:
    """Standardized API response wrapper."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    elapsed_ms: float
    token_usage: Optional[Dict[str, int]] = None

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400


@dataclass
class SSEEvent:
    """Parsed SSE event."""
    event_type: str
    data: Dict[str, Any]
    id: Optional[str] = None
    retry: Optional[int] = None


class APITestClient:
    """Test client for API endpoint testing."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        auth_token: Optional[str] = None
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.auth_token = auth_token
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def get(self, path: str, params: Optional[Dict] = None) -> APIResponse:
        """Make GET request."""
        start = time.time()
        response = await self._client.get(path, params=params)
        elapsed = (time.time() - start) * 1000

        return self._parse_response(response, elapsed)

    async def post(self, path: str, json_data: Optional[Dict] = None) -> APIResponse:
        """Make POST request."""
        start = time.time()
        response = await self._client.post(path, json=json_data)
        elapsed = (time.time() - start) * 1000

        return self._parse_response(response, elapsed)

    async def put(self, path: str, json_data: Optional[Dict] = None) -> APIResponse:
        """Make PUT request."""
        start = time.time()
        response = await self._client.put(path, json=json_data)
        elapsed = (time.time() - start) * 1000

        return self._parse_response(response, elapsed)

    async def delete(self, path: str, params: Optional[Dict] = None) -> APIResponse:
        """Make DELETE request."""
        start = time.time()
        response = await self._client.delete(path, params=params)
        elapsed = (time.time() - start) * 1000

        return self._parse_response(response, elapsed)

    async def patch(self, path: str, json_data: Optional[Dict] = None) -> APIResponse:
        """Make PATCH request."""
        start = time.time()
        response = await self._client.patch(path, json=json_data)
        elapsed = (time.time() - start) * 1000

        return self._parse_response(response, elapsed)

    async def post_stream(self, path: str, json_data: Optional[Dict] = None) -> AsyncIterator[SSEEvent]:
        """Make POST request expecting SSE stream response."""
        async with self._client.stream("POST", path, json=json_data) as response:
            event_type = None
            data_lines = []

            async for line in response.aiter_lines():
                line = line.strip()

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif line == "" and data_lines:
                    # End of event
                    data_str = "\n".join(data_lines)
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {"raw": data_str}

                    yield SSEEvent(
                        event_type=event_type or "message",
                        data=data
                    )
                    event_type = None
                    data_lines = []

    async def get_stream(self, path: str, params: Optional[Dict] = None) -> AsyncIterator[SSEEvent]:
        """Make GET request expecting SSE stream response."""
        async with self._client.stream("GET", path, params=params) as response:
            event_type = None
            data_lines = []

            async for line in response.aiter_lines():
                line = line.strip()

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif line == "" and data_lines:
                    # End of event
                    data_str = "\n".join(data_lines)
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {"raw": data_str}

                    yield SSEEvent(
                        event_type=event_type or "message",
                        data=data
                    )
                    event_type = None
                    data_lines = []

    async def collect_stream(self, path: str, json_data: Optional[Dict] = None) -> List[SSEEvent]:
        """Collect all SSE events from a stream."""
        events = []
        async for event in self.post_stream(path, json_data):
            events.append(event)
        return events

    def _parse_response(self, response: httpx.Response, elapsed_ms: float) -> APIResponse:
        """Parse httpx response into APIResponse."""
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = response.text

        # Extract token usage if present
        token_usage = None
        if isinstance(data, dict):
            token_usage = data.get("token_usage") or data.get("tokens")

        return APIResponse(
            status_code=response.status_code,
            data=data,
            headers=dict(response.headers),
            elapsed_ms=elapsed_ms,
            token_usage=token_usage
        )


class EndpointTester:
    """Helper for testing specific endpoint patterns."""

    def __init__(self, client: APITestClient):
        self.client = client

    async def test_health(self, path: str = "/health") -> bool:
        """Test health endpoint."""
        response = await self.client.get(path)
        return response.is_success

    async def test_crud(
        self,
        base_path: str,
        create_data: Dict,
        update_data: Dict,
        id_field: str = "id"
    ) -> Dict[str, APIResponse]:
        """Test CRUD operations on an endpoint."""
        results = {}

        # Create
        results["create"] = await self.client.post(base_path, create_data)

        if results["create"].is_success:
            item_id = results["create"].data.get(id_field)

            # Read
            results["read"] = await self.client.get(f"{base_path}/{item_id}")

            # Update
            results["update"] = await self.client.post(
                f"{base_path}/{item_id}",
                update_data
            )

            # Delete
            results["delete"] = await self.client.post(
                f"{base_path}/{item_id}/delete"
            )

        return results

    async def test_validation(
        self,
        path: str,
        valid_data: Dict,
        invalid_cases: List[Dict]
    ) -> Dict[str, APIResponse]:
        """Test input validation."""
        results = {"valid": await self.client.post(path, valid_data)}

        for i, invalid_data in enumerate(invalid_cases):
            results[f"invalid_{i}"] = await self.client.post(path, invalid_data)

        return results

    async def test_rate_limiting(
        self,
        path: str,
        request_data: Optional[Dict] = None,
        num_requests: int = 10,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """
        Test rate limiting on an endpoint.

        Args:
            path: API path to test
            request_data: Optional request body for POST
            num_requests: Number of rapid requests to make
            method: HTTP method (GET or POST)

        Returns:
            Dict with success_count, rate_limited_count, and responses
        """
        responses = []

        for _ in range(num_requests):
            if method.upper() == "POST":
                response = await self.client.post(path, request_data)
            else:
                response = await self.client.get(path)
            responses.append(response)

        success_count = sum(1 for r in responses if r.is_success)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)

        return {
            "success_count": success_count,
            "rate_limited_count": rate_limited_count,
            "responses": responses
        }

    async def test_authentication(
        self,
        path: str,
        valid_token: Optional[str] = None,
        invalid_token: str = "invalid_token_123"
    ) -> Dict[str, APIResponse]:
        """
        Test authentication on an endpoint.

        Args:
            path: API path to test
            valid_token: Valid auth token (uses client's token if None)
            invalid_token: Invalid token for testing rejection

        Returns:
            Dict with responses for no_auth, invalid_auth, and valid_auth
        """
        results = {}

        # Test without auth
        original_token = self.client.auth_token
        self.client.auth_token = None
        self.client._client.headers.pop("Authorization", None)
        results["no_auth"] = await self.client.get(path)

        # Test with invalid auth
        self.client._client.headers["Authorization"] = f"Bearer {invalid_token}"
        results["invalid_auth"] = await self.client.get(path)

        # Restore auth and test with valid token
        token_to_use = valid_token or original_token
        if token_to_use:
            self.client.auth_token = token_to_use
            self.client._client.headers["Authorization"] = f"Bearer {token_to_use}"
            results["valid_auth"] = await self.client.get(path)

        return results


# Pytest fixtures
import pytest


@pytest.fixture
async def api_client():
    """Fixture providing configured API test client."""
    async with APITestClient() as client:
        yield client


@pytest.fixture
async def node_api_client():
    """Fixture providing API test client for Node.js server (port 3000)."""
    async with APITestClient(base_url="http://localhost:3000") as client:
        yield client


@pytest.fixture
async def endpoint_tester(api_client):
    """Fixture providing endpoint testing helper."""
    return EndpointTester(api_client)


@pytest.fixture
async def node_endpoint_tester(node_api_client):
    """Fixture providing endpoint testing helper for Node.js server."""
    return EndpointTester(node_api_client)
