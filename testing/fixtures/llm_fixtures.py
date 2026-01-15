"""
Local LLM Test Fixtures
=======================

Fixtures for testing with local llama.cpp endpoints ONLY.
NO EXTERNAL APIs (OpenAI, Anthropic, etc.) ARE PERMITTED.

Available endpoints:
- SQL Model: localhost:8080
- General Model: localhost:8081
- Code Model: localhost:8082
"""

import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from local LLM."""
    success: bool
    text: str = ""
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: int = 0


class LocalLLMClient:
    """
    Synchronous client for local llama.cpp servers.

    IMPORTANT: Only local endpoints are permitted.
    No external API calls allowed.
    """

    def __init__(
        self,
        sql_endpoint: str = "http://localhost:8080",
        general_endpoint: str = "http://localhost:8081",
        code_endpoint: str = "http://localhost:8082",
        timeout: int = 120,
    ):
        """
        Initialize local LLM client.

        Args:
            sql_endpoint: SQL model endpoint (llama.cpp on port 8080)
            general_endpoint: General model endpoint (llama.cpp on port 8081)
            code_endpoint: Code model endpoint (llama.cpp on port 8082)
            timeout: Request timeout in seconds
        """
        # Validate all endpoints are local
        for name, endpoint in [
            ("sql", sql_endpoint),
            ("general", general_endpoint),
            ("code", code_endpoint),
        ]:
            if not endpoint.startswith(("http://localhost", "http://127.0.0.1")):
                raise ValueError(
                    f"Only local endpoints allowed for {name}. Got: {endpoint}. "
                    "External APIs (OpenAI, Anthropic, etc.) are not permitted."
                )

        self.sql_endpoint = sql_endpoint.rstrip("/")
        self.general_endpoint = general_endpoint.rstrip("/")
        self.code_endpoint = code_endpoint.rstrip("/")
        self.timeout = timeout

    def _get_endpoint(self, model_type: str = "general") -> str:
        """Get endpoint URL for model type."""
        endpoints = {
            "sql": self.sql_endpoint,
            "general": self.general_endpoint,
            "code": self.code_endpoint,
        }
        return endpoints.get(model_type, self.general_endpoint)

    def generate(
        self,
        prompt: str,
        model_type: str = "general",
        endpoint: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate completion from local LLM.

        Args:
            prompt: Input prompt
            model_type: One of "sql", "general", "code"
            endpoint: Alias for model_type (for compatibility)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            LLMResponse with generated text
        """
        # Support 'endpoint' as alias for 'model_type'
        actual_model_type = endpoint if endpoint else model_type
        llm_endpoint = self._get_endpoint(actual_model_type)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{llm_endpoint}/v1/completions",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stop": stop or [],
                    }
                )
                response.raise_for_status()
                data = response.json()

                usage = data.get("usage", {})
                return LLMResponse(
                    success=True,
                    text=data.get("choices", [{}])[0].get("text", "").strip(),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )

        except httpx.TimeoutException:
            return LLMResponse(success=False, error="Request timeout")
        except httpx.HTTPStatusError as e:
            return LLMResponse(success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return LLMResponse(success=False, error=str(e))

    def health_check(self, model_type: str = "general") -> Dict[str, Any]:
        """
        Check if LLM endpoint is healthy.

        Args:
            model_type: One of "sql", "general", "code"

        Returns:
            Dict with health status
        """
        endpoint = self._get_endpoint(model_type)

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{endpoint}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("id") for m in data.get("data", [])]
                    return {
                        "healthy": True,
                        "endpoint": endpoint,
                        "models": models,
                    }
                return {
                    "healthy": False,
                    "endpoint": endpoint,
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {
                "healthy": False,
                "endpoint": endpoint,
                "error": str(e),
            }


class AsyncLocalLLMClient:
    """
    Async client for local llama.cpp servers.

    IMPORTANT: Only local endpoints are permitted.
    No external API calls allowed.
    """

    def __init__(
        self,
        sql_endpoint: str = "http://localhost:8080",
        general_endpoint: str = "http://localhost:8081",
        code_endpoint: str = "http://localhost:8082",
        timeout: int = 120,
    ):
        """Initialize async local LLM client."""
        # Validate all endpoints are local
        for name, endpoint in [
            ("sql", sql_endpoint),
            ("general", general_endpoint),
            ("code", code_endpoint),
        ]:
            if not endpoint.startswith(("http://localhost", "http://127.0.0.1")):
                raise ValueError(
                    f"Only local endpoints allowed for {name}. Got: {endpoint}. "
                    "External APIs (OpenAI, Anthropic, etc.) are not permitted."
                )

        self.sql_endpoint = sql_endpoint.rstrip("/")
        self.general_endpoint = general_endpoint.rstrip("/")
        self.code_endpoint = code_endpoint.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_endpoint(self, model_type: str = "general") -> str:
        """Get endpoint URL for model type."""
        endpoints = {
            "sql": self.sql_endpoint,
            "general": self.general_endpoint,
            "code": self.code_endpoint,
        }
        return endpoints.get(model_type, self.general_endpoint)

    async def generate(
        self,
        prompt: str,
        model_type: str = "general",
        endpoint: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate completion from local LLM (async).

        Args:
            prompt: Input prompt
            model_type: One of "sql", "general", "code"
            endpoint: Alias for model_type (for compatibility)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            LLMResponse with generated text
        """
        # Support 'endpoint' as alias for 'model_type'
        actual_model_type = endpoint if endpoint else model_type
        llm_endpoint = self._get_endpoint(actual_model_type)
        client = await self._get_client()

        try:
            response = await client.post(
                f"{llm_endpoint}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop or [],
                }
            )
            response.raise_for_status()
            data = response.json()

            usage = data.get("usage", {})
            return LLMResponse(
                success=True,
                text=data.get("choices", [{}])[0].get("text", "").strip(),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        except httpx.TimeoutException:
            return LLMResponse(success=False, error="Request timeout")
        except httpx.HTTPStatusError as e:
            return LLMResponse(success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return LLMResponse(success=False, error=str(e))

    async def health_check(self, model_type: str = "general") -> Dict[str, Any]:
        """Check if LLM endpoint is healthy (async)."""
        endpoint = self._get_endpoint(model_type)
        client = await self._get_client()

        try:
            response = await client.get(f"{endpoint}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                return {
                    "healthy": True,
                    "endpoint": endpoint,
                    "models": models,
                }
            return {
                "healthy": False,
                "endpoint": endpoint,
                "error": f"HTTP {response.status_code}",
            }
        except Exception as e:
            return {
                "healthy": False,
                "endpoint": endpoint,
                "error": str(e),
            }

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def check_llm_health(
    sql_endpoint: str = "http://localhost:8080",
    general_endpoint: str = "http://localhost:8081",
    code_endpoint: str = "http://localhost:8082",
) -> Dict[str, Dict[str, Any]]:
    """
    Check health of all local LLM endpoints.

    Returns:
        Dict with health status for each endpoint
    """
    client = LocalLLMClient(
        sql_endpoint=sql_endpoint,
        general_endpoint=general_endpoint,
        code_endpoint=code_endpoint,
    )

    return {
        "sql": client.health_check("sql"),
        "general": client.health_check("general"),
        "code": client.health_check("code"),
    }
