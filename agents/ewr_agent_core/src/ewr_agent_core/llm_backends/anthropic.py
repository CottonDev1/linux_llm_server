"""
EWR Agent Anthropic Backend
===========================

LLM backend for Anthropic API (Claude models)
https://docs.anthropic.com/en/api/messages
"""

import asyncio
import time
import os
from typing import Optional, Dict, Any, AsyncIterator, List
import aiohttp

from .base import LLMBackend, LLMResponse, LLMStreamChunk, LLMMessage


class AnthropicBackend(LLMBackend):
    """
    Anthropic LLM backend.

    Supports Claude 3.5, Claude 3, and other Anthropic models.

    Example:
        backend = AnthropicBackend(
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-..."
        )
        response = await backend.generate("Hello, world!")

        # With streaming
        async for chunk in backend.generate_stream("Tell me a story"):
            print(chunk.content, end="")
    """

    # Anthropic API version header
    API_VERSION = "2023-06-01"

    def __init__(self, **kwargs):
        # Get API key from env if not provided
        if "api_key" not in kwargs or kwargs.get("api_key") is None:
            kwargs["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-3-5-sonnet-20241022"

    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Anthropic messages API."""
        start_time = time.time()

        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
        }

        # Temperature is optional for Anthropic
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            payload["temperature"] = temp

        if system:
            payload["system"] = system

        # Add any extra options
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return LLMResponse(
                            content="",
                            model=self.model,
                            finish_reason="error",
                            metadata={"error": error_text, "status": resp.status}
                        )

                    data = await resp.json()

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract content from response
            content_blocks = data.get("content", [])
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")

            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                finish_reason=data.get("stop_reason", "end_turn"),
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                duration_ms=duration_ms,
                metadata={
                    "id": data.get("id"),
                    "type": data.get("type"),
                }
            )

        except asyncio.TimeoutError:
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
                metadata={"error": "Request timed out"}
            )
        except aiohttp.ClientError as e:
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
                metadata={"error": str(e)}
            )

    async def generate_stream(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream a response using Anthropic messages API."""
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            payload["temperature"] = temp

        if system:
            payload["system"] = system

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        yield LLMStreamChunk(
                            content="",
                            is_final=True,
                            finish_reason="error",
                            metadata={"error": error_text}
                        )
                        return

                    async for line in resp.content:
                        line = line.decode().strip()
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue

                        try:
                            import json
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            event_type = data.get("type")

                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield LLMStreamChunk(
                                        content=delta.get("text", ""),
                                        is_final=False,
                                        metadata={}
                                    )

                            elif event_type == "message_stop":
                                yield LLMStreamChunk(
                                    content="",
                                    is_final=True,
                                    finish_reason="end_turn",
                                    metadata={}
                                )
                                break

                            elif event_type == "message_delta":
                                stop_reason = data.get("delta", {}).get("stop_reason")
                                if stop_reason:
                                    yield LLMStreamChunk(
                                        content="",
                                        is_final=True,
                                        finish_reason=stop_reason,
                                        metadata={}
                                    )
                                    break

                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            yield LLMStreamChunk(
                content="",
                is_final=True,
                finish_reason="error",
                metadata={"error": "Request timed out"}
            )
        except aiohttp.ClientError as e:
            yield LLMStreamChunk(
                content="",
                is_final=True,
                finish_reason="error",
                metadata={"error": str(e)}
            )

    async def generate_chat(
        self,
        messages: List[LLMMessage],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic's native messages endpoint."""
        start_time = time.time()

        # Convert messages, separating system message
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            payload["temperature"] = temp

        if system_content:
            payload["system"] = system_content

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return LLMResponse(
                            content="",
                            model=self.model,
                            finish_reason="error",
                            metadata={"error": error_text, "status": resp.status}
                        )

                    data = await resp.json()

            duration_ms = int((time.time() - start_time) * 1000)

            content_blocks = data.get("content", [])
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")

            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                finish_reason=data.get("stop_reason", "end_turn"),
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                duration_ms=duration_ms,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
                metadata={"error": str(e)}
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check if Anthropic API is accessible."""
        # Anthropic doesn't have a dedicated health/models endpoint
        # We'll try a minimal request to check connectivity
        try:
            async with aiohttp.ClientSession() as session:
                # Try to make a minimal request
                url = f"{self.base_url}/v1/messages"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                }

                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 401:
                        return {
                            "healthy": False,
                            "model": self.model,
                            "error": "Invalid API key"
                        }
                    elif resp.status == 400:
                        # Bad request might mean invalid model
                        error_data = await resp.json()
                        return {
                            "healthy": False,
                            "model": self.model,
                            "error": error_data.get("error", {}).get("message", "Bad request")
                        }
                    elif resp.status != 200:
                        return {
                            "healthy": False,
                            "model": self.model,
                            "error": f"API returned status {resp.status}"
                        }

                    return {
                        "healthy": True,
                        "model": self.model,
                        "available_models": self._get_known_models(),
                        "model_available": True
                    }

        except aiohttp.ClientError as e:
            return {
                "healthy": False,
                "model": self.model,
                "error": f"Connection failed: {e}"
            }

    def _get_known_models(self) -> List[str]:
        """Return list of known Anthropic models."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    async def list_models(self) -> List[str]:
        """List known Anthropic models (API doesn't provide this endpoint)."""
        return self._get_known_models()
