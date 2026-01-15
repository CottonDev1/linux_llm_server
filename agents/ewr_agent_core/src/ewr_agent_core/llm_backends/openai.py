"""
EWR Agent OpenAI Backend
========================

LLM backend for OpenAI API (GPT-3.5, GPT-4, etc.)
https://platform.openai.com/docs/api-reference
"""

import asyncio
import time
import os
from typing import Optional, Dict, Any, AsyncIterator, List
import aiohttp

from .base import LLMBackend, LLMResponse, LLMStreamChunk, LLMMessage


class OpenAIBackend(LLMBackend):
    """
    OpenAI LLM backend.

    Supports GPT-3.5, GPT-4, and other OpenAI models.

    Example:
        backend = OpenAIBackend(
            model="gpt-4o-mini",
            api_key="sk-..."
        )
        response = await backend.generate("Hello, world!")

        # With streaming
        async for chunk in backend.generate_stream("Tell me a story"):
            print(chunk.content, end="")
    """

    def __init__(self, **kwargs):
        # Get API key from env if not provided
        if "api_key" not in kwargs or kwargs.get("api_key") is None:
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI chat completions API."""
        start_time = time.time()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }

        # Add any extra options
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
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
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            return LLMResponse(
                content=message.get("content", ""),
                model=data.get("model", self.model),
                finish_reason=choice.get("finish_reason", "stop"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                duration_ms=duration_ms,
                metadata={
                    "id": data.get("id"),
                    "created": data.get("created"),
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
        """Stream a response using OpenAI chat completions API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }

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
                        if not line or line == "data: [DONE]":
                            continue
                        if not line.startswith("data: "):
                            continue

                        try:
                            import json
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            content = delta.get("content", "")
                            is_final = finish_reason is not None

                            yield LLMStreamChunk(
                                content=content,
                                is_final=is_final,
                                finish_reason=finish_reason,
                                metadata={}
                            )

                            if is_final:
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
        """Generate using OpenAI's native chat endpoint."""
        start_time = time.time()

        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }

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
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            return LLMResponse(
                content=message.get("content", ""),
                model=data.get("model", self.model),
                finish_reason=choice.get("finish_reason", "stop"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                duration_ms=duration_ms,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
                metadata={"error": str(e)}
            )

    async def embed(
        self,
        text: str,
        model: str = None,
        **kwargs
    ) -> List[float]:
        """Generate embedding using OpenAI embeddings API."""
        url = f"{self.base_url}/embeddings"
        embed_model = model or "text-embedding-3-small"

        payload = {
            "model": embed_model,
            "input": text,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Embedding failed: {error_text}")
                data = await resp.json()
                return data.get("data", [{}])[0].get("embedding", [])

    async def health_check(self) -> Dict[str, Any]:
        """Check if OpenAI API is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 401:
                        return {
                            "healthy": False,
                            "model": self.model,
                            "error": "Invalid API key"
                        }
                    elif resp.status != 200:
                        return {
                            "healthy": False,
                            "model": self.model,
                            "error": f"API returned status {resp.status}"
                        }

                    data = await resp.json()
                    models = [m.get("id") for m in data.get("data", [])]

                    return {
                        "healthy": True,
                        "model": self.model,
                        "available_models": models[:20],  # Limit for display
                        "model_available": self.model in models
                    }

        except aiohttp.ClientError as e:
            return {
                "healthy": False,
                "model": self.model,
                "error": f"Connection failed: {e}"
            }

    async def list_models(self) -> List[str]:
        """List available OpenAI models."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to list models: {await resp.text()}")
                data = await resp.json()
                return [m.get("id") for m in data.get("data", [])]
