"""
EWR Agent LLM Backend Base
==========================

Abstract base class and models for LLM backends.
All LLM backends must inherit from LLMBackend.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, List
from pydantic import BaseModel, Field
from datetime import datetime


class LLMResponse(BaseModel):
    """Response from LLM generation."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return bool(self.content) and self.finish_reason != "error"


class LLMStreamChunk(BaseModel):
    """A chunk from streaming LLM response."""
    content: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """A message in a conversation."""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    All LLM backends must implement:
    - generate(): Generate a response from a prompt
    - generate_stream(): Stream a response from a prompt
    - health_check(): Check if the backend is available

    Backends may optionally implement:
    - generate_chat(): Generate from a list of messages
    - embed(): Generate embeddings
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize the LLM backend.

        Args:
            model: Model name to use
            base_url: Base URL for API
            api_key: API key for authentication
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional backend-specific options
        """
        self.model = model or self.default_model
        self.base_url = base_url or self.default_base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_options = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this backend."""
        pass

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        """Default base URL for this backend."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from a prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation options

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream a response from a prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation options

        Yields:
            LLMStreamChunk with partial content
        """
        pass

    async def generate_chat(
        self,
        messages: List[LLMMessage],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from a list of messages.

        Default implementation converts to prompt format.
        Backends should override for native chat support.

        Args:
            messages: List of conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation options

        Returns:
            LLMResponse with generated content
        """
        # Default: convert messages to prompt
        system = None
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts)
        if prompt_parts:
            prompt += "\nAssistant:"

        return await self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    async def embed(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        Generate embedding vector for text.

        Not all backends support embeddings.
        Override in subclass if supported.

        Args:
            text: Text to embed
            **kwargs: Additional options

        Returns:
            List of floats representing the embedding

        Raises:
            NotImplementedError: If backend doesn't support embeddings
        """
        raise NotImplementedError(f"{self.name} backend does not support embeddings")

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the backend is available and working.

        Returns:
            Dict with:
            - healthy: bool
            - model: str (current model)
            - available_models: list (if applicable)
            - error: str (if not healthy)
        """
        pass

    async def list_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model names

        Raises:
            NotImplementedError: If backend doesn't support listing
        """
        raise NotImplementedError(f"{self.name} backend does not support model listing")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, base_url={self.base_url!r})"
