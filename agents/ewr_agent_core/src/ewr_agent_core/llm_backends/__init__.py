"""
EWR Agent LLM Backends
======================

Pluggable LLM backends supporting:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude)

Usage:
    from ewr_agent_core.llm_backends import get_backend

    # Get backend by name
    backend = get_backend("openai", model="gpt-4", api_key="sk-...")

    # Generate response
    response = await backend.generate("Write a Python function")
"""

from .base import LLMBackend, LLMResponse, LLMStreamChunk
from .openai import OpenAIBackend
from .anthropic import AnthropicBackend

# Registry of available backends
_BACKENDS = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
}


def get_backend(
    backend_name: str,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    **kwargs
) -> LLMBackend:
    """
    Get an LLM backend instance by name.

    Args:
        backend_name: Name of backend (openai, anthropic)
        model: Model name to use
        base_url: Optional custom base URL
        api_key: API key (required for openai, anthropic)
        **kwargs: Additional backend-specific options

    Returns:
        LLMBackend instance

    Example:
        backend = get_backend("openai", model="gpt-4", api_key="sk-...")
        backend = get_backend("anthropic", model="claude-3-opus", api_key="sk-...")
    """
    backend_name = backend_name.lower()
    if backend_name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend: {backend_name}. Available: {available}")

    backend_class = _BACKENDS[backend_name]
    return backend_class(
        model=model,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


def register_backend(name: str, backend_class: type) -> None:
    """
    Register a custom LLM backend.

    Args:
        name: Name to register the backend under
        backend_class: Class that extends LLMBackend

    Example:
        class MyBackend(LLMBackend):
            ...

        register_backend("mybackend", MyBackend)
    """
    if not issubclass(backend_class, LLMBackend):
        raise TypeError("Backend must be a subclass of LLMBackend")
    _BACKENDS[name.lower()] = backend_class


def list_backends() -> list:
    """List available backend names."""
    return list(_BACKENDS.keys())


__all__ = [
    "LLMBackend",
    "LLMResponse",
    "LLMStreamChunk",
    "OpenAIBackend",
    "AnthropicBackend",
    "get_backend",
    "register_backend",
    "list_backends",
]
