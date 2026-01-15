"""
Response Generator Service
==========================

Generates LLM responses for code assistance queries with streaming support.
Now supports TracedLLMClient for automatic monitoring and tracing.

Design Rationale:
-----------------
This service wraps the LLMService to provide code-specific generation:

1. Model Selection: Uses the code model (qwen2.5-coder) by default for
   better code understanding and explanation quality.

2. Streaming Support: Provides async generators for SSE streaming,
   enabling real-time response delivery to the frontend.

3. Token Tracking: Captures token usage for monitoring and billing.

4. Error Handling: Graceful degradation with informative error messages
   when LLM calls fail.

5. Automatic Tracing: Uses TracedLLMClient for LLM monitoring (when available)

Architecture:
- Uses TracedLLMClient from llm module (preferred)
- Falls back to LLMService for actual LLM operations
- Provides streaming via async generators
- Stateless - all configuration passed via method arguments
"""

import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any

from code_assistance_pipeline.models.query_models import (
    TokenUsage,
    SSEEvent,
)

logger = logging.getLogger(__name__)

# Try to import TracedLLMClient
try:
    from llm.integration import generate_text
    TRACED_LLM_AVAILABLE = True
except ImportError:
    TRACED_LLM_AVAILABLE = False
    logger.info("TracedLLMClient not available, using legacy LLM service")


class ResponseGenerator:
    """
    Generates LLM responses for code assistance queries.

    This service handles:
    - Non-streaming generation for simple requests
    - Streaming generation for real-time UI updates
    - Token usage tracking
    - Error handling and fallback messages

    Usage:
        generator = ResponseGenerator()
        await generator.initialize()

        # Non-streaming
        answer, usage, gen_time = await generator.generate(prompt)

        # Streaming
        async for chunk in generator.generate_stream(prompt):
            print(chunk, end="", flush=True)
    """

    # Default generation parameters for code assistance
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TIMEOUT = 120  # seconds

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize the response generator.

        Args:
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for response
            timeout: Request timeout in seconds
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._llm_service = None
        self._initialized = False
        self._use_traced = TRACED_LLM_AVAILABLE

    async def initialize(self) -> None:
        """
        Initialize the underlying LLM service.

        Uses lazy loading to avoid connection overhead until first use.
        """
        if self._initialized:
            return

        # Only load legacy service if traced is not available
        if not self._use_traced:
            from services.llm_service import get_llm_service
            self._llm_service = await get_llm_service()

        self._initialized = True
        logger.info(f"ResponseGenerator initialized (traced={self._use_traced})")

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized before use."""
        if not self._initialized:
            await self.initialize()

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        user_id: str = None,
        project: str = None
    ) -> tuple[str, TokenUsage, int]:
        """
        Generate a non-streaming response.

        Args:
            prompt: Complete prompt including context and query
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            user_id: Optional user ID for tracing
            project: Optional project name for tracing

        Returns:
            Tuple of (answer_text, token_usage, generation_time_ms)

        Raises:
            ValueError: If LLM generation fails after retries

        Design Note:
        Uses the code model (use_code_model=True) for better
        code understanding. Falls back to general model if
        code model is unavailable.
        """
        await self._ensure_initialized()

        start_time = time.time()

        # Try TracedLLMClient first
        if self._use_traced:
            return await self._generate_traced(
                prompt, temperature, max_tokens, user_id, project, start_time
            )

        # Fallback to legacy LLM service
        return await self._generate_legacy(prompt, temperature, max_tokens, start_time)

    async def _generate_traced(
        self,
        prompt: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        user_id: str,
        project: str,
        start_time: float
    ) -> tuple[str, TokenUsage, int]:
        """Generate using TracedLLMClient."""
        response = await generate_text(
            prompt=prompt,
            operation="code_assistance",
            pipeline="code_assistance",
            user_id=user_id,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            tags=["code_assistance", "code"],
            context_dict={"database": project} if project else None,
        )

        generation_time = int((time.time() - start_time) * 1000)

        if not response.success:
            logger.error(f"TracedLLM generation failed: {response.error}")
            error_msg = "I encountered an error generating the response. Please try again or rephrase your question."
            return error_msg, TokenUsage(), generation_time

        # Build token usage
        token_usage = TokenUsage(
            prompt_tokens=response.prompt_tokens or self._estimate_tokens(prompt),
            completion_tokens=response.response_tokens or self._estimate_tokens(response.text),
            total_tokens=response.total_tokens or 0
        )

        if token_usage.total_tokens == 0:
            token_usage.total_tokens = token_usage.prompt_tokens + token_usage.completion_tokens

        logger.info(
            f"Generated response in {generation_time}ms [TRACED], "
            f"{token_usage.total_tokens} tokens"
        )

        return response.text.strip(), token_usage, generation_time

    async def _generate_legacy(
        self,
        prompt: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        start_time: float
    ) -> tuple[str, TokenUsage, int]:
        """Generate using legacy LLM service."""
        result = await self._llm_service.generate(
            prompt=prompt,
            system="",  # System prompt is included in the prompt itself
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            use_code_model=True,  # Use code-specialized model
            use_cache=True
        )

        generation_time = int((time.time() - start_time) * 1000)

        if not result.success:
            logger.error(f"LLM generation failed: {result.error}")
            # Return error message instead of raising
            error_msg = "I encountered an error generating the response. Please try again or rephrase your question."
            return error_msg, TokenUsage(), generation_time

        # Build token usage
        token_usage = TokenUsage(
            prompt_tokens=result.token_usage.get("prompt_tokens", 0),
            completion_tokens=result.token_usage.get("response_tokens", 0),
            total_tokens=result.token_usage.get("total_tokens", 0)
        )

        # If we don't have token counts, estimate them
        if token_usage.prompt_tokens == 0:
            token_usage.prompt_tokens = self._estimate_tokens(prompt)
        if token_usage.completion_tokens == 0:
            token_usage.completion_tokens = self._estimate_tokens(result.response)
        if token_usage.total_tokens == 0:
            token_usage.total_tokens = token_usage.prompt_tokens + token_usage.completion_tokens

        logger.info(
            f"Generated response in {generation_time}ms, "
            f"{token_usage.total_tokens} tokens"
        )

        return result.response.strip(), token_usage, generation_time

    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Yields tokens as they are generated by the LLM, enabling
        real-time display in the UI.

        Args:
            prompt: Complete prompt including context and query
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            String tokens as they are generated

        Design Note:
        Streaming is preferred for code assistance because responses
        can be lengthy. Real-time streaming improves perceived latency
        and allows users to start reading while generation continues.
        """
        await self._ensure_initialized()

        async for chunk in self._llm_service.generate_stream(
            prompt=prompt,
            system="",
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            use_code_model=True
        ):
            if chunk.error:
                logger.error(f"Streaming error: {chunk.error}")
                yield f"Error: {chunk.error}"
                break

            if chunk.content:
                yield chunk.content

            if chunk.done:
                break

    async def generate_stream_sse(
        self,
        prompt: str,
        response_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Generate a streaming response with SSE event formatting.

        Yields SSEEvent objects suitable for Server-Sent Events
        streaming endpoints.

        Args:
            prompt: Complete prompt including context and query
            response_id: Unique ID for this response (for tracking)
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            SSEEvent objects for each stage of generation

        Design Note:
        This method provides structured SSE events that the frontend
        can parse to show progress indicators, streaming text, and
        final completion status.
        """
        await self._ensure_initialized()

        start_time = time.time()
        full_answer = ""
        total_tokens = 0

        # Emit generating status
        yield SSEEvent(
            event="status",
            data={"status": "generating", "message": "Generating response..."}
        )

        try:
            async for chunk in self._llm_service.generate_stream(
                prompt=prompt,
                system="",
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                use_code_model=True
            ):
                if chunk.error:
                    yield SSEEvent(
                        event="error",
                        data={"status": "error", "error": chunk.error}
                    )
                    return

                if chunk.content:
                    full_answer += chunk.content
                    total_tokens += 1
                    yield SSEEvent(
                        event="streaming",
                        data={"status": "streaming", "token": chunk.content}
                    )

                if chunk.done:
                    # Get final token usage
                    token_usage = chunk.token_usage or {}
                    total_time = int((time.time() - start_time) * 1000)

                    yield SSEEvent(
                        event="complete",
                        data={
                            "status": "complete",
                            "answer": full_answer,
                            "response_id": response_id,
                            "timing": {"total_ms": total_time},
                            "token_usage": {
                                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                                "completion_tokens": token_usage.get("response_tokens", total_tokens),
                                "total_tokens": token_usage.get("total_tokens", total_tokens)
                            }
                        }
                    )
                    return

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield SSEEvent(
                event="error",
                data={"status": "error", "error": str(e)}
            )

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Uses a simple heuristic of 1.3 tokens per word.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        words = text.split()
        return int(len(words) * 1.3)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check LLM service health.

        Returns:
            Dict with health status and endpoint information
        """
        await self._ensure_initialized()
        return await self._llm_service.health_check()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get LLM cache statistics.

        Returns:
            Dict with cache size and configuration
        """
        await self._ensure_initialized()
        return await self._llm_service.get_cache_stats()

    async def clear_cache(self) -> None:
        """Clear the LLM response cache."""
        await self._ensure_initialized()
        await self._llm_service.clear_cache()
        logger.info("LLM cache cleared")
