"""
LLM Generation Service
======================

Handles LLM prompt building and generation for RAG queries.
Uses the General model (port 8081) for document Q&A.
Now supports TracedLLMClient for automatic monitoring and tracing.

Design Rationale:
-----------------
This service encapsulates all LLM interaction logic:
1. Prompt construction with retrieved context
2. System prompt selection based on query type
3. Conversation history formatting
4. Response streaming for real-time UX

Key features:
- Separate prompts for code vs documentation queries
- Conversation history injection for follow-ups
- Token usage tracking
- Graceful timeout handling
- Automatic tracing with TracedLLMClient (when available)

Architecture Notes:
- Uses TracedLLMClient from llm module (preferred)
- Falls back to LLMService from services/llm_service.py
- Routes to General model (port 8081) for document Q&A
- Supports both blocking and streaming generation
"""

import logging
from typing import List, Optional, AsyncGenerator, Dict, Any
from dataclasses import dataclass

from query_pipeline.models.query_models import (
    VectorSearchResult,
    ChatMessage,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Try to import TracedLLMClient
try:
    from llm.integration import generate_text, get_llm_service as get_traced_llm_service
    TRACED_LLM_AVAILABLE = True
except ImportError:
    TRACED_LLM_AVAILABLE = False
    logger.info("TracedLLMClient not available, using legacy LLM service")


# =============================================================================
# Prompt Templates
# =============================================================================

CODE_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the company's codebase.
You have access to code, documentation, stored procedures, and business rules.
Answer based ONLY on the provided context. If you don't know, say so.
Be concise but thorough. Include relevant code snippets or procedure names when helpful."""

KNOWLEDGE_BASE_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about EWR's documentation, work instructions, and operational procedures.
Answer based ONLY on the provided documentation context. If you don't know, say so.
Be concise but thorough. Include relevant details from the documentation when helpful.
When the user uses pronouns like "it", "its", "they", "this", etc., refer to the conversation history to understand what they are referring to."""


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    timeout_seconds: int = 180


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    success: bool
    text: str = ""
    error: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    generation_time_seconds: float = 0.0


class LLMGenerationService:
    """
    Service for LLM-based text generation.

    This service handles:
    - Building prompts with retrieved context
    - Formatting conversation history
    - Calling the LLM backend (llama.cpp)
    - Streaming responses for real-time UI
    - Automatic tracing with TracedLLMClient (when available)

    Usage:
        service = LLMGenerationService()
        await service.initialize()

        result = await service.generate(
            query="How does the recap feature work?",
            context=search_results,
            history=[],
            is_knowledge_base=False
        )
    """

    _instance: Optional["LLMGenerationService"] = None

    def __init__(self):
        """Initialize the service (use get_instance for singleton)."""
        self._llm_service = None
        self._initialized = False
        self._default_config = GenerationConfig()
        self._use_traced = TRACED_LLM_AVAILABLE

    @classmethod
    async def get_instance(cls) -> "LLMGenerationService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """Initialize the LLM service connection."""
        if self._initialized:
            return

        # Only load legacy service if traced is not available
        if not self._use_traced:
            from services.llm_service import get_llm_service
            self._llm_service = await get_llm_service()

        self._initialized = True
        logger.info(f"LLMGenerationService initialized (traced={self._use_traced})")

    async def generate(
        self,
        query: str,
        context: List[VectorSearchResult],
        history: Optional[List[ChatMessage]] = None,
        is_knowledge_base: bool = False,
        config: Optional[GenerationConfig] = None,
        user_id: str = None,
        session_id: str = None,
        project: str = None
    ) -> GenerationResult:
        """
        Generate a response using retrieved context.

        Args:
            query: User's question
            context: Retrieved documents for context
            history: Conversation history for follow-ups
            is_knowledge_base: True for documentation, False for code
            config: Generation configuration
            user_id: Optional user ID for tracing
            session_id: Optional session ID for tracing
            project: Optional project name for tracing

        Returns:
            GenerationResult with generated text and metadata
        """
        if not self._initialized:
            await self.initialize()

        config = config or self._default_config

        try:
            # Build the prompt
            system_prompt = (
                KNOWLEDGE_BASE_SYSTEM_PROMPT if is_knowledge_base
                else CODE_SYSTEM_PROMPT
            )
            user_prompt = self._build_user_prompt(query, context, history, is_knowledge_base)

            # Combine for llama.cpp (OpenAI-compatible endpoint)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            logger.debug(f"Generating response (prompt length: {len(full_prompt)} chars)")

            # Try TracedLLMClient first
            if self._use_traced:
                return await self._generate_traced(
                    full_prompt, query, config, user_id, session_id, project, is_knowledge_base
                )

            # Fallback to legacy LLM service
            return await self._generate_legacy(full_prompt, config)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                error=str(e)
            )

    async def _generate_traced(
        self,
        full_prompt: str,
        query: str,
        config: GenerationConfig,
        user_id: str = None,
        session_id: str = None,
        project: str = None,
        is_knowledge_base: bool = False
    ) -> GenerationResult:
        """Generate using TracedLLMClient."""
        operation = "knowledge_base_qa" if is_knowledge_base else "code_qa"
        tags = ["query_pipeline", "rag"]
        if is_knowledge_base:
            tags.append("knowledge_base")
        else:
            tags.append("code")

        response = await generate_text(
            prompt=full_prompt,
            operation=operation,
            pipeline="query",
            user_id=user_id,
            session_id=session_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tags=tags,
            context_dict={
                "database": project,
                "user_question": query,
            } if project else {"user_question": query},
        )

        if not response.success:
            return GenerationResult(
                success=False,
                error=response.error or "LLM generation failed"
            )

        return GenerationResult(
            success=True,
            text=response.text,
            token_usage=TokenUsage(
                prompt_tokens=response.prompt_tokens,
                response_tokens=response.response_tokens,
                total_tokens=response.total_tokens
            ),
            generation_time_seconds=response.latency_ms / 1000 if response.latency_ms else 0
        )

    async def _generate_legacy(
        self,
        full_prompt: str,
        config: GenerationConfig
    ) -> GenerationResult:
        """Generate using legacy LLM service."""
        result = await self._llm_service.generate(
            prompt=full_prompt,
            system="",  # Already included in full_prompt
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            use_cache=True,
            use_sql_model=False,  # Use general model
            use_code_model=False
        )

        if not result.success:
            return GenerationResult(
                success=False,
                error=result.error or "LLM generation failed"
            )

        return GenerationResult(
            success=True,
            text=result.response,
            token_usage=TokenUsage(
                prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                response_tokens=result.token_usage.get("response_tokens", 0),
                total_tokens=result.token_usage.get("total_tokens", 0)
            ),
            generation_time_seconds=result.generation_time_ms / 1000
        )

    async def generate_stream(
        self,
        query: str,
        context: List[VectorSearchResult],
        history: Optional[List[ChatMessage]] = None,
        is_knowledge_base: bool = False,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response token by token.

        Yields dictionaries with:
        - {"type": "token", "token": "..."} for each token
        - {"type": "done", "token_usage": {...}} when complete
        - {"type": "error", "error": "..."} on failure

        Args:
            query: User's question
            context: Retrieved documents for context
            history: Conversation history
            is_knowledge_base: True for documentation
            config: Generation configuration

        Yields:
            Dictionaries for SSE formatting
        """
        if not self._initialized:
            await self.initialize()

        config = config or self._default_config

        try:
            # Build the prompt
            system_prompt = (
                KNOWLEDGE_BASE_SYSTEM_PROMPT if is_knowledge_base
                else CODE_SYSTEM_PROMPT
            )
            user_prompt = self._build_user_prompt(query, context, history, is_knowledge_base)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            logger.debug(f"Streaming response (prompt length: {len(full_prompt)} chars)")

            # Stream from LLM service
            total_tokens = 0
            async for chunk in self._llm_service.generate_stream(
                prompt=full_prompt,
                system="",
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                use_sql_model=False,
                use_code_model=False
            ):
                if chunk.error:
                    yield {"type": "error", "error": chunk.error}
                    return

                if chunk.done:
                    yield {
                        "type": "done",
                        "token_usage": chunk.token_usage or {
                            "prompt_tokens": 0,
                            "response_tokens": total_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                    return

                if chunk.content:
                    total_tokens += 1
                    yield {"type": "token", "token": chunk.content}

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    def _build_user_prompt(
        self,
        query: str,
        context: List[VectorSearchResult],
        history: Optional[List[ChatMessage]],
        is_knowledge_base: bool
    ) -> str:
        """
        Build the user prompt with context and history.

        Args:
            query: User's question
            context: Retrieved documents
            history: Conversation history
            is_knowledge_base: True for documentation

        Returns:
            Formatted user prompt string
        """
        parts = []

        # Build context section
        context_header = (
            "Documentation context:" if is_knowledge_base
            else "Context from codebase:"
        )
        parts.append(context_header)
        parts.append("")

        for idx, result in enumerate(context, 1):
            source_name = (
                result.title or result.file_name or "Document"
                if is_knowledge_base
                else f"{result.project or 'unknown'} - {result.file_name or result.metadata.get('file', 'Unknown')}"
            )
            parts.append(f"[Source {idx}: {source_name}]")
            parts.append(result.content)
            parts.append("")
            parts.append("---")
            parts.append("")

        # Add conversation history if present
        if history:
            parts.append("")
            parts.append("Recent conversation history (for context):")
            for msg in history:
                role_name = "User" if msg.role == "user" else "Assistant"
                parts.append(f"{role_name}: {msg.content}")
            parts.append("")

        # Add the current question
        parts.append(f"Question: {query}")
        parts.append("")
        parts.append("Please provide a clear, accurate answer based on the context above.")

        if history:
            parts.append(
                "If this is a follow-up question, use the conversation history "
                "to understand the context."
            )

        return "\n".join(parts)

    def build_context_from_results(
        self,
        results: List[VectorSearchResult],
        max_length: int = 2000
    ) -> str:
        """
        Build a context string from search results.

        Truncates individual results to max_length to fit within
        token limits.

        Args:
            results: Vector search results
            max_length: Max chars per result

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, result in enumerate(results, 1):
            source = result.file_name or result.metadata.get("file", "Unknown")
            project = result.project or "unknown"

            content = result.content
            if len(content) > max_length:
                content = content[:max_length] + "\n...(truncated)"

            context_parts.append(
                f"[Source {idx}: {project} - {source}]\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)


# Module-level singleton accessor
_llm_generation_service: Optional[LLMGenerationService] = None


async def get_llm_generation_service() -> LLMGenerationService:
    """Get or create the global LLM generation service instance."""
    global _llm_generation_service
    if _llm_generation_service is None:
        _llm_generation_service = LLMGenerationService()
        await _llm_generation_service.initialize()
    return _llm_generation_service
