"""
Context Builder Service
=======================

Assembles context from retrieved code entities for LLM prompting.

Design Rationale:
-----------------
The ContextBuilder transforms raw retrieval results into a structured
prompt format optimized for code understanding. Key design decisions:

1. Semantic Organization: Context is organized by entity type (methods,
   classes, events) with clear section headers for LLM comprehension.

2. Information Density: Each entity includes key metadata (file, line,
   summary) without overwhelming the context window with raw code.

3. Call Chain Visualization: Call chains are formatted as flow diagrams
   (A -> B -> C) for easy comprehension of execution paths.

4. Token Efficiency: Context is constructed with token limits in mind,
   prioritizing high-similarity results and truncating when necessary.

Architecture:
- Stateless service - all state passed via method arguments
- Template-based formatting for consistency
- Configurable limits to control context size
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from code_assistance_pipeline.models.query_models import (
    SourceInfo,
    SourceType,
    ConversationMessage,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds structured context strings from retrieved code entities.

    This service transforms raw MongoDB results into formatted text
    suitable for LLM prompting. It handles:

    - Method information with signatures and SQL indicators
    - Class structures with inheritance and members
    - Event handler mappings
    - Call chain visualization
    - Conversation history integration

    Usage:
        builder = ContextBuilder()
        context, sources = builder.build_context(
            methods=method_results,
            classes=class_results,
            event_handlers=event_results,
            call_chain=["A.Method1", "B.Method2"]
        )
    """

    # Default limits for context building
    DEFAULT_METHOD_LIMIT = 8
    DEFAULT_CLASS_LIMIT = 4
    DEFAULT_EVENT_LIMIT = 3

    # Section separators for readability
    SECTION_SEPARATOR = "\n---\n"

    def __init__(
        self,
        method_limit: int = DEFAULT_METHOD_LIMIT,
        class_limit: int = DEFAULT_CLASS_LIMIT,
        event_limit: int = DEFAULT_EVENT_LIMIT
    ):
        """
        Initialize the context builder with configurable limits.

        Args:
            method_limit: Maximum methods to include in context
            class_limit: Maximum classes to include in context
            event_limit: Maximum event handlers to include in context
        """
        self.method_limit = method_limit
        self.class_limit = class_limit
        self.event_limit = event_limit

    def build_context(
        self,
        methods: List[Dict[str, Any]],
        classes: List[Dict[str, Any]],
        event_handlers: List[Dict[str, Any]],
        call_chain: List[str],
        history: Optional[List[ConversationMessage]] = None
    ) -> Tuple[str, List[SourceInfo]]:
        """
        Build complete context string from retrieved entities.

        Assembles context in the following order:
        1. Conversation history (if present)
        2. Methods section
        3. Classes section
        4. Event handlers section (if present)
        5. Call chain section (if present)

        Args:
            methods: List of method dictionaries from retrieval
            classes: List of class dictionaries from retrieval
            event_handlers: List of event handler dictionaries
            call_chain: List of method names in call flow order
            history: Optional conversation history for multi-turn

        Returns:
            Tuple of (context_string, sources_list)

        Design Note:
        The context is built incrementally with sources tracked
        alongside. This enables the response to cite specific
        sources that informed the answer.
        """
        context_parts: List[str] = []
        sources: List[SourceInfo] = []

        # Add conversation history if present
        if history:
            history_context = self._format_history(history)
            if history_context:
                context_parts.append(history_context)

        # Add methods
        if methods:
            method_context, method_sources = self._format_methods(methods)
            context_parts.append(method_context)
            sources.extend(method_sources)

        # Add classes
        if classes:
            class_context, class_sources = self._format_classes(classes)
            context_parts.append(class_context)
            sources.extend(class_sources)

        # Add event handlers
        if event_handlers:
            event_context, event_sources = self._format_event_handlers(event_handlers)
            context_parts.append(event_context)
            sources.extend(event_sources)

        # Add call chain
        if call_chain:
            chain_context = self._format_call_chain(call_chain)
            context_parts.append(chain_context)

        context_str = self.SECTION_SEPARATOR.join(context_parts)

        logger.debug(
            f"Built context with {len(methods)} methods, {len(classes)} classes, "
            f"{len(event_handlers)} events, {len(call_chain)} call chain items"
        )

        return context_str, sources

    def _format_history(self, history: List[ConversationMessage]) -> str:
        """
        Format conversation history for context.

        Only includes the last 4 messages to stay within token limits
        while maintaining relevant conversation context.

        Args:
            history: List of conversation messages

        Returns:
            Formatted history string
        """
        if not history:
            return ""

        recent = history[-4:]  # Last 4 messages
        formatted = []

        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:500]  # Truncate long messages
            formatted.append(f"{role_label}: {content}")

        return "Previous conversation:\n" + "\n\n".join(formatted)

    def _format_methods(
        self,
        methods: List[Dict[str, Any]]
    ) -> Tuple[str, List[SourceInfo]]:
        """
        Format method information for context.

        Each method includes:
        - Fully qualified name
        - File location
        - Signature
        - Summary (if available)
        - SQL operation indicator (if present)

        Args:
            methods: List of method dictionaries

        Returns:
            Tuple of (formatted_string, sources_list)
        """
        parts = []
        sources = []

        for method in methods[:self.method_limit]:
            class_name = method.get("class_name", "")
            method_name = method.get("method_name", "")
            full_name = f"{class_name}.{method_name}" if class_name else method_name

            lines = [f"Method: {full_name}"]

            # File location
            file_path = method.get("file_path", "Unknown")
            line_number = method.get("line_number", 0)
            lines.append(f"File: {file_path}:{line_number}")

            # Signature
            signature = method.get("signature", "")
            if signature:
                lines.append(f"Signature: {signature}")

            # Summary
            summary = method.get("summary", "")
            if summary:
                lines.append(f"Summary: {summary}")
            else:
                lines.append("Summary: No summary available")

            # SQL operations indicator
            sql_calls = method.get("sql_calls", [])
            if sql_calls:
                lines.append(f"SQL Operations: {len(sql_calls)} database calls")

            parts.append("\n".join(lines))

            # Build source info
            sources.append(SourceInfo(
                type=SourceType.METHOD,
                name=full_name,
                file=file_path,
                line=line_number,
                similarity=method.get("similarity", 0.0),
                snippet=signature or f"{method_name}(...)"
            ))

        return "\n\n".join(parts), sources

    def _format_classes(
        self,
        classes: List[Dict[str, Any]]
    ) -> Tuple[str, List[SourceInfo]]:
        """
        Format class information for context.

        Each class includes:
        - Fully qualified name
        - File location
        - Base class (if any)
        - Implemented interfaces
        - Key methods overview

        Args:
            classes: List of class dictionaries

        Returns:
            Tuple of (formatted_string, sources_list)
        """
        parts = []
        sources = []

        for cls in classes[:self.class_limit]:
            namespace = cls.get("namespace", "")
            class_name = cls.get("class_name", "")
            full_name = f"{namespace}.{class_name}" if namespace else class_name

            lines = [f"Class: {full_name}"]

            # File location
            file_path = cls.get("file_path", "Unknown")
            lines.append(f"File: {file_path}")

            # Inheritance
            base_class = cls.get("base_class", "")
            if base_class:
                lines.append(f"Base Class: {base_class}")
            else:
                lines.append("Base Class: None")

            # Interfaces
            interfaces = cls.get("interfaces", [])
            if interfaces:
                lines.append(f"Interfaces: {', '.join(interfaces)}")
            else:
                lines.append("Interfaces: None")

            # Methods overview
            methods = cls.get("methods", [])
            if methods:
                method_preview = ", ".join(methods[:5])
                if len(methods) > 5:
                    method_preview += f"... ({len(methods)} total)"
                lines.append(f"Methods: {method_preview}")
            else:
                lines.append("Methods: None")

            parts.append("\n".join(lines))

            # Build source info
            snippet = f"class {class_name}"
            if base_class:
                snippet += f" : {base_class}"

            sources.append(SourceInfo(
                type=SourceType.CLASS,
                name=full_name,
                file=file_path,
                line=0,
                similarity=cls.get("similarity", 0.0),
                snippet=snippet
            ))

        return "\n\n".join(parts), sources

    def _format_event_handlers(
        self,
        event_handlers: List[Dict[str, Any]]
    ) -> Tuple[str, List[SourceInfo]]:
        """
        Format event handler information for context.

        Each handler includes:
        - Event name and handler method
        - UI element information
        - Handler class

        Args:
            event_handlers: List of event handler dictionaries

        Returns:
            Tuple of (formatted_string, sources_list)
        """
        parts = []
        sources = []

        for handler in event_handlers[:self.event_limit]:
            event_name = handler.get("event_name", "Unknown")
            handler_method = handler.get("handler_method", "Unknown")

            lines = [f"Event Handler: {event_name} -> {handler_method}"]

            # UI element
            element_name = handler.get("element_name", "")
            element_type = handler.get("ui_element_type", "")
            if element_name or element_type:
                lines.append(f"UI Element: {element_name} ({element_type})")

            # Handler class
            handler_class = handler.get("handler_class", "")
            if handler_class:
                lines.append(f"Handler Class: {handler_class}")

            parts.append("\n".join(lines))

            # Build source info
            sources.append(SourceInfo(
                type=SourceType.EVENT_HANDLER,
                name=f"{event_name} -> {handler_method}",
                file=handler.get("file_path", ""),
                line=handler.get("line_number", 0),
                similarity=handler.get("similarity", 0.0),
                snippet=f"{element_name}.{event_name} += {handler_method}"
            ))

        return "\n\n".join(parts), sources

    def _format_call_chain(self, call_chain: List[str]) -> str:
        """
        Format call chain as a flow diagram.

        Converts ["A.Method1", "B.Method2", "C.Method3"] to:
        "Call Flow: A.Method1 -> B.Method2 -> C.Method3"

        Args:
            call_chain: List of method names in order

        Returns:
            Formatted call chain string
        """
        if not call_chain:
            return ""

        return f"Call Flow:\n{' -> '.join(call_chain)}"

    def build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build the complete LLM prompt from query and context.

        Combines system instructions, retrieved context, and user
        query into a structured prompt format.

        Args:
            query: User's natural language question
            context: Pre-built context string
            system_prompt: Optional custom system prompt

        Returns:
            Complete prompt string for LLM

        Design Note:
        The prompt structure emphasizes:
        1. Clear role definition (code assistant)
        2. Instructions for using context
        3. Output format guidance (cite code, explain flow)
        """
        default_system = """You are a helpful code assistant that explains C# code to developers.
Based on the following code context, answer the user's question.
Always cite specific methods, classes, and file paths in your answer.
If you're not sure about something, say so."""

        system = system_prompt or default_system

        prompt_parts = [
            system,
            "",
            "CODE CONTEXT:",
            context,
            "",
            f"USER QUESTION: {query}",
            "",
            """Provide a clear, concise answer that:
1. Directly addresses the question
2. References specific code elements (ClassName.MethodName format)
3. Explains the flow if relevant
4. Mentions file paths for key code

ANSWER:"""
        ]

        return "\n".join(prompt_parts)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Uses a simple word-based heuristic (1.3 tokens per word)
        which is reasonably accurate for code-mixed content.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        Design Note:
        This is an approximation. For precise token counting,
        use the actual tokenizer for the target model.
        """
        words = text.split()
        return int(len(words) * 1.3)

    def truncate_context(
        self,
        context: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Truncate context to fit within token limits.

        Truncates from the end while preserving complete sections.

        Args:
            context: Full context string
            max_tokens: Maximum allowed tokens

        Returns:
            Truncated context string

        Design Note:
        Truncation is a last resort. Better to limit retrieval
        results upstream than to truncate assembled context.
        """
        estimated = self.estimate_tokens(context)
        if estimated <= max_tokens:
            return context

        # Simple truncation - split by sections and remove from end
        sections = context.split(self.SECTION_SEPARATOR)
        truncated_sections = []
        current_tokens = 0

        for section in sections:
            section_tokens = self.estimate_tokens(section)
            if current_tokens + section_tokens > max_tokens:
                break
            truncated_sections.append(section)
            current_tokens += section_tokens

        truncated = self.SECTION_SEPARATOR.join(truncated_sections)
        logger.warning(
            f"Context truncated from ~{estimated} to ~{self.estimate_tokens(truncated)} tokens"
        )

        return truncated
