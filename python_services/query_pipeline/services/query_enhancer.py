"""
Query Enhancer Service
======================

Enhances queries with conversation history for better vector search.

Design Rationale:
-----------------
Follow-up questions often use pronouns or implicit references:
- "What does it do?" (referring to a feature mentioned earlier)
- "How is it configured?" (referring to a component)
- "Show me more details" (referring to previous context)

Without enhancement, vector search may fail to find relevant documents
because the query alone lacks sufficient semantic content.

The QueryEnhancer solves this by:
1. Detecting follow-up patterns (pronouns, short questions)
2. Extracting relevant context from conversation history
3. Building an enhanced query for vector search
4. Keeping the original query for LLM prompting

Enhancement Strategies:
-----------------------
1. **Pronoun detection**: Queries with "it", "its", "they", etc.
2. **Length heuristic**: Very short queries (<50 chars) are likely follow-ups
3. **Question patterns**: "How", "What", "Why" without subject
4. **Reference extraction**: Pull key terms from recent messages

Best Practices:
- Use enhanced query ONLY for vector search (not LLM prompting)
- Limit history window to 2-3 messages for relevance
- Combine, don't replace, original query terms
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass

from query_pipeline.models.query_models import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for query enhancement."""
    max_history_messages: int = 2       # Look back at last N user messages
    min_query_length: int = 50          # Queries shorter than this may be follow-ups
    pronoun_patterns: List[str] = None  # Patterns indicating follow-ups

    def __post_init__(self):
        if self.pronoun_patterns is None:
            self.pronoun_patterns = [
                r'\bit\b',      # "it"
                r'\bits\b',     # "its"
                r'\bthey\b',    # "they"
                r'\bthis\b',    # "this"
                r'\bthat\b',    # "that"
                r'\bthese\b',   # "these"
                r'\bthose\b',   # "those"
            ]


@dataclass
class EnhancementResult:
    """Result of query enhancement."""
    original_query: str
    enhanced_query: str
    was_enhanced: bool
    enhancement_reason: Optional[str] = None


class QueryEnhancer:
    """
    Enhances queries with conversation context for better retrieval.

    This service analyzes queries to detect follow-up patterns and
    enriches them with context from conversation history. The enhanced
    query is used for vector search while the original is preserved
    for LLM prompting.

    Usage:
        enhancer = QueryEnhancer()

        result = enhancer.enhance(
            query="How does it handle errors?",
            history=[
                {"role": "user", "content": "Tell me about RecapGet"},
                {"role": "assistant", "content": "RecapGet is..."}
            ]
        )

        # Use result.enhanced_query for vector search
        # Use result.original_query for LLM prompt
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        """
        Initialize the enhancer.

        Args:
            config: Enhancement configuration
        """
        self.config = config or EnhancementConfig()
        self._pronoun_regex = re.compile(
            '|'.join(self.config.pronoun_patterns),
            re.IGNORECASE
        )
        self._question_starters = {'how', 'what', 'why', 'when', 'where', 'which', 'who'}

    def enhance(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None
    ) -> EnhancementResult:
        """
        Enhance a query with conversation context.

        Args:
            query: The user's current query
            history: List of previous chat messages

        Returns:
            EnhancementResult with original and enhanced queries
        """
        if not history:
            return EnhancementResult(
                original_query=query,
                enhanced_query=query,
                was_enhanced=False
            )

        # Check if this looks like a follow-up query
        is_follow_up, reason = self._detect_follow_up(query)

        if not is_follow_up:
            return EnhancementResult(
                original_query=query,
                enhanced_query=query,
                was_enhanced=False
            )

        # Extract context from history
        context_terms = self._extract_context(history)

        if not context_terms:
            return EnhancementResult(
                original_query=query,
                enhanced_query=query,
                was_enhanced=False
            )

        # Build enhanced query
        enhanced = f"{context_terms} {query}"

        logger.info(f"Enhanced query: '{query}' -> '{enhanced[:100]}...'")

        return EnhancementResult(
            original_query=query,
            enhanced_query=enhanced,
            was_enhanced=True,
            enhancement_reason=reason
        )

    def _detect_follow_up(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Detect if a query is likely a follow-up question.

        Uses multiple heuristics:
        1. Contains pronouns like "it", "its", "they"
        2. Is very short (under threshold)
        3. Starts with question word without clear subject

        Args:
            query: The query to analyze

        Returns:
            Tuple of (is_follow_up, reason)
        """
        query_lower = query.lower().strip()

        # Check for pronouns
        if self._pronoun_regex.search(query_lower):
            return True, "pronoun_detected"

        # Check for short query
        if len(query) < self.config.min_query_length:
            # Also check if it starts with a question word
            first_word = query_lower.split()[0] if query_lower.split() else ""
            if first_word in self._question_starters:
                return True, "short_question"

        # Check for "more" or "also" patterns
        if any(word in query_lower for word in ['more about', 'also', 'another', 'else']):
            return True, "continuation_pattern"

        return False, None

    def _extract_context(self, history: List[ChatMessage]) -> str:
        """
        Extract relevant context terms from conversation history.

        Focuses on user messages (not assistant responses) and
        extracts key terms that might be referenced.

        Args:
            history: List of chat messages

        Returns:
            String of context terms to prepend to query
        """
        # Get recent user messages
        user_messages = [
            msg.content for msg in history
            if msg.role == "user"
        ][-self.config.max_history_messages:]

        if not user_messages:
            return ""

        # Combine recent user messages
        context = " ".join(user_messages)

        # Clean up and truncate
        context = self._clean_context(context)

        return context

    def _clean_context(self, context: str) -> str:
        """
        Clean and normalize context text.

        Removes excessive whitespace, limits length, and
        filters out common stop words.

        Args:
            context: Raw context string

        Returns:
            Cleaned context string
        """
        # Normalize whitespace
        context = re.sub(r'\s+', ' ', context).strip()

        # Limit length (to avoid overwhelming the query)
        max_context_length = 200
        if len(context) > max_context_length:
            # Try to cut at word boundary
            context = context[:max_context_length]
            last_space = context.rfind(' ')
            if last_space > max_context_length // 2:
                context = context[:last_space]

        return context

    def should_enhance(self, query: str, history: Optional[List[ChatMessage]]) -> bool:
        """
        Quick check if enhancement is likely beneficial.

        Useful for skipping the full enhancement process when not needed.

        Args:
            query: The query to check
            history: Conversation history

        Returns:
            True if enhancement should be attempted
        """
        if not history:
            return False

        is_follow_up, _ = self._detect_follow_up(query)
        return is_follow_up


# Module-level singleton accessor
_query_enhancer: Optional[QueryEnhancer] = None


def get_query_enhancer(config: Optional[EnhancementConfig] = None) -> QueryEnhancer:
    """Get or create the global query enhancer instance."""
    global _query_enhancer
    if _query_enhancer is None:
        _query_enhancer = QueryEnhancer(config)
    return _query_enhancer
