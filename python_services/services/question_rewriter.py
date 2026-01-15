"""
Question Rewriter Service - Context-aware question transformation.

Based on GPT summarization approach (SOTA on CoSQL benchmark).
Transforms follow-up questions to be self-contained by:
1. Resolving pronouns (it, they, those)
2. Expanding temporal references (yesterday, last week)
3. Carrying forward entity references

Architecture:
-------------
The Question Rewriter sits at the front of the SQL generation pipeline.
When conversation history is present, it rewrites ambiguous questions
before they reach the schema loader or LLM generator.

Example:
    User: "Show me ticket counts by user for today"
    Assistant: [returns results]
    User: "What about yesterday?"  # Follow-up with pronoun/reference

    Rewritten: "Show me ticket counts by user for yesterday"
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """Result from question rewriting."""
    original_question: str
    rewritten_question: str
    was_rewritten: bool
    context_used: List[str] = field(default_factory=list)
    confidence: float = 1.0
    rewrite_reason: Optional[str] = None


class QuestionRewriter:
    """
    Rewrites follow-up questions to be self-contained.

    This service uses an LLM to analyze conversation history and
    rewrite ambiguous questions to include all necessary context.

    Usage:
        rewriter = QuestionRewriter(llm_service)
        result = await rewriter.rewrite(
            current_question="What about yesterday?",
            history=[{"question": "Show tickets for today", "sql": "..."}]
        )
        # result.rewritten_question = "Show tickets for yesterday"
    """

    # Prompt template for rewriting questions
    REWRITE_PROMPT_TEMPLATE = '''You are a question rewriter for a SQL query system.

Given the conversation history and a new question, rewrite the question
to be completely self-contained. The rewritten question should:
1. Replace pronouns (it, they, those, etc.) with specific entities
2. Include any implicit filters or context from previous turns
3. Maintain the exact intent of the original question
4. Be a valid standalone question for SQL generation

If the question is already self-contained, return it unchanged.

Conversation History:
{history}

Current Question: "{question}"

Output ONLY the rewritten question, nothing else:'''

    # Patterns that indicate a follow-up question
    FOLLOW_UP_PATTERNS = [
        # Pronouns
        r'\b(it|they|them|those|these|that|this)\b',
        # Temporal follow-ups
        r'\b(what about|how about|and for|but for|now show|instead show|also show)\b',
        # Comparative references
        r'\b(same|also|too|as well|instead|more|less|different)\b',
        # Short continuation questions
        r'^(and|or|but)\s',
    ]

    # Patterns that indicate a self-contained question
    SELF_CONTAINED_PATTERNS = [
        r'^(show|list|get|find|count|how many|what is|which|display)\s+\w+',  # Clear action with subject
    ]

    def __init__(self, llm_service):
        """
        Initialize the question rewriter.

        Args:
            llm_service: LLM service for generation (LLMService instance)
        """
        self.llm = llm_service

    async def rewrite(
        self,
        current_question: str,
        history: List[Dict[str, Any]],
        max_history_turns: int = 5
    ) -> RewriteResult:
        """
        Rewrite a question to be self-contained.

        Args:
            current_question: The current user question
            history: List of previous turns with 'question' and 'sql' keys
            max_history_turns: Maximum history turns to consider

        Returns:
            RewriteResult with rewritten question and metadata
        """
        # If no history, question is already self-contained
        if not history:
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                confidence=1.0,
                rewrite_reason="No conversation history"
            )

        # Check if question is likely self-contained already
        if self._is_self_contained(current_question):
            logger.debug(f"Question appears self-contained: {current_question}")
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                confidence=0.95,
                rewrite_reason="Question appears self-contained"
            )

        # Check if question needs rewriting
        if not self._needs_rewriting(current_question):
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                confidence=0.9,
                rewrite_reason="No follow-up patterns detected"
            )

        # Format history for the prompt
        recent_history = history[-max_history_turns:]
        formatted_history = self._format_history(recent_history)

        # Build the prompt
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            history=formatted_history,
            question=current_question
        )

        # Generate rewritten question
        try:
            result = await self.llm.generate(
                prompt=prompt,
                system="You are a precise question rewriter. Output only the rewritten question.",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200,
                use_cache=False  # Don't cache rewrites as they depend on history
            )

            if not result.success:
                logger.warning(f"Question rewrite failed: {result.error}")
                return RewriteResult(
                    original_question=current_question,
                    rewritten_question=current_question,
                    was_rewritten=False,
                    confidence=0.5,
                    rewrite_reason=f"LLM generation failed: {result.error}"
                )

            rewritten = self._clean_response(result.response)

            # Validate the rewrite
            if not rewritten or len(rewritten) < 5:
                rewritten = current_question
                was_rewritten = False
                confidence = 0.6
                reason = "Rewrite too short, using original"
            else:
                was_rewritten = rewritten.lower().strip() != current_question.lower().strip()
                confidence = 0.85 if was_rewritten else 0.95
                reason = "Successfully rewritten" if was_rewritten else "No rewrite needed"

            context_used = [
                h.get('question', '')[:50]
                for h in recent_history
                if h.get('question')
            ]

            if was_rewritten:
                logger.info(f"Question rewritten: '{current_question}' -> '{rewritten}'")

            return RewriteResult(
                original_question=current_question,
                rewritten_question=rewritten,
                was_rewritten=was_rewritten,
                context_used=context_used,
                confidence=confidence,
                rewrite_reason=reason
            )

        except Exception as e:
            logger.error(f"Question rewrite failed with exception: {e}")
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                confidence=0.5,
                rewrite_reason=f"Exception: {str(e)}"
            )

    def _is_self_contained(self, question: str) -> bool:
        """
        Heuristic check if question is likely self-contained.

        Returns True if the question appears to be complete and
        doesn't reference prior context.
        """
        question_lower = question.lower().strip()

        # Check for self-contained patterns
        for pattern in self.SELF_CONTAINED_PATTERNS:
            if re.search(pattern, question_lower, re.IGNORECASE):
                # Still check for pronouns that might indicate follow-up
                has_pronoun = any(
                    re.search(r'\b' + pronoun + r'\b', question_lower)
                    for pronoun in ['it', 'they', 'them', 'those', 'these']
                )
                if not has_pronoun:
                    return True

        # Long questions are more likely self-contained
        if len(question.split()) >= 8:
            # But check for follow-up indicators
            for pattern in self.FOLLOW_UP_PATTERNS:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return False
            return True

        return False

    def _needs_rewriting(self, question: str) -> bool:
        """
        Check if the question likely needs rewriting based on patterns.

        Returns True if follow-up patterns are detected.
        """
        question_lower = question.lower().strip()

        # Check for very short questions (likely follow-ups)
        if len(question.split()) < 4:
            return True

        # Check for follow-up patterns
        for pattern in self.FOLLOW_UP_PATTERNS:
            if re.search(pattern, question_lower, re.IGNORECASE):
                return True

        # Check for questions starting with conjunctions
        if re.match(r'^(and|or|but|also|what about|how about)\s', question_lower):
            return True

        return False

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Format conversation history for the prompt.

        Handles two formats:
        1. OpenAI-style: {"role": "user/assistant", "content": "..."}
        2. Legacy-style: {"question": "...", "sql": "..."}
        """
        formatted = []
        turn_num = 0

        for turn in history:
            # Handle OpenAI-style format (from frontend)
            if 'role' in turn:
                role = turn.get('role', '')
                content = turn.get('content', '')

                if role == 'user':
                    turn_num += 1
                    formatted.append(f"Turn {turn_num}:")
                    formatted.append(f"  Question: {content}")
                elif role == 'assistant':
                    # Check if content contains SQL (format: "...\nSQL: ...")
                    if '\nSQL:' in content:
                        parts = content.split('\nSQL:')
                        response_text = parts[0].strip()[:100]
                        sql = parts[1].strip() if len(parts) > 1 else ''
                        # Truncate long SQL
                        if len(sql) > 200:
                            sql = sql[:200] + "..."
                        formatted.append(f"  Response: {response_text}")
                        if sql:
                            formatted.append(f"  SQL: {sql}")
                    else:
                        formatted.append(f"  Response: {content[:100]}")
            else:
                # Handle legacy format
                turn_num += 1
                question = turn.get('question', turn.get('user_query', ''))
                sql = turn.get('sql', turn.get('generated_sql', ''))

                # Truncate long SQL for context
                if sql and len(sql) > 200:
                    sql = sql[:200] + "..."

                formatted.append(f"Turn {turn_num}:")
                formatted.append(f"  Question: {question}")
                if sql:
                    formatted.append(f"  SQL: {sql}")

        return "\n".join(formatted)

    def _clean_response(self, response: str) -> str:
        """Clean the LLM response to extract just the question."""
        if not response:
            return ""

        # Remove common prefixes that LLMs sometimes add
        response = response.strip()

        # Remove quotes if wrapped
        response = response.strip('"').strip("'")

        # Remove prefixes like "Rewritten question:" or "Question:"
        prefixes = [
            r'^rewritten\s*question\s*:\s*',
            r'^question\s*:\s*',
            r'^output\s*:\s*',
            r'^answer\s*:\s*',
        ]
        for prefix in prefixes:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE).strip()

        # Remove markdown formatting
        response = response.strip('`')

        # Take only the first line if multiple lines
        if '\n' in response:
            response = response.split('\n')[0].strip()

        return response


# Factory function for easy instantiation
async def create_question_rewriter(llm_service=None):
    """
    Create a QuestionRewriter instance.

    Args:
        llm_service: Optional LLM service. If None, will get singleton instance.

    Returns:
        QuestionRewriter instance
    """
    if llm_service is None:
        from .llm_service import LLMService
        llm_service = await LLMService.get_instance()

    return QuestionRewriter(llm_service)
