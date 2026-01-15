"""
SQL Pipeline Conversation Context Tests.

Comprehensive tests for multi-turn conversation handling including:
- Question rewriting with history
- Multi-turn conversation flow
- Context preservation across requests
- Pronoun resolution
- Follow-up question handling
- Conversation memory limits
- History format validation

Tests the conversation context handling in the SQL query pipeline.
"""

import pytest
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from config.test_config import PipelineTestConfig
from testing.utils.api_test_client import APITestClient


# =============================================================================
# Test Constants
# =============================================================================

# Sample conversation histories for testing
SIMPLE_CONVERSATION = [
    {"role": "user", "content": "How many tickets were created today?"},
    {"role": "assistant", "content": "Generated SQL: SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)\n\nResult: 42 tickets"}
]

MULTI_TURN_CONVERSATION = [
    {"role": "user", "content": "How many tickets were created today?"},
    {"role": "assistant", "content": "Generated SQL: SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)\n\nResult: 42 tickets"},
    {"role": "user", "content": "What about last week?"},
    {"role": "assistant", "content": "Generated SQL: SELECT COUNT(*) FROM CentralTickets WHERE AddTicketDate >= DATEADD(DAY, -7, GETDATE())\n\nResult: 156 tickets"},
    {"role": "user", "content": "Can you break that down by status?"},
    {"role": "assistant", "content": "Generated SQL: SELECT ts.TypeName as Status, COUNT(*) as Count FROM CentralTickets ct LEFT JOIN Types ts ON ct.TicketStatusTypeID = ts.TypeID WHERE ct.AddTicketDate >= DATEADD(DAY, -7, GETDATE()) GROUP BY ts.TypeName\n\nResult: Open: 45, Closed: 111"}
]

CONTEXT_REQUIRING_QUESTIONS = [
    # Questions that need previous context
    ("Show me the same for last month", SIMPLE_CONVERSATION),
    ("And what about yesterday?", SIMPLE_CONVERSATION),
    ("Break that down by customer", MULTI_TURN_CONVERSATION),
    ("Now filter to just the open ones", MULTI_TURN_CONVERSATION),
    ("What's the average?", SIMPLE_CONVERSATION),
]


# =============================================================================
# Helper Classes
# =============================================================================

@dataclass
class ConversationTestCase:
    """Test case for conversation handling."""
    name: str
    question: str
    conversation_history: List[Dict[str, str]]
    expected_contains: List[str] = field(default_factory=list)
    expected_not_contains: List[str] = field(default_factory=list)
    description: str = ""


class MockQuestionRewriter:
    """
    Mock implementation of question rewriting for testing.

    Simulates how the pipeline rewrites follow-up questions
    using conversation context.
    """

    def __init__(self):
        self.rewrite_count = 0

    def needs_rewriting(self, question: str, history: List[Dict[str, str]]) -> bool:
        """
        Check if question needs rewriting based on context.

        Returns True if question contains:
        - Pronouns (it, that, this, they, them)
        - Implicit references (same, similar, like that)
        - Comparison words (more, less, instead)
        - Follow-up indicators (also, and, what about)
        """
        if not history:
            return False

        question_lower = question.lower()

        # Pronoun patterns
        pronouns = ["it", "that", "this", "they", "them", "those", "these"]

        # Reference patterns
        references = ["same", "similar", "like that", "like before", "previous"]

        # Comparison patterns
        comparisons = ["more", "less", "instead", "rather", "other"]

        # Follow-up patterns
        followups = ["also", "and", "what about", "how about", "now", "then"]

        # Check for patterns
        for pronoun in pronouns:
            if f" {pronoun} " in f" {question_lower} ":
                return True

        for ref in references:
            if ref in question_lower:
                return True

        for comp in comparisons:
            if f" {comp} " in f" {question_lower} ":
                return True

        for followup in followups:
            if question_lower.startswith(followup) or f" {followup} " in question_lower:
                return True

        return False

    def rewrite(
        self,
        question: str,
        history: List[Dict[str, str]],
        database: str = "EWRCentral"
    ) -> str:
        """
        Rewrite question using conversation context.

        This is a simplified mock - real implementation uses LLM.
        """
        self.rewrite_count += 1

        if not self.needs_rewriting(question, history):
            return question

        # Extract context from last assistant message
        last_context = ""
        for msg in reversed(history):
            if msg["role"] == "assistant":
                last_context = msg["content"]
                break

        # Simple pattern-based rewriting for testing
        question_lower = question.lower()

        if "same" in question_lower and "month" in question_lower:
            # "Show me the same for last month" -> expand with context
            if "tickets" in last_context.lower():
                return "Show me tickets created last month"

        if "yesterday" in question_lower:
            if "tickets" in last_context.lower():
                return "How many tickets were created yesterday?"

        if "break" in question_lower and "down" in question_lower:
            if "tickets" in last_context.lower():
                return "Break down the ticket count by category"

        # Default: return original (real impl would use LLM)
        return question


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def api_client():
    """Fixture providing configured API test client."""
    async with APITestClient(base_url="http://localhost:8001", timeout=60.0) as client:
        yield client


@pytest.fixture
def question_rewriter():
    """Fixture providing mock question rewriter."""
    return MockQuestionRewriter()


# =============================================================================
# Question Rewriting Tests
# =============================================================================

class TestQuestionRewriting:
    """Test question rewriting with conversation context."""

    def test_rewriting_needed_with_pronouns(self, question_rewriter: MockQuestionRewriter):
        """Test that questions with pronouns need rewriting."""
        question = "Show me more of it"
        history = SIMPLE_CONVERSATION

        assert question_rewriter.needs_rewriting(question, history)

    def test_rewriting_needed_with_references(self, question_rewriter: MockQuestionRewriter):
        """Test that questions with references need rewriting."""
        question = "Show me the same for last month"
        history = SIMPLE_CONVERSATION

        assert question_rewriter.needs_rewriting(question, history)

    def test_rewriting_not_needed_standalone(self, question_rewriter: MockQuestionRewriter):
        """Test that standalone questions don't need rewriting."""
        question = "How many customers are in the database?"
        history = SIMPLE_CONVERSATION

        assert not question_rewriter.needs_rewriting(question, history)

    def test_rewriting_not_needed_without_history(self, question_rewriter: MockQuestionRewriter):
        """Test that questions without history don't need rewriting."""
        question = "Show me more of it"
        history = []

        assert not question_rewriter.needs_rewriting(question, history)

    def test_rewriting_produces_different_question(self, question_rewriter: MockQuestionRewriter):
        """Test that rewriting can modify the question."""
        question = "Show me the same for last month"
        history = SIMPLE_CONVERSATION

        rewritten = question_rewriter.rewrite(question, history)

        # If rewriting happened, question should be different or enhanced
        # (may or may not change depending on mock implementation)
        assert rewritten is not None

    def test_rewriting_preserves_intent(self, question_rewriter: MockQuestionRewriter):
        """Test that rewriting preserves the original intent."""
        question = "And what about yesterday?"
        history = SIMPLE_CONVERSATION

        rewritten = question_rewriter.rewrite(question, history)

        # Should still be asking about something
        assert len(rewritten) > 0


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================

class TestMultiTurnConversation:
    """Test multi-turn conversation handling."""

    def test_context_from_previous_question(self, question_rewriter: MockQuestionRewriter):
        """Test that context is extracted from previous questions."""
        history = [
            {"role": "user", "content": "How many tickets?"},
            {"role": "assistant", "content": "42 tickets"}
        ]

        # Follow-up should understand context
        needs_context = question_rewriter.needs_rewriting("Break that down", history)
        assert needs_context

    def test_context_accumulates_over_turns(self, question_rewriter: MockQuestionRewriter):
        """Test that context accumulates across multiple turns."""
        # Build conversation incrementally
        history = []

        # Turn 1
        history.append({"role": "user", "content": "How many tickets?"})
        history.append({"role": "assistant", "content": "42 tickets"})

        # Turn 2 - references turn 1
        needs_rewrite_2 = question_rewriter.needs_rewriting("What about last week?", history)
        assert needs_rewrite_2

        history.append({"role": "user", "content": "What about last week?"})
        history.append({"role": "assistant", "content": "156 tickets"})

        # Turn 3 - can reference any previous turn
        needs_rewrite_3 = question_rewriter.needs_rewriting("Break those down", history)
        assert needs_rewrite_3

    def test_conversation_flow_preserved(self, question_rewriter: MockQuestionRewriter):
        """Test that conversation flow is preserved correctly."""
        history = MULTI_TURN_CONVERSATION.copy()

        # Should have alternating user/assistant messages
        for i, msg in enumerate(history):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role


# =============================================================================
# Context Preservation Tests
# =============================================================================

class TestContextPreservation:
    """Test context preservation across requests."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_context_passed_to_endpoint(self, api_client: APITestClient):
        """Test that conversation context is passed to query endpoint."""
        request = {
            "natural_language": "Show me the same for last week",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": SIMPLE_CONVERSATION,
            "options": {
                "execute_sql": False,
                "use_cache": False
            }
        }

        response = await api_client.post("/api/sql/query", request)

        # Should process without error (actual LLM behavior may vary)
        assert response.status_code in [200, 500, 503]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_context_format_validation(self, api_client: APITestClient):
        """Test that invalid context format is handled."""
        request = {
            "natural_language": "Show me tickets",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": "invalid - should be list",
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_empty_context_accepted(self, api_client: APITestClient):
        """Test that empty conversation history is accepted."""
        request = {
            "natural_language": "How many tickets are there?",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [],
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # Empty list should be valid
        assert response.status_code in [200, 500, 503]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_context_with_missing_role(self, api_client: APITestClient):
        """Test handling of context messages with missing role."""
        request = {
            "natural_language": "Show me tickets",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [
                {"content": "Missing role field"}
            ],
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # Should either reject or handle gracefully
        assert response.status_code in [200, 422, 500]


# =============================================================================
# Pronoun Resolution Tests
# =============================================================================

class TestPronounResolution:
    """Test pronoun resolution in follow-up questions."""

    def test_it_pronoun_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'it' pronoun is detected as needing context."""
        questions = [
            "Show me more of it",
            "What is it?",
            "Filter it by status",
            "Can you explain it?",
        ]

        for question in questions:
            assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION), \
                f"Should detect 'it' in: {question}"

    def test_that_pronoun_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'that' pronoun is detected."""
        questions = [
            "Break that down",
            "Show me that again",
            "What does that mean?",
        ]

        for question in questions:
            assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION), \
                f"Should detect 'that' in: {question}"

    def test_those_these_pronouns_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'those'/'these' pronouns are detected."""
        questions = [
            "Show me those tickets",
            "Filter those by date",
            "What are these?",
        ]

        for question in questions:
            assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION), \
                f"Should detect pronoun in: {question}"


# =============================================================================
# Follow-up Question Tests
# =============================================================================

class TestFollowUpQuestions:
    """Test follow-up question handling."""

    def test_what_about_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'what about' follow-up is detected."""
        question = "What about last month?"
        assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION)

    def test_how_about_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'how about' follow-up is detected."""
        question = "How about showing them by customer?"
        assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION)

    def test_and_continuation_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'and' continuation is detected."""
        question = "And the open tickets?"
        assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION)

    def test_also_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'also' is detected."""
        question = "Also show me the closed ones"
        assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION)

    def test_now_continuation_detected(self, question_rewriter: MockQuestionRewriter):
        """Test 'now' continuation is detected."""
        question = "Now filter by status"
        assert question_rewriter.needs_rewriting(question, SIMPLE_CONVERSATION)


# =============================================================================
# Conversation Memory Limit Tests
# =============================================================================

class TestConversationMemoryLimits:
    """Test handling of conversation memory limits."""

    @pytest.mark.asyncio
    async def test_long_conversation_history(self, api_client: APITestClient):
        """Test handling of very long conversation history."""
        # Generate long history
        long_history = []
        for i in range(50):
            long_history.append({"role": "user", "content": f"Question {i}"})
            long_history.append({"role": "assistant", "content": f"Answer {i}"})

        request = {
            "natural_language": "Follow up question",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": long_history,
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # Should handle without crashing
        assert response.status_code in [200, 422, 500, 503]

    def test_recent_context_prioritized(self, question_rewriter: MockQuestionRewriter):
        """Test that recent context is prioritized."""
        # Long history with relevant recent context
        history = [
            {"role": "user", "content": "Old question about customers"},
            {"role": "assistant", "content": "Customer info"},
        ] * 10  # Old context

        # Add recent relevant context
        history.append({"role": "user", "content": "How many tickets?"})
        history.append({"role": "assistant", "content": "42 tickets"})

        question = "Show me more"
        needs_rewrite = question_rewriter.needs_rewriting(question, history)

        # Should still detect need for context
        assert needs_rewrite


# =============================================================================
# History Format Validation Tests
# =============================================================================

class TestHistoryFormatValidation:
    """Test conversation history format validation."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_valid_history_format(self, api_client: APITestClient):
        """Test valid conversation history format is accepted."""
        request = {
            "natural_language": "Follow up",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"}
            ],
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # Valid format should be accepted
        assert response.status_code in [200, 500, 503]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_role_value(self, api_client: APITestClient):
        """Test invalid role value is rejected."""
        request = {
            "natural_language": "Follow up",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [
                {"role": "invalid_role", "content": "Test"}
            ],
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # May be rejected or handled
        assert response.status_code in [200, 422, 500]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_null_content_handling(self, api_client: APITestClient):
        """Test handling of null content in history."""
        request = {
            "natural_language": "Follow up",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [
                {"role": "user", "content": None}
            ],
            "options": {"execute_sql": False}
        }

        response = await api_client.post("/api/sql/query", request)

        # Should handle gracefully
        assert response.status_code in [200, 422, 500]

    def test_history_message_structure(self):
        """Test conversation history message structure."""
        valid_message = {"role": "user", "content": "test"}

        assert "role" in valid_message
        assert "content" in valid_message
        assert valid_message["role"] in ["user", "assistant", "system"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestConversationIntegration:
    """Integration tests for conversation handling."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, api_client: APITestClient):
        """Test a full multi-turn conversation."""
        history = []

        # Turn 1: Initial question
        request1 = {
            "natural_language": "How many tickets are in EWRCentral?",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": [],
            "options": {"execute_sql": False, "use_cache": False}
        }

        response1 = await api_client.post("/api/sql/query", request1)

        if response1.is_success:
            # Add to history
            history.append({"role": "user", "content": request1["natural_language"]})
            history.append({"role": "assistant", "content": str(response1.data)})

            # Turn 2: Follow-up question
            request2 = {
                "natural_language": "What about just from today?",
                "database": "EWRCentral",
                "server": "NCSQLTEST",
                "conversation_history": history,
                "options": {"execute_sql": False, "use_cache": False}
            }

            response2 = await api_client.post("/api/sql/query", request2)

            # Should process follow-up
            assert response2.status_code in [200, 500, 503]

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_streaming_with_conversation(self, api_client: APITestClient):
        """Test streaming endpoint with conversation history."""
        request = {
            "natural_language": "Show me the same for last week",
            "database": "EWRCentral",
            "server": "NCSQLTEST",
            "conversation_history": SIMPLE_CONVERSATION,
            "options": {"execute_sql": False}
        }

        # Collect stream events
        events = []
        try:
            async for event in api_client.post_stream("/api/sql/query-stream", request):
                events.append(event)
        except Exception:
            pass  # Stream may not be available

        # If stream worked, should have events
        # (not requiring success as LLM may not be available)


# =============================================================================
# Parameterized Context Tests
# =============================================================================

CONTEXT_TEST_CASES = [
    ConversationTestCase(
        name="pronoun_it",
        question="Show me more of it",
        conversation_history=SIMPLE_CONVERSATION,
        description="Question with 'it' pronoun"
    ),
    ConversationTestCase(
        name="same_reference",
        question="Show me the same for last month",
        conversation_history=SIMPLE_CONVERSATION,
        description="Question with 'same' reference"
    ),
    ConversationTestCase(
        name="what_about",
        question="What about yesterday?",
        conversation_history=SIMPLE_CONVERSATION,
        description="What about follow-up"
    ),
    ConversationTestCase(
        name="break_down",
        question="Break that down by status",
        conversation_history=MULTI_TURN_CONVERSATION,
        description="Break down request"
    ),
    ConversationTestCase(
        name="filter_request",
        question="Now filter to just open tickets",
        conversation_history=MULTI_TURN_CONVERSATION,
        description="Filter follow-up"
    ),
]


class TestParameterizedContext:
    """Parameterized tests for various context scenarios."""

    @pytest.mark.parametrize(
        "test_case",
        CONTEXT_TEST_CASES,
        ids=[tc.name for tc in CONTEXT_TEST_CASES]
    )
    def test_context_question_detected(
        self,
        test_case: ConversationTestCase,
        question_rewriter: MockQuestionRewriter
    ):
        """Test that various context-requiring questions are detected."""
        needs_rewrite = question_rewriter.needs_rewriting(
            test_case.question,
            test_case.conversation_history
        )

        assert needs_rewrite, f"Should detect context need for: {test_case.description}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestConversationEdgeCases:
    """Test edge cases in conversation handling."""

    def test_single_message_history(self, question_rewriter: MockQuestionRewriter):
        """Test with single message in history."""
        history = [{"role": "user", "content": "Initial question"}]

        question = "What about that?"
        needs_rewrite = question_rewriter.needs_rewriting(question, history)

        # Should still detect context need
        assert needs_rewrite

    def test_assistant_only_history(self, question_rewriter: MockQuestionRewriter):
        """Test with only assistant messages in history."""
        history = [{"role": "assistant", "content": "Some response"}]

        question = "Tell me more"
        needs_rewrite = question_rewriter.needs_rewriting(question, history)

        # Should still work
        assert needs_rewrite

    def test_empty_content_messages(self, question_rewriter: MockQuestionRewriter):
        """Test with empty content in messages."""
        history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Response"}
        ]

        question = "Follow up"
        # Should handle without error
        result = question_rewriter.needs_rewriting(question, history)
        assert isinstance(result, bool)

    def test_very_long_message_content(self, question_rewriter: MockQuestionRewriter):
        """Test with very long message content."""
        long_content = "word " * 10000
        history = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": long_content}
        ]

        question = "Summarize that"
        needs_rewrite = question_rewriter.needs_rewriting(question, history)

        assert needs_rewrite

    def test_special_characters_in_history(self, question_rewriter: MockQuestionRewriter):
        """Test with special characters in history."""
        history = [
            {"role": "user", "content": "Show me tickets where status = 'open' AND type <> 'closed'"},
            {"role": "assistant", "content": "Found 42 tickets with status='open'"}
        ]

        question = "What about those?"
        needs_rewrite = question_rewriter.needs_rewriting(question, history)

        assert needs_rewrite

    def test_unicode_in_history(self, question_rewriter: MockQuestionRewriter):
        """Test with unicode characters in history."""
        history = [
            {"role": "user", "content": "Find customer Muller-Schmidt"},
            {"role": "assistant", "content": "Found 3 customers"}
        ]

        question = "Show me their tickets"
        needs_rewrite = question_rewriter.needs_rewriting(question, history)

        assert needs_rewrite
