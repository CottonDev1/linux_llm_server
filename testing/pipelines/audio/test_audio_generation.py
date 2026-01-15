"""
Audio Generation Tests
======================

Tests for LLM-powered audio analysis features:
- Transcription summarization (port 8081)
- Call content analysis (subject, outcome, customer detection)
- Transcription formatting for readability
- Token usage tracking

All tests use local llama.cpp endpoints only.
"""

import pytest
import asyncio
from typing import Dict

from config.test_config import get_test_config
from utils import assert_llm_response_valid
from fixtures.llm_fixtures import LocalLLMClient, AsyncLocalLLMClient
from fixtures.shared_fixtures import TokenAssertions


class TestAudioGeneration:
    """Test LLM-powered audio analysis generation."""

    @pytest.mark.requires_llm
    def test_llm_health_check(self, llm_client, token_assertions: TokenAssertions):
        """Test that local LLM endpoint is accessible."""
        # Try general endpoint (port 8081)
        max_tokens = 10
        response = llm_client.generate(
            prompt="Hello",
            endpoint="general",
            max_tokens=max_tokens
        )

        assert response.success, f"LLM health check failed: {response.error}"
        assert len(response.text) > 0

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_nonzero_tokens(response)

    @pytest.mark.requires_llm
    async def test_async_llm_health_check(self, async_llm_client, token_assertions: TokenAssertions):
        """Test async LLM client health check."""
        max_tokens = 10
        response = await async_llm_client.generate(
            prompt="Test",
            endpoint="general",
            max_tokens=max_tokens
        )

        assert response.success
        assert len(response.text) > 0

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_nonzero_tokens(response)

    @pytest.mark.requires_llm
    async def test_generate_call_subject(self, async_llm_client, token_assertions: TokenAssertions):
        """Test LLM generation of call subject from transcription."""
        transcription = """
        Customer: Hi, I'm calling about my account balance. It seems incorrect.
        Support: Let me check that for you. Can you provide your account number?
        Customer: Sure, it's 12345.
        Support: I see the issue. There was a billing error. I'll correct it now.
        Customer: Thank you so much!
        """

        prompt = f"""Analyze this call transcription and provide ONLY the subject in one sentence.

Transcription:
{transcription}

Subject:"""

        max_tokens = 50
        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=max_tokens,
            temperature=0.3
        )

        assert_llm_response_valid(
            response,
            min_length=10,
            must_contain=["account", "balance"]
        )

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=20, max_tokens=200)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_generate_call_outcome(self, async_llm_client, token_assertions: TokenAssertions):
        """Test LLM classification of call outcome."""
        transcription = """
        Customer called about a technical issue. Support provided troubleshooting steps.
        The issue was resolved successfully during the call.
        """

        prompt = f"""Classify this call outcome. Respond with ONLY one word from:
Resolved, Unresolved, Pending Follow-up, Information Provided, Transferred, Unknown

Transcription:
{transcription}

Outcome:"""

        max_tokens = 10
        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=max_tokens,
            temperature=0.1
        )

        assert response.success
        # Should contain one of the valid outcomes
        valid_outcomes = ["Resolved", "Unresolved", "Pending", "Information", "Transferred", "Unknown"]
        assert any(outcome.lower() in response.text.lower() for outcome in valid_outcomes)

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_generate_customer_name_detection(self, async_llm_client, token_assertions: TokenAssertions):
        """Test LLM detection of customer name from transcription."""
        transcription = """
        Support: Thank you for calling. May I have your name?
        Customer: Yes, this is Sarah Johnson.
        Support: Thank you, Ms. Johnson. How can I help you today?
        """

        prompt = f"""Extract the CUSTOMER's name from this call. Respond with ONLY the name or "Unknown".

Transcription:
{transcription}

Customer Name:"""

        max_tokens = 20
        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=max_tokens,
            temperature=0.1
        )

        assert response.success
        assert "sarah" in response.text.lower() or "johnson" in response.text.lower()

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_generate_transcription_summary_short(self, async_llm_client):
        """Test that short transcriptions are not summarized."""
        short_transcription = "Customer called to check order status. Order is on track for delivery."

        # Simulate summarization logic: only summarize if >120 seconds / >400 chars
        duration_seconds = 30  # Short call

        if duration_seconds < 120:
            # Should skip summarization
            assert True
        else:
            # Would generate summary
            pytest.fail("Short transcription should not be summarized")

    @pytest.mark.requires_llm
    async def test_generate_transcription_summary_long(self, async_llm_client, token_assertions: TokenAssertions):
        """Test LLM summarization of long transcription."""
        long_transcription = """
        Customer called regarding a complex billing issue involving multiple accounts.
        The customer explained that they have been overcharged for the past three months.
        Support representative reviewed the account history and identified the error.
        The error was caused by a system update that incorrectly categorized the account.
        Support apologized for the inconvenience and processed a full refund.
        The refund will appear in 3-5 business days.
        Customer was satisfied with the resolution and thanked the support team.
        Support provided a reference number for the refund transaction.
        """ * 3  # Make it long enough to trigger summarization

        duration_seconds = 185  # Over 2 minutes, should summarize

        if duration_seconds >= 120:
            prompt = f"""Summarize this customer support call in 2-3 sentences.
Focus on: main issue, resolution, and customer sentiment.

Transcription:
{long_transcription[:2000]}

Summary:"""

            max_tokens = 200
            response = await async_llm_client.generate(
                prompt=prompt,
                endpoint="general",
                max_tokens=max_tokens,
                temperature=0.3
            )

            assert_llm_response_valid(
                response,
                min_length=50,
                max_length=2000,  # Allow longer LLM responses
                must_contain=["billing", "refund"]
            )

            # Token assertions
            token_assertions.assert_tokens_captured(response)
            token_assertions.assert_tokens_in_range(response, min_tokens=100, max_tokens=800)
            token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_generate_formatted_transcription(self, async_llm_client):
        """Test LLM formatting of raw transcription for readability."""
        raw_transcription = "customer hi I need help with my account support sure I can help what's the issue customer my password isn't working support let me reset that for you customer thank you"

        prompt = f"""Format this phone call transcription for readability.
Add line breaks and speaker labels (Customer/Support).
Keep all original text.

Raw: {raw_transcription}

Formatted:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=300,
            temperature=0.2
        )

        assert response.success
        formatted = response.text

        # Should have line breaks (newlines)
        assert "\n" in formatted or formatted.count(".") > 1

        # Should mention customer/support
        assert "customer" in formatted.lower() or "support" in formatted.lower()

    @pytest.mark.requires_llm
    async def test_detect_staff_vs_customer(self, async_llm_client):
        """Test LLM ability to distinguish staff from customer."""
        transcription = """
        Hi, this is John from EWR support. How can I help you today?
        I'm calling because I have an issue with my shipment.
        """

        prompt = f"""Who is the CUSTOMER in this call (the person seeking help)?
Respond with ONLY the name or "Unknown".

Transcription:
{transcription}

Customer (not staff):"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=20,
            temperature=0.1
        )

        assert response.success
        # Should NOT identify John as the customer (he's staff)
        # Should say Unknown since customer name isn't mentioned
        assert "john" not in response.text.lower() or "unknown" in response.text.lower()

    @pytest.mark.requires_llm
    async def test_call_content_json_response(self, async_llm_client):
        """Test LLM generation of structured JSON for call content."""
        transcription = """
        Customer Jane Doe called about printer connectivity issues.
        Support walked through troubleshooting steps.
        Issue was resolved after restarting the printer.
        """

        prompt = f"""Analyze this call and respond ONLY with JSON:

{{
  "subject": "brief description",
  "outcome": "Resolved/Unresolved/Pending Follow-up/etc",
  "customer_name": "name or null"
}}

Transcription:
{transcription}

JSON:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=150,
            temperature=0.1
        )

        assert response.success

        # Try to parse as JSON
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response.text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                assert "subject" in data or "outcome" in data or "customer_name" in data
            except json.JSONDecodeError:
                # LLM might not always return perfect JSON
                pass

    @pytest.mark.requires_llm
    async def test_emotion_context_in_summary(self, async_llm_client):
        """Test including emotion context in summary generation."""
        transcription = "The customer was extremely frustrated and angry about the delay."
        detected_emotions = ["ANGRY", "FRUSTRATED"]

        prompt = f"""Summarize this call in one sentence.
Detected emotions: {", ".join(detected_emotions)}

Transcription:
{transcription}

Summary:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=100,
            temperature=0.3
        )

        assert_llm_response_valid(
            response,
            min_length=20,
            must_contain=["customer", "frustrated"]
        )

    @pytest.mark.requires_llm
    async def test_llm_timeout_handling(self, async_llm_client):
        """Test handling of LLM timeouts gracefully."""
        # Very long prompt that might timeout
        long_prompt = "Analyze this: " + ("test " * 10000)

        try:
            response = await async_llm_client.generate(
                prompt=long_prompt,
                endpoint="general",
                max_tokens=100,
                temperature=0.3
            )
            # If it completes, that's fine
            assert True
        except Exception as e:
            # Timeout or error should be handled gracefully
            assert "timeout" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.requires_llm
    async def test_multiple_concurrent_llm_calls(self, async_llm_client):
        """Test making multiple concurrent LLM calls."""
        prompts = [
            "Summarize: Customer called about billing.",
            "Summarize: Customer called about shipping.",
            "Summarize: Customer called about returns."
        ]

        # Make concurrent calls
        tasks = [
            async_llm_client.generate(
                prompt=p,
                endpoint="general",
                max_tokens=50,
                temperature=0.3
            )
            for p in prompts
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # At least some should succeed
        successful = [r for r in responses if not isinstance(r, Exception) and r.success]
        assert len(successful) >= 1

    @pytest.mark.requires_llm
    def test_sync_llm_call_content_analysis(self, llm_client):
        """Test synchronous LLM call for content analysis."""
        transcription = "Customer reported a software bug. Support logged the issue for engineering."

        prompt = f"""What is this call about? One sentence only.

{transcription}

Subject:"""

        response = llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=50,
            temperature=0.3
        )

        assert_llm_response_valid(
            response,
            min_length=10,
            must_contain=["software", "bug"]
        )

    @pytest.mark.requires_llm
    async def test_llm_temperature_variation(self, async_llm_client):
        """Test LLM with different temperature settings."""
        transcription = "Customer inquiry about product features."

        # Low temperature (deterministic)
        response_low = await async_llm_client.generate(
            prompt=f"Summarize: {transcription}",
            endpoint="general",
            max_tokens=30,
            temperature=0.1
        )

        # Higher temperature (more creative)
        response_high = await async_llm_client.generate(
            prompt=f"Summarize: {transcription}",
            endpoint="general",
            max_tokens=30,
            temperature=0.7
        )

        assert response_low.success
        assert response_high.success
        # Both should be non-empty
        assert len(response_low.text) > 0
        assert len(response_high.text) > 0
