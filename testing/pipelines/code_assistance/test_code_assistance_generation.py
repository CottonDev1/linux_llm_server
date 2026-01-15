"""
Code Assistance Generation Tests
=================================

Test code assistance response generation using local llama.cpp (port 8082).
Includes token usage tracking for all LLM operations.
"""

import pytest
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.shared_fixtures import TokenAssertions
from utils import assert_llm_response_valid


class TestCodeAssistanceGeneration:
    """Test code assistance generation using local LLM."""

    @pytest.mark.requires_llm
    def test_generate_code_explanation(self, llm_client: LocalLLMClient, pipeline_config,
                                        token_assertions: TokenAssertions):
        """Test generating a code explanation."""
        context = """
        Method: SaveBale
        Class: BaleService
        Purpose: Saves a bale record to the database with validation
        """

        prompt = f"""Given this code context:

{context}

Explain how this method works:"""

        max_tokens = 300
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.2,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["bale"],
        )

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=30, max_tokens=500)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_answer_how_to_question(self, llm_client: LocalLLMClient, pipeline_config,
                                     token_assertions: TokenAssertions):
        """Test answering a how-to question."""
        context = """
        Available methods:
        - ValidateBale: Validates bale data
        - SaveBale: Saves bale to database
        - GetBale: Retrieves bale by ID
        """

        prompt = f"""Context:
{context}

Question: How do I save a new bale?

Answer:"""

        max_tokens = 200
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=20)
        text_lower = response.text.lower()
        assert any(word in text_lower for word in ["validate", "save"])

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_generate_with_call_chain(self, llm_client: LocalLLMClient, pipeline_config,
                                       token_assertions: TokenAssertions):
        """Test generating explanation with call chain context."""
        context = """
        Call chain:
        1. btnSaveBale_Click (UI Event)
        2. ValidateBale (Validation)
        3. SaveBale (Database INSERT into Bales table)
        4. RefreshGrid (UI Update)
        """

        prompt = f"""Explain this execution flow:

{context}

Explanation:"""

        max_tokens = 300
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.1,
        )

        assert_llm_response_valid(response, min_length=30)

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=30, max_tokens=500)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_summarize_code_context(self, llm_client: LocalLLMClient, pipeline_config,
                                     token_assertions: TokenAssertions):
        """Test summarizing retrieved code context."""
        context = """
        Method: ProcessOrder
        - Validates order data
        - Calculates order total
        - Saves to Orders table
        - Updates inventory
        - Sends confirmation email
        """

        prompt = f"""Summarize what this method does:

{context}

Summary:"""

        max_tokens = 150
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        assert_llm_response_valid(response, min_length=20, must_contain=["order"])

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=20, max_tokens=300)
        token_assertions.assert_max_tokens_respected(response, max_tokens)
