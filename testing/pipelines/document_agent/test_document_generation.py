"""
Document Generation Tests
=========================

Tests for LLM-powered document Q&A and analysis using local llama.cpp (port 8081).

Features tested:
- Document question answering
- Content summarization
- Information extraction
- Context-aware responses
- Token usage tracking

All tests use local llama.cpp endpoints only - NO external APIs.
"""

import pytest
import asyncio
from typing import Dict

from config.test_config import get_test_config
from utils import assert_llm_response_valid
from fixtures.llm_fixtures import LocalLLMClient, AsyncLocalLLMClient
from fixtures.shared_fixtures import TokenAssertions


class TestDocumentGeneration:
    """Test LLM-powered document analysis generation."""

    @pytest.mark.requires_llm
    def test_llm_health_check(self, llm_client, token_assertions: TokenAssertions):
        """Test that local LLM endpoint is accessible."""
        response = llm_client.generate(
            prompt="Test connection",
            endpoint="general",
            max_tokens=10
        )

        assert response.success
        assert len(response.text) > 0

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_nonzero_tokens(response)

    @pytest.mark.requires_llm
    async def test_document_qa_basic(self, async_llm_client, token_assertions: TokenAssertions):
        """Test basic document Q&A."""
        document_content = """
        The company's revenue increased by 25% in Q4 2024.
        Operating expenses remained stable at $5M.
        Net profit reached $2.5M, exceeding projections.
        """

        question = "What was the revenue increase percentage?"

        prompt = f"""Answer the question based on the document content.

Document:
{document_content}

Question: {question}

Answer:"""

        max_tokens = 100
        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=max_tokens,
            temperature=0.1
        )

        assert_llm_response_valid(
            response,
            min_length=5,
            must_contain=["25"]
        )

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=20, max_tokens=300)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_document_summarization(self, async_llm_client, token_assertions: TokenAssertions):
        """Test document summarization."""
        long_document = """
        The quarterly report shows strong performance across all metrics.
        Revenue grew by 30% year-over-year, driven by new product launches.
        Customer acquisition costs decreased by 15% due to improved marketing efficiency.
        The engineering team expanded by 20 new hires.
        Product development cycles shortened from 6 months to 4 months.
        Customer satisfaction scores improved to 4.7 out of 5.
        The company plans to expand into three new markets next quarter.
        """

        prompt = f"""Summarize this document in 2-3 sentences.

Document:
{long_document}

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
            max_length=1500,  # LLMs can be verbose with summaries
            must_contain=["revenue", "performance"]
        )

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=50, max_tokens=400)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_extract_key_points(self, async_llm_client, token_assertions: TokenAssertions):
        """Test extracting key points from document."""
        document = """
        Meeting agenda:
        1. Review Q4 sales performance
        2. Discuss new product roadmap
        3. Plan team restructuring
        4. Budget allocation for 2025
        5. Marketing strategy review
        """

        prompt = f"""List the top 3 key points from this document.

Document:
{document}

Key points:"""

        max_tokens = 150
        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=max_tokens,
            temperature=0.2
        )

        assert response.success
        assert len(response.text) > 20

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_answer_from_table_data(self, async_llm_client, token_assertions: TokenAssertions):
        """Test answering questions from tabular data."""
        table_content = """
        Sales by Region:
        North | 150000 | 20%
        South | 200000 | 35%
        East | 180000 | 25%
        West | 170000 | 20%
        """

        question = "Which region had the highest sales?"

        prompt = f"""Answer based on this table data.

Table:
{table_content}

Question: {question}

Answer:"""

        max_tokens = 50
        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=max_tokens,
            temperature=0.1
        )

        assert_llm_response_valid(
            response,
            min_length=5,
            must_contain=["South"]
        )

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    async def test_extract_named_entities(self, async_llm_client):
        """Test extracting named entities from document."""
        document = """
        John Smith from Acme Corporation contacted us about the Q4 report.
        The meeting is scheduled for January 15, 2025 in New York.
        Sarah Johnson will present the findings to the board.
        """

        prompt = f"""Extract person names and company names from this text.

Text:
{document}

Entities:
- People:
- Companies:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=100,
            temperature=0.1
        )

        assert response.success
        # Should contain the names
        assert "john" in response.text.lower() or "smith" in response.text.lower()

    @pytest.mark.requires_llm
    async def test_document_comparison(self, async_llm_client):
        """Test comparing two document versions."""
        doc_v1 = "Revenue: $100M, Profit: $20M, Employees: 500"
        doc_v2 = "Revenue: $120M, Profit: $25M, Employees: 550"

        prompt = f"""Compare these two versions and summarize the changes.

Version 1:
{doc_v1}

Version 2:
{doc_v2}

Changes:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=150,
            temperature=0.2
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["revenue", "increased"]
        )

    @pytest.mark.requires_llm
    async def test_classify_document_type(self, async_llm_client):
        """Test classifying document type/purpose."""
        document = """
        Product: XYZ Widget
        Price: $49.99
        Features: Durable, Lightweight, Easy to use
        Customer Reviews: 4.5 stars (2000 reviews)
        Buy Now button available
        """

        prompt = f"""Classify this document type. Choose from: Invoice, Product Page, Report, Email, Manual.

Document:
{document}

Type:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=30,
            temperature=0.1
        )

        assert response.success
        assert "product" in response.text.lower() or "page" in response.text.lower()

    @pytest.mark.requires_llm
    async def test_extract_action_items(self, async_llm_client):
        """Test extracting action items from meeting notes."""
        notes = """
        Meeting Notes - Jan 15, 2025:
        - John to finalize Q4 report by Friday
        - Sarah will schedule follow-up with clients
        - Team needs to review budget proposal
        - Marketing to launch campaign next week
        """

        prompt = f"""List all action items from these meeting notes.

Notes:
{notes}

Action Items:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=200,
            temperature=0.2
        )

        assert response.success
        # Should contain action-related words
        assert any(word in response.text.lower() for word in ["finalize", "schedule", "review", "launch"])

    @pytest.mark.requires_llm
    async def test_chunk_based_qa(self, async_llm_client):
        """Test answering from specific document chunk."""
        chunk_content = """
        Chapter 3: Security Best Practices

        Always use strong passwords with at least 12 characters.
        Enable two-factor authentication on all critical accounts.
        Regularly update software to patch security vulnerabilities.
        Use a password manager to store credentials securely.
        """

        question = "What is the recommended minimum password length?"

        prompt = f"""Answer the question based only on this content.

Content:
{chunk_content}

Question: {question}

Answer:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=50,
            temperature=0.1
        )

        assert_llm_response_valid(
            response,
            min_length=5,
            must_contain=["12"]
        )

    @pytest.mark.requires_llm
    async def test_context_aware_response(self, async_llm_client):
        """Test context-aware document responses."""
        context = """
        The Solar System consists of the Sun and all objects that orbit it.
        Earth is the third planet from the Sun.
        Mars is the fourth planet and is known as the Red Planet.
        """

        question = "Which planet is known as the Red Planet?"

        prompt = f"""Use the context to answer the question accurately.

Context:
{context}

Question: {question}

Answer:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=30,
            temperature=0.1
        )

        assert_llm_response_valid(
            response,
            min_length=3,
            must_contain=["Mars"]
        )

    @pytest.mark.requires_llm
    async def test_generate_document_title(self, async_llm_client):
        """Test generating title for untitled document."""
        content = """
        This guide covers the installation and setup of the XYZ software.
        Step 1: Download the installer from the official website.
        Step 2: Run the installer and follow the prompts.
        Step 3: Configure initial settings.
        """

        prompt = f"""Generate a concise title for this document (5 words or less).

Content:
{content}

Title:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=30,
            temperature=0.2
        )

        assert response.success
        # Extract just the title (first line) - LLMs sometimes add extra text
        title = response.text.strip().split(chr(10))[0].strip()
        assert len(title.split()) <= 15  # Reasonable length for a title

    @pytest.mark.requires_llm
    async def test_extract_dates_and_deadlines(self, async_llm_client):
        """Test extracting dates and deadlines."""
        document = """
        Project Timeline:
        - Kickoff meeting: January 10, 2025
        - Design phase deadline: February 15, 2025
        - Development complete by: March 30, 2025
        - Final delivery: April 15, 2025
        """

        prompt = f"""Extract all dates and associated events from this document.

Document:
{document}

Dates and Events:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=200,
            temperature=0.1
        )

        assert response.success
        # Should contain date-related information
        assert "january" in response.text.lower() or "february" in response.text.lower()

    @pytest.mark.requires_llm
    async def test_sentiment_analysis_document(self, async_llm_client):
        """Test analyzing sentiment of document."""
        review = """
        This product exceeded all my expectations! The quality is outstanding
        and the customer service was incredibly helpful. Highly recommend to anyone
        looking for a reliable solution.
        """

        prompt = f"""Analyze the sentiment of this text. Respond with: Positive, Negative, or Neutral.

Text:
{review}

Sentiment:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=20,
            temperature=0.1
        )

        assert response.success
        assert "positive" in response.text.lower()

    @pytest.mark.requires_llm
    async def test_multiple_qa_concurrent(self, async_llm_client):
        """Test multiple Q&A requests concurrently."""
        document = "The company revenue is $10M. The profit margin is 20%. Employees: 100."

        questions = [
            "What is the revenue?",
            "What is the profit margin?",
            "How many employees?"
        ]

        tasks = []
        for q in questions:
            prompt = f"""Answer based on: {document}\n\nQuestion: {q}\n\nAnswer:"""
            tasks.append(
                async_llm_client.generate(
                    prompt=prompt,
                    endpoint="general",
                    max_tokens=50,
                    temperature=0.1
                )
            )

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # At least some should succeed
        successful = [r for r in responses if not isinstance(r, Exception) and r.success]
        assert len(successful) >= 1

    @pytest.mark.requires_llm
    def test_sync_document_qa(self, llm_client):
        """Test synchronous document Q&A."""
        document = "Python was created by Guido van Rossum in 1991."
        question = "Who created Python?"

        prompt = f"""Answer: {document}\n\nQ: {question}\n\nA:"""

        response = llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=30,
            temperature=0.1
        )

        assert_llm_response_valid(
            response,
            min_length=5,
            must_contain=["Guido"]
        )
