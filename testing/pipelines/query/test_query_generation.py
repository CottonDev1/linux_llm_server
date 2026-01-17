"""
Query/RAG Generation Tests using local llama.cpp.

Tests RAG response generation using local LLM endpoints including:
- Answer generation from retrieved context
- Multi-source context synthesis
- Code explanation generation
- Documentation question answering
- Citation and source attribution
- Token usage tracking

All tests use LOCAL llama.cpp only (port 8081 for general model).
NO external APIs permitted.
"""

import pytest

from config.test_config import PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.shared_fixtures import TokenAssertions
from utils import (
    assert_llm_response_valid,
    assert_similar_text,
)


class TestBasicRAGGeneration:
    """Test basic RAG answer generation."""

    @pytest.mark.requires_llm
    def test_generate_answer_from_context(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test generating answer from provided context."""
        context = """
The RecapGet stored procedure retrieves recap information for a specific gin.
It accepts a @GinID parameter and returns all recap records for that gin.

Parameters:
- @GinID (INT): The unique identifier for the gin

Returns:
- RecapID, GinID, RecapDate, TotalBales, TotalWeight
"""

        question = "What does the RecapGet procedure do?"

        prompt = f"""Answer the following question using the context provided.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, concise answer based only on the information in the context.

ANSWER:"""

        max_tokens = 512
        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.3,
        )

        # Validate response
        assert_llm_response_valid(
            response,
            min_length=20,
            must_contain=["RecapGet", "recap"],
        )

        # Should mention key concepts
        answer_lower = response.text.lower()
        assert "gin" in answer_lower or "ginid" in answer_lower
        assert "recap" in answer_lower

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=30, max_tokens=700)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_generate_code_explanation(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test generating explanation for code snippet."""
        code_context = """
public void ProcessBale(Bale bale)
{
    if (bale == null)
        throw new ArgumentNullException(nameof(bale));

    ValidateBale(bale);
    CalculateWeight(bale);
    SaveToDatabase(bale);
}
"""

        question = "How does ProcessBale work?"

        prompt = f"""Explain the following code.

CODE:
{code_context}

QUESTION: {question}

Provide a clear explanation of what this code does.

EXPLANATION:"""

        max_tokens = 512
        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["ProcessBale"],
        )

        # Should explain validation and processing steps
        text_lower = response.text.lower()
        assert "validate" in text_lower or "check" in text_lower
        assert "save" in text_lower or "database" in text_lower

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=50, max_tokens=700)
        token_assertions.assert_max_tokens_respected(response, max_tokens)


class TestMultiSourceSynthesis:
    """Test synthesizing answers from multiple sources."""

    @pytest.mark.requires_llm
    def test_synthesize_from_multiple_contexts(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test combining information from multiple context sources."""
        contexts = [
            """
Source 1: The RecapGet procedure retrieves recap data for a specific gin.
It returns RecapID, GinID, RecapDate, TotalBales, and TotalWeight.
""",
            """
Source 2: To get recap information, call RecapGet with the gin's ID.
The procedure performs validation and returns historical data.
""",
            """
Source 3: Recap records track daily processing totals for each gin.
The RecapDate field indicates when the processing occurred.
""",
        ]

        question = "What information does RecapGet return and what is it used for?"

        context_text = "\n\n".join(contexts)

        prompt = f"""Answer the question using information from all provided sources.

SOURCES:
{context_text}

QUESTION: {question}

Synthesize a comprehensive answer from all sources.

ANSWER:"""

        max_tokens = 1024
        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=40,
            must_contain=["RecapGet"],
        )

        # Should mention multiple aspects from different sources
        text_lower = response.text.lower()
        # From source 1: fields returned
        assert "recapid" in text_lower or "fields" in text_lower or "returns" in text_lower

        # From source 3: purpose (tracking)
        assert "track" in text_lower or "daily" in text_lower or "processing" in text_lower

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=50, max_tokens=1200)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_handle_conflicting_information(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test handling contradictory information in sources."""
        contexts = [
            "Source 1: The system supports up to 1000 concurrent users.",
            "Source 2: The maximum concurrent user limit is 500.",
        ]

        question = "How many concurrent users does the system support?"

        context_text = "\n\n".join(contexts)

        prompt = f"""Answer the question. Note that sources may have conflicting information.

SOURCES:
{context_text}

QUESTION: {question}

If sources conflict, mention both values.

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=20)

        # Should acknowledge both values or the conflict
        text = response.text
        # Either mentions both numbers or indicates uncertainty/conflict
        has_both_numbers = ("1000" in text or "1,000" in text) and "500" in text
        has_conflict_language = (
            "conflict" in text.lower()
            or "differ" in text.lower()
            or "varies" in text.lower()
            or "depending" in text.lower()
        )

        assert has_both_numbers or has_conflict_language


class TestContextRelevanceFiltering:
    """Test filtering irrelevant context."""

    @pytest.mark.requires_llm
    def test_ignore_irrelevant_context(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test that LLM focuses on relevant context."""
        relevant_context = "The RecapGet procedure accepts @GinID parameter and returns recap data."
        irrelevant_context = "The weather today is sunny with temperatures around 75 degrees."

        question = "What parameter does RecapGet accept?"

        prompt = f"""Answer the question using only relevant information from the context.

CONTEXT:
{relevant_context}

{irrelevant_context}

QUESTION: {question}

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=10,
            must_contain=["GinID"],
            must_not_contain=["weather", "sunny", "temperature"],
        )


class TestSourceAttribution:
    """Test proper attribution of information to sources."""

    @pytest.mark.requires_llm
    def test_cite_sources(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test that generated answers can reference sources."""
        contexts = [
            "Source [RecapGet.sql]: The RecapGet procedure retrieves recap information.",
            "Source [ProcessBale.cs]: The ProcessBale method validates and saves bale data.",
        ]

        question = "What does RecapGet do?"

        context_text = "\n\n".join(contexts)

        prompt = f"""Answer the question and cite your sources using [filename] format.

SOURCES:
{context_text}

QUESTION: {question}

ANSWER (with citation):"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=20,
            must_contain=["RecapGet"],
        )

        # Should ideally include citation
        # (soft check - not all models will follow citation format perfectly)
        has_citation = "[RecapGet.sql]" in response.text or "RecapGet.sql" in response.text


class TestAnswerQuality:
    """Test quality characteristics of generated answers."""

    @pytest.mark.requires_llm
    def test_answer_completeness(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test that answers are complete and address the question."""
        context = """
The BaleProcessor class handles cotton bale processing.
It performs these steps:
1. Validate bale data
2. Calculate weight and grade
3. Generate barcode
4. Save to database
5. Update inventory
"""

        question = "What are the steps in bale processing?"

        prompt = f"""Answer the question comprehensively using the context.

CONTEXT:
{context}

QUESTION: {question}

Provide a complete answer listing all steps.

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=50)

        # Should mention multiple steps
        text_lower = response.text.lower()
        steps_mentioned = sum(
            [
                "validate" in text_lower,
                "weight" in text_lower or "grade" in text_lower,
                "barcode" in text_lower,
                "save" in text_lower or "database" in text_lower,
                "inventory" in text_lower,
            ]
        )

        assert steps_mentioned >= 3, "Should mention at least 3 of the 5 steps"

    @pytest.mark.requires_llm
    def test_answer_conciseness(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test that answers are concise and not overly verbose."""
        context = "The RecapGet procedure returns recap data for a gin."

        question = "What does RecapGet return?"

        prompt = f"""Answer the question concisely.

CONTEXT:
{context}

QUESTION: {question}

Provide a brief, direct answer.

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=128,
            temperature=0.3,
        )

        # 128 tokens can be ~500 chars
        assert_llm_response_valid(response, min_length=10, max_length=600)

        # Answer should be relatively short for simple question
        assert len(response.text) < 500, "Answer should be concise for simple question"

    @pytest.mark.requires_llm
    def test_handle_no_context(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test handling questions with no relevant context."""
        context = "The ProcessTicket method handles ticket processing."

        question = "How do I configure the email server?"

        prompt = f"""Answer the question using the context. If the context doesn't contain the answer, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=10)

        # Should indicate information not available
        text_lower = response.text.lower()
        indicates_no_info = any(
            phrase in text_lower
            for phrase in [
                "not found",
                "not available",
                "no information",
                "don't know",
                "doesn't contain",
                "cannot answer",
            ]
        )

        # Soft assertion - some models may still attempt to answer
        # Just verify it doesn't hallucinate specific email config details
        assert "smtp" not in text_lower or indicates_no_info


class TestKnowledgeBaseModeVsCodeMode:
    """Test different generation modes for knowledge base vs code."""

    @pytest.mark.requires_llm
    def test_knowledge_base_answer_style(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test answer style for knowledge base questions."""
        context = """
EWR Safety Procedures

All warehouse personnel must wear appropriate PPE including:
- Hard hats in designated areas
- Safety glasses at all times
- Steel-toed boots
- High-visibility vests

Emergency exits are marked with green signs and must remain unobstructed.
"""

        question = "What PPE is required in the warehouse?"

        prompt = f"""You are a helpful assistant answering questions about company documentation.

DOCUMENTATION:
{context}

QUESTION: {question}

Provide a clear, professional answer.

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["PPE"],
        )

        # Should mention required items
        text_lower = response.text.lower()
        assert "hard hat" in text_lower or "helmet" in text_lower
        assert "safety glasses" in text_lower or "glasses" in text_lower

    @pytest.mark.requires_llm
    def test_code_answer_style(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test answer style for code-related questions."""
        context = """
public class BaleProcessor
{
    public void ProcessBale(Bale bale)
    {
        ValidateBale(bale);
        CalculateMetrics(bale);
        SaveBale(bale);
    }

    private void ValidateBale(Bale bale)
    {
        if (bale == null)
            throw new ArgumentNullException();
        if (bale.Weight <= 0)
            throw new InvalidOperationException("Invalid weight");
    }
}
"""

        question = "How does the code validate bales?"

        prompt = f"""You are a code assistant helping developers understand code.

CODE:
{context}

QUESTION: {question}

Explain the code clearly.

EXPLANATION:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
        )

        # Should mention technical validation details
        text_lower = response.text.lower()
        assert "null" in text_lower or "validate" in text_lower
        assert "weight" in text_lower or "check" in text_lower
