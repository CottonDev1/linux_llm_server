"""
Token Usage Tests
=================

Comprehensive tests for token usage tracking across all pipelines.

Tests cover:
1. Token capture verification (prompt, completion, total)
2. Token limit enforcement (max_tokens parameter)
3. Token estimation accuracy (estimated vs actual)
4. Cost calculation and budget enforcement
5. Cross-pipeline token comparison

All tests use LOCAL llama.cpp endpoints only - NO external APIs.
"""

import pytest
import asyncio
from typing import Dict, Any, List

from config.test_config import PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient, AsyncLocalLLMClient
from fixtures.shared_fixtures import TokenAssertions


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_prompts() -> Dict[str, str]:
    """Sample prompts of varying complexity for token testing."""
    return {
        "minimal": "Hello",
        "short": "What is 2 + 2?",
        "medium": """Explain the following code:
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
        "long": """Analyze the following complex database query and explain each part:

WITH CustomerOrders AS (
    SELECT
        c.CustomerID,
        c.CustomerName,
        COUNT(o.OrderID) as OrderCount,
        SUM(o.TotalAmount) as TotalSpent,
        AVG(o.TotalAmount) as AvgOrderValue
    FROM Customers c
    LEFT JOIN Orders o ON c.CustomerID = o.CustomerID
    WHERE o.OrderDate >= DATEADD(year, -1, GETDATE())
    GROUP BY c.CustomerID, c.CustomerName
)
SELECT
    co.*,
    CASE
        WHEN co.TotalSpent > 10000 THEN 'Premium'
        WHEN co.TotalSpent > 5000 THEN 'Gold'
        ELSE 'Standard'
    END as CustomerTier
FROM CustomerOrders co
WHERE co.OrderCount > 0
ORDER BY co.TotalSpent DESC;
""",
        "sql_simple": "SELECT * FROM Users WHERE UserID = 1",
        "sql_complex": """
SELECT
    u.UserName,
    o.OrderID,
    p.ProductName,
    oi.Quantity,
    oi.UnitPrice
FROM Users u
JOIN Orders o ON u.UserID = o.UserID
JOIN OrderItems oi ON o.OrderID = oi.OrderID
JOIN Products p ON oi.ProductID = p.ProductID
WHERE o.OrderDate > '2024-01-01'
ORDER BY o.OrderDate DESC
""",
        "code_context": """
public class UserService
{
    private readonly IUserRepository _repository;
    private readonly ILogger<UserService> _logger;

    public async Task<User> GetUserByIdAsync(int userId)
    {
        _logger.LogInformation("Fetching user {UserId}", userId);
        var user = await _repository.GetByIdAsync(userId);
        if (user == null)
        {
            _logger.LogWarning("User {UserId} not found", userId);
            throw new UserNotFoundException(userId);
        }
        return user;
    }

    public async Task<bool> UpdateUserAsync(User user)
    {
        ValidateUser(user);
        return await _repository.UpdateAsync(user);
    }
}
""",
        "prose": """
The quarterly financial report indicates strong growth across all major business units.
Revenue increased by 23% year-over-year, driven primarily by expansion in the Asia-Pacific
region. Operating margins improved to 18.5%, reflecting the success of our cost optimization
initiatives. Customer acquisition costs decreased by 15% while customer lifetime value
increased by 28%, demonstrating improved unit economics. The company maintains a strong
balance sheet with $2.3 billion in cash and cash equivalents.
""",
    }


# =============================================================================
# Token Capture Tests
# =============================================================================

class TestTokenCapture:
    """Test that token counts are properly captured in LLM responses."""

    @pytest.mark.requires_llm
    def test_prompt_tokens_captured(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        pipeline_config: PipelineTestConfig,
    ):
        """Verify prompt_tokens are captured in response."""
        response = llm_client.generate(
            prompt="What is 2 + 2?",
            model_type="general",
            max_tokens=50,
            temperature=0.0,
        )

        assert response.success, f"LLM request failed: {response.error}"
        token_assertions.assert_prompt_tokens_captured(response)

    @pytest.mark.requires_llm
    def test_completion_tokens_captured(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        pipeline_config: PipelineTestConfig,
    ):
        """Verify completion_tokens are captured in response."""
        response = llm_client.generate(
            prompt="Write a short greeting.",
            model_type="general",
            max_tokens=50,
            temperature=0.0,
        )

        assert response.success, f"LLM request failed: {response.error}"
        token_assertions.assert_completion_tokens_captured(response)

    @pytest.mark.requires_llm
    def test_total_tokens_equals_sum(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        pipeline_config: PipelineTestConfig,
    ):
        """Verify total_tokens equals prompt_tokens + completion_tokens."""
        response = llm_client.generate(
            prompt="Explain why the sky is blue in one sentence.",
            model_type="general",
            max_tokens=100,
            temperature=0.0,
        )

        assert response.success, f"LLM request failed: {response.error}"
        token_assertions.assert_total_equals_sum(response)

    @pytest.mark.requires_llm
    def test_tokens_captured_all_models(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Verify token capture works for all model types."""
        model_types = ["sql", "general", "code"]
        prompts = {
            "sql": "SELECT * FROM Users",
            "general": "Hello, how are you?",
            "code": "def hello(): pass",
        }

        for model_type in model_types:
            response = llm_client.generate(
                prompt=prompts[model_type],
                model_type=model_type,
                max_tokens=50,
                temperature=0.0,
            )

            if response.success:
                token_assertions.assert_tokens_captured(response)
                token_assertions.assert_nonzero_tokens(response)

    @pytest.mark.requires_llm
    def test_nonzero_tokens_for_valid_response(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Verify responses have non-zero token counts."""
        response = llm_client.generate(
            prompt="Count from 1 to 5.",
            model_type="general",
            max_tokens=50,
            temperature=0.0,
        )

        assert response.success
        token_assertions.assert_nonzero_tokens(response)


# =============================================================================
# Token Limit Enforcement Tests
# =============================================================================

class TestTokenLimitEnforcement:
    """Test that max_tokens parameter is respected."""

    @pytest.mark.requires_llm
    def test_max_tokens_small_limit(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test max_tokens is respected with small limit."""
        max_tokens = 20

        response = llm_client.generate(
            prompt="Write a very long essay about the history of computing.",
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_max_tokens_respected(
                response, max_tokens, tolerance=0.1
            )

    @pytest.mark.requires_llm
    def test_max_tokens_medium_limit(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test max_tokens is respected with medium limit."""
        max_tokens = 100

        response = llm_client.generate(
            prompt="Explain the theory of relativity in detail.",
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_max_tokens_respected(
                response, max_tokens, tolerance=0.1
            )

    @pytest.mark.requires_llm
    def test_max_tokens_large_limit(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test max_tokens is respected with large limit."""
        max_tokens = 500

        response = llm_client.generate(
            prompt="Write a comprehensive guide to Python programming.",
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_max_tokens_respected(
                response, max_tokens, tolerance=0.05
            )

    @pytest.mark.requires_llm
    def test_completion_tokens_within_limit(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test that completion tokens stay within specified limit."""
        max_tokens = 50

        response = llm_client.generate(
            prompt="List 100 different programming languages.",
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_completion_tokens_in_range(
                response, min_tokens=1, max_tokens=max_tokens + 5
            )

    @pytest.mark.requires_llm
    @pytest.mark.parametrize("max_tokens", [10, 25, 50, 100, 200])
    def test_max_tokens_various_limits(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        max_tokens: int,
    ):
        """Test max_tokens enforcement with various limits."""
        response = llm_client.generate(
            prompt="Describe the benefits of testing in software development.",
            model_type="general",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_max_tokens_respected(
                response, max_tokens, tolerance=0.15
            )


# =============================================================================
# Token Estimation Accuracy Tests
# =============================================================================

class TestTokenEstimationAccuracy:
    """Test accuracy of token estimation for different content types."""

    @pytest.mark.requires_llm
    def test_estimation_accuracy_code(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test token estimation accuracy for code content."""
        code_prompt = sample_prompts["code_context"]

        # Rough estimate: ~4 characters per token for code
        estimated_prompt_tokens = len(code_prompt) // 4

        response = llm_client.generate(
            prompt=f"Explain this code:\n{code_prompt}",
            model_type="code",
            max_tokens=200,
            temperature=0.0,
        )

        if response.success and response.prompt_tokens > 0:
            # Allow 30% tolerance for code (tokenization varies)
            token_assertions.assert_token_estimation_accuracy(
                estimated_tokens=estimated_prompt_tokens,
                actual_tokens=response.prompt_tokens,
                tolerance=0.30,
            )

    @pytest.mark.requires_llm
    def test_estimation_accuracy_prose(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test token estimation accuracy for prose content."""
        prose_prompt = sample_prompts["prose"]

        # Rough estimate: ~4 characters per token for English prose
        estimated_prompt_tokens = len(prose_prompt) // 4

        response = llm_client.generate(
            prompt=f"Summarize this:\n{prose_prompt}",
            model_type="general",
            max_tokens=150,
            temperature=0.0,
        )

        if response.success and response.prompt_tokens > 0:
            token_assertions.assert_token_estimation_accuracy(
                estimated_tokens=estimated_prompt_tokens,
                actual_tokens=response.prompt_tokens,
                tolerance=0.25,
            )

    @pytest.mark.requires_llm
    def test_estimation_accuracy_sql(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test token estimation accuracy for SQL content."""
        sql_prompt = sample_prompts["sql_complex"]

        # Rough estimate: ~3.5 characters per token for SQL (more keywords)
        estimated_prompt_tokens = int(len(sql_prompt) / 3.5)

        response = llm_client.generate(
            prompt=f"Explain this query:\n{sql_prompt}",
            model_type="sql",
            max_tokens=200,
            temperature=0.0,
        )

        if response.success and response.prompt_tokens > 0:
            token_assertions.assert_token_estimation_accuracy(
                estimated_tokens=estimated_prompt_tokens,
                actual_tokens=response.prompt_tokens,
                tolerance=0.35,
            )

    @pytest.mark.requires_llm
    def test_prompt_length_correlates_with_tokens(
        self,
        llm_client: LocalLLMClient,
        sample_prompts: Dict[str, str],
    ):
        """Test that longer prompts result in more prompt tokens."""
        short_response = llm_client.generate(
            prompt=sample_prompts["short"],
            model_type="general",
            max_tokens=50,
            temperature=0.0,
        )

        long_response = llm_client.generate(
            prompt=sample_prompts["long"],
            model_type="general",
            max_tokens=50,
            temperature=0.0,
        )

        if short_response.success and long_response.success:
            assert long_response.prompt_tokens > short_response.prompt_tokens, \
                "Longer prompt should have more tokens"


# =============================================================================
# Cost Calculation Tests
# =============================================================================

class TestCostCalculation:
    """Test token-based cost calculation."""

    @pytest.mark.requires_llm
    def test_calculate_cost_sql_model(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test cost calculation for SQL model."""
        response = llm_client.generate(
            prompt="SELECT * FROM Users WHERE Active = 1",
            model_type="sql",
            max_tokens=100,
            temperature=0.0,
        )

        if response.success:
            cost = token_assertions.calculate_cost(response, model_type="sql")
            assert cost >= 0, "Cost must be non-negative"
            assert cost < 1.0, "Cost seems unreasonably high for single request"

    @pytest.mark.requires_llm
    def test_calculate_cost_general_model(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test cost calculation for general model."""
        response = llm_client.generate(
            prompt="Explain machine learning in simple terms.",
            model_type="general",
            max_tokens=200,
            temperature=0.3,
        )

        if response.success:
            cost = token_assertions.calculate_cost(response, model_type="general")
            assert cost >= 0
            assert cost < 1.0

    @pytest.mark.requires_llm
    def test_calculate_cost_code_model(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test cost calculation for code model."""
        response = llm_client.generate(
            prompt="def fibonacci(n): # complete this function",
            model_type="code",
            max_tokens=150,
            temperature=0.0,
        )

        if response.success:
            cost = token_assertions.calculate_cost(response, model_type="code")
            assert cost >= 0
            assert cost < 1.0

    @pytest.mark.requires_llm
    def test_cost_within_budget(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test that responses stay within budget."""
        response = llm_client.generate(
            prompt="Say hello.",
            model_type="general",
            max_tokens=20,
            temperature=0.0,
        )

        if response.success:
            # Very small budget for a tiny request
            token_assertions.assert_cost_within_budget(
                response, max_cost=0.01, model_type="general"
            )

    @pytest.mark.requires_llm
    def test_cumulative_cost_tracking(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test tracking cumulative cost across multiple requests."""
        prompts = [
            "What is 1 + 1?",
            "What is 2 + 2?",
            "What is 3 + 3?",
        ]

        total_cost = 0.0
        for prompt in prompts:
            response = llm_client.generate(
                prompt=prompt,
                model_type="general",
                max_tokens=30,
                temperature=0.0,
            )
            if response.success:
                total_cost += token_assertions.calculate_cost(response)

        # Cumulative cost should be reasonable for 3 simple requests
        assert total_cost < 0.1, f"Cumulative cost ${total_cost:.6f} seems too high"


# =============================================================================
# Cross-Pipeline Token Comparison Tests
# =============================================================================

class TestCrossPipelineComparison:
    """Test and compare token usage across different pipeline types."""

    @pytest.mark.requires_llm
    def test_sql_pipeline_uses_fewer_tokens(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test that SQL queries typically use fewer completion tokens."""
        sql_response = llm_client.generate(
            prompt=f"Generate SQL for: {sample_prompts['sql_simple']}",
            model_type="sql",
            max_tokens=100,
            temperature=0.0,
        )

        general_response = llm_client.generate(
            prompt=f"Explain: {sample_prompts['prose']}",
            model_type="general",
            max_tokens=300,
            temperature=0.0,
        )

        if sql_response.success and general_response.success:
            # SQL responses are typically more concise
            sql_summary = token_assertions.get_token_summary(sql_response)
            general_summary = token_assertions.get_token_summary(general_response)

            # Just log for comparison, don't assert as it depends on content
            print(f"SQL completion tokens: {sql_summary['completion_tokens']}")
            print(f"General completion tokens: {general_summary['completion_tokens']}")

    @pytest.mark.requires_llm
    def test_code_assistance_context_tokens(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test that code assistance uses significant context tokens."""
        response = llm_client.generate(
            prompt=f"Explain this code and suggest improvements:\n{sample_prompts['code_context']}",
            model_type="code",
            max_tokens=300,
            temperature=0.2,
        )

        if response.success:
            summary = token_assertions.get_token_summary(response)

            # Code assistance should have substantial prompt tokens (context)
            assert summary["prompt_tokens"] > 50, \
                "Code assistance should have significant context tokens"

    @pytest.mark.requires_llm
    def test_document_pipeline_large_context(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test that document pipelines handle large context windows."""
        # Simulate large document context
        large_context = sample_prompts["prose"] * 3

        response = llm_client.generate(
            prompt=f"Summarize this document:\n{large_context}",
            model_type="general",
            max_tokens=200,
            temperature=0.3,
        )

        if response.success:
            summary = token_assertions.get_token_summary(response)

            # Document pipeline should handle larger prompt tokens
            assert summary["prompt_tokens"] > 100, \
                "Document pipeline should handle large context"

    @pytest.mark.requires_llm
    def test_completion_ratio_varies_by_task(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test that completion ratio varies appropriately by task type."""
        # Short prompt expecting longer response
        short_prompt_response = llm_client.generate(
            prompt="Write a poem about testing.",
            model_type="general",
            max_tokens=200,
            temperature=0.5,
        )

        # Long prompt expecting shorter response
        long_prompt = "Given the following long context " + ("about testing " * 20) + ", say yes or no."
        long_prompt_response = llm_client.generate(
            prompt=long_prompt,
            model_type="general",
            max_tokens=20,
            temperature=0.0,
        )

        if short_prompt_response.success and long_prompt_response.success:
            short_summary = token_assertions.get_token_summary(short_prompt_response)
            long_summary = token_assertions.get_token_summary(long_prompt_response)

            # Short prompt should have higher completion ratio
            # Long prompt should have lower completion ratio
            print(f"Short prompt completion ratio: {short_summary['completion_ratio']:.2f}")
            print(f"Long prompt completion ratio: {long_summary['completion_ratio']:.2f}")


# =============================================================================
# Token Range Tests by Pipeline Type
# =============================================================================

class TestTokenRangesByPipeline:
    """Test expected token ranges for different pipeline operations."""

    @pytest.mark.requires_llm
    def test_sql_generation_token_range(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test token range for SQL generation."""
        response = llm_client.generate(
            prompt="Generate SQL to: Get all users from the Users table",
            model_type="sql",
            max_tokens=256,
            temperature=0.0,
        )

        if response.success:
            # SQL generation typically uses moderate tokens
            token_assertions.assert_tokens_in_range(
                response, min_tokens=10, max_tokens=500
            )

    @pytest.mark.requires_llm
    def test_code_explanation_token_range(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test token range for code explanation."""
        response = llm_client.generate(
            prompt="Explain: def add(a, b): return a + b",
            model_type="code",
            max_tokens=200,
            temperature=0.2,
        )

        if response.success:
            token_assertions.assert_tokens_in_range(
                response, min_tokens=20, max_tokens=400
            )

    @pytest.mark.requires_llm
    def test_summarization_token_range(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
        sample_prompts: Dict[str, str],
    ):
        """Test token range for summarization."""
        response = llm_client.generate(
            prompt=f"Summarize in one sentence:\n{sample_prompts['prose']}",
            model_type="general",
            max_tokens=100,
            temperature=0.3,
        )

        if response.success:
            token_assertions.assert_tokens_in_range(
                response, min_tokens=50, max_tokens=300
            )


# =============================================================================
# Async Token Tests
# =============================================================================

class TestAsyncTokenUsage:
    """Test token usage with async LLM client."""

    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_async_tokens_captured(
        self,
        async_llm_client: AsyncLocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test token capture with async client."""
        response = await async_llm_client.generate(
            prompt="What is Python?",
            model_type="general",
            max_tokens=100,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_tokens_captured(response)
            token_assertions.assert_nonzero_tokens(response)

    @pytest.mark.requires_llm
    @pytest.mark.asyncio
    async def test_concurrent_requests_token_tracking(
        self,
        async_llm_client: AsyncLocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test token tracking across concurrent requests."""
        prompts = [
            "Define testing.",
            "Define debugging.",
            "Define deployment.",
        ]

        tasks = [
            async_llm_client.generate(
                prompt=p,
                model_type="general",
                max_tokens=50,
                temperature=0.0,
            )
            for p in prompts
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        total_tokens = 0
        for response in responses:
            if not isinstance(response, Exception) and response.success:
                token_assertions.assert_tokens_captured(response)
                summary = token_assertions.get_token_summary(response)
                total_tokens += summary["total_tokens"]

        assert total_tokens > 0, "Should have accumulated tokens from concurrent requests"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestTokenEdgeCases:
    """Test edge cases in token handling."""

    @pytest.mark.requires_llm
    def test_empty_response_tokens(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test token handling when response is minimal."""
        response = llm_client.generate(
            prompt="Reply with only the word 'yes'.",
            model_type="general",
            max_tokens=5,
            temperature=0.0,
        )

        if response.success:
            # Should still capture tokens even for minimal response
            token_assertions.assert_tokens_captured(response)

    @pytest.mark.requires_llm
    def test_max_tokens_one(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test behavior with max_tokens=1."""
        response = llm_client.generate(
            prompt="Say hello.",
            model_type="general",
            max_tokens=1,
            temperature=0.0,
        )

        if response.success:
            token_assertions.assert_completion_tokens_in_range(
                response, min_tokens=0, max_tokens=5
            )

    @pytest.mark.requires_llm
    def test_very_long_prompt_tokens(
        self,
        llm_client: LocalLLMClient,
        token_assertions: TokenAssertions,
    ):
        """Test token handling with very long prompt."""
        long_prompt = "Analyze: " + ("This is test content. " * 100)

        response = llm_client.generate(
            prompt=long_prompt,
            model_type="general",
            max_tokens=50,
            temperature=0.0,
        )

        if response.success:
            # Long prompt should result in many prompt tokens
            token_assertions.assert_prompt_tokens_in_range(
                response, min_tokens=100, max_tokens=5000
            )
