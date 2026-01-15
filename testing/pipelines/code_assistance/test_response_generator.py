"""
Response Generator Service Tests
================================

Test the ResponseGenerator service including:
- Code explanation generation
- Completion suggestions
- Refactoring suggestions
- Error explanation with source references
- Token tracking
- Streaming generation
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from fixtures.llm_fixtures import LocalLLMClient, LLMResponse
from utils import assert_llm_response_valid


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_prompt() -> str:
    """Create a sample prompt for testing."""
    return """You are a helpful code assistant that explains C# code.

CODE CONTEXT:
Method: BaleService.SaveBale
File: /src/Services/BaleService.cs:42
Signature: public void SaveBale(Bale bale)
Summary: Saves a bale record to the database

USER QUESTION: How do I save a bale in the system?

ANSWER:"""


@pytest.fixture
def code_explanation_prompt() -> str:
    """Prompt for code explanation."""
    return """Explain what this code does:

```csharp
public void SaveBale(Bale bale)
{
    if (bale == null) throw new ArgumentNullException(nameof(bale));
    _validator.Validate(bale);
    _repository.Insert(bale);
    _logger.LogInfo($"Saved bale {bale.Id}");
}
```

Explanation:"""


@pytest.fixture
def refactoring_prompt() -> str:
    """Prompt for refactoring suggestions."""
    return """Suggest improvements for this code:

```csharp
public List<Bale> GetBales(string warehouse, DateTime date)
{
    var query = "SELECT * FROM Bales WHERE Warehouse = '" + warehouse + "' AND Date = '" + date + "'";
    return _db.Query(query);
}
```

Suggestions:"""


@pytest.fixture
def error_explanation_prompt() -> str:
    """Prompt for error explanation."""
    return """Explain this error and how to fix it:

Error: NullReferenceException in BaleService.SaveBale at line 42
Stack trace:
  at BaleService.SaveBale(Bale bale) in BaleService.cs:line 42
  at BaleEntryForm.btnSave_Click(Object sender, EventArgs e) in BaleEntryForm.cs:line 150

Context: The error occurs when the user clicks the Save button without entering bale data.

Explanation:"""


# =============================================================================
# Initialization Tests
# =============================================================================

class TestResponseGeneratorInit:
    """Test ResponseGenerator initialization."""

    @pytest.mark.asyncio
    async def test_generator_lazy_initialization(self, pipeline_config):
        """Test that generator initializes lazily."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        generator = ResponseGenerator()

        assert generator._llm_service is None
        assert generator._initialized is False

    @pytest.mark.asyncio
    async def test_generator_double_init_safe(self, pipeline_config):
        """Test that double initialization is safe."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            generator = ResponseGenerator()
            await generator.initialize()
            await generator.initialize()  # Should be no-op

            assert generator._initialized is True

    @pytest.mark.asyncio
    async def test_generator_uses_traced_when_available(self, pipeline_config):
        """Test that generator uses TracedLLMClient when available."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            generator = ResponseGenerator()
            await generator.initialize()

            assert generator._use_traced is True

    @pytest.mark.asyncio
    async def test_generator_falls_back_to_legacy(self, pipeline_config):
        """Test that generator falls back to legacy LLM service."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', False):
            with patch('services.llm_service.get_llm_service') as mock_get:
                mock_service = AsyncMock()
                mock_get.return_value = mock_service

                generator = ResponseGenerator()
                generator._use_traced = False
                await generator.initialize()

                assert generator._use_traced is False


# =============================================================================
# Generate Tests (Non-Streaming)
# =============================================================================

class TestResponseGeneratorGenerate:
    """Test non-streaming generation."""

    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self, pipeline_config, sample_prompt):
        """Test that generate returns (answer, token_usage, time_ms)."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator
        from code_assistance_pipeline.models.query_models import TokenUsage

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            with patch('code_assistance_pipeline.services.response_generator.generate_text') as mock_gen:
                mock_response = MagicMock()
                mock_response.success = True
                mock_response.text = "To save a bale, call BaleService.SaveBale(bale)."
                mock_response.prompt_tokens = 100
                mock_response.response_tokens = 50
                mock_response.total_tokens = 150

                mock_gen.return_value = mock_response

                generator = ResponseGenerator()
                await generator.initialize()

                answer, token_usage, gen_time = await generator.generate(sample_prompt)

                assert isinstance(answer, str)
                assert len(answer) > 0
                assert isinstance(token_usage, TokenUsage)
                assert isinstance(gen_time, int)
                assert gen_time >= 0

    @pytest.mark.asyncio
    async def test_generate_captures_token_usage(self, pipeline_config, sample_prompt):
        """Test that token usage is captured correctly."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            with patch('code_assistance_pipeline.services.response_generator.generate_text') as mock_gen:
                mock_response = MagicMock()
                mock_response.success = True
                mock_response.text = "Answer text"
                mock_response.prompt_tokens = 120
                mock_response.response_tokens = 80
                mock_response.total_tokens = 200

                mock_gen.return_value = mock_response

                generator = ResponseGenerator()
                await generator.initialize()

                answer, token_usage, gen_time = await generator.generate(sample_prompt)

                assert token_usage.prompt_tokens == 120
                assert token_usage.completion_tokens == 80
                assert token_usage.total_tokens == 200

    @pytest.mark.asyncio
    async def test_generate_respects_temperature(self, pipeline_config, sample_prompt):
        """Test that temperature parameter is passed correctly."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            with patch('code_assistance_pipeline.services.response_generator.generate_text') as mock_gen:
                mock_response = MagicMock()
                mock_response.success = True
                mock_response.text = "Answer"
                mock_response.prompt_tokens = 100
                mock_response.response_tokens = 50
                mock_response.total_tokens = 150

                mock_gen.return_value = mock_response

                generator = ResponseGenerator()
                await generator.initialize()

                await generator.generate(sample_prompt, temperature=0.5)

                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args.kwargs
                assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_respects_max_tokens(self, pipeline_config, sample_prompt):
        """Test that max_tokens parameter is passed correctly."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            with patch('code_assistance_pipeline.services.response_generator.generate_text') as mock_gen:
                mock_response = MagicMock()
                mock_response.success = True
                mock_response.text = "Answer"
                mock_response.prompt_tokens = 100
                mock_response.response_tokens = 50
                mock_response.total_tokens = 150

                mock_gen.return_value = mock_response

                generator = ResponseGenerator()
                await generator.initialize()

                await generator.generate(sample_prompt, max_tokens=500)

                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args.kwargs
                assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_generate_handles_error(self, pipeline_config, sample_prompt):
        """Test that generation errors are handled gracefully."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', True):
            with patch('code_assistance_pipeline.services.response_generator.generate_text') as mock_gen:
                mock_response = MagicMock()
                mock_response.success = False
                mock_response.error = "Connection timeout"

                mock_gen.return_value = mock_response

                generator = ResponseGenerator()
                await generator.initialize()

                answer, token_usage, gen_time = await generator.generate(sample_prompt)

                # Should return error message instead of raising
                assert "error" in answer.lower()


# =============================================================================
# Code Explanation Tests
# =============================================================================

class TestResponseGeneratorCodeExplanation:
    """Test code explanation generation."""

    @pytest.mark.requires_llm
    def test_generate_code_explanation(
        self, llm_client: LocalLLMClient, code_explanation_prompt, pipeline_config
    ):
        """Test generating code explanation with real LLM."""
        response = llm_client.generate(
            prompt=code_explanation_prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.2,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["bale"],
        )

    @pytest.mark.requires_llm
    def test_explain_validation_logic(
        self, llm_client: LocalLLMClient, pipeline_config
    ):
        """Test explaining validation logic."""
        prompt = """Explain what validation this code performs:

```csharp
public bool ValidateBale(Bale bale)
{
    if (bale.Weight <= 0) return false;
    if (string.IsNullOrEmpty(bale.GinId)) return false;
    if (bale.BaleDate > DateTime.Now) return false;
    return true;
}
```

Explanation:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.2,
        )

        assert_llm_response_valid(response, min_length=30)
        # Should mention at least some of the validations
        text_lower = response.text.lower()
        assert any(word in text_lower for word in ["weight", "gin", "date", "valid"])


# =============================================================================
# Refactoring Suggestions Tests
# =============================================================================

class TestResponseGeneratorRefactoring:
    """Test refactoring suggestion generation."""

    @pytest.mark.requires_llm
    def test_suggest_sql_injection_fix(
        self, llm_client: LocalLLMClient, refactoring_prompt, pipeline_config
    ):
        """Test suggesting fix for SQL injection vulnerability."""
        response = llm_client.generate(
            prompt=refactoring_prompt,
            model_type="code",
            max_tokens=400,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=30)
        # Should mention parameterized queries or SQL injection
        text_lower = response.text.lower()
        assert any(
            word in text_lower
            for word in ["parameter", "injection", "sql", "query"]
        )

    @pytest.mark.requires_llm
    def test_suggest_async_improvements(
        self, llm_client: LocalLLMClient, pipeline_config
    ):
        """Test suggesting async improvements."""
        prompt = """Suggest improvements for this code:

```csharp
public List<Bale> GetAllBales()
{
    Thread.Sleep(100);  // Simulate network delay
    var result = _db.Query("SELECT * FROM Bales");
    Thread.Sleep(50);
    return result;
}
```

Suggestions:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=30)
        # Should mention async/await
        text_lower = response.text.lower()
        assert any(
            word in text_lower
            for word in ["async", "await", "task", "thread"]
        )


# =============================================================================
# Error Explanation Tests
# =============================================================================

class TestResponseGeneratorErrorExplanation:
    """Test error explanation with source references."""

    @pytest.mark.requires_llm
    def test_explain_null_reference_error(
        self, llm_client: LocalLLMClient, error_explanation_prompt, pipeline_config
    ):
        """Test explaining NullReferenceException."""
        response = llm_client.generate(
            prompt=error_explanation_prompt,
            model_type="code",
            max_tokens=400,
            temperature=0.2,
        )

        assert_llm_response_valid(response, min_length=30)
        # Should mention null checking
        text_lower = response.text.lower()
        assert any(
            word in text_lower
            for word in ["null", "check", "validate", "empty"]
        )

    @pytest.mark.requires_llm
    def test_explain_with_source_reference(
        self, llm_client: LocalLLMClient, pipeline_config
    ):
        """Test that explanation references source files."""
        prompt = """Explain this error:

Error: InvalidOperationException in OrderService.ProcessOrder
Source: /src/Services/OrderService.cs:88
Message: "Order must have at least one item"

The OrderService.ProcessOrder method calls ItemValidator.ValidateItems which throws this error.

Explanation:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.2,
        )

        assert_llm_response_valid(response, min_length=30)
        # Should reference the relevant classes/methods
        text_lower = response.text.lower()
        assert any(
            word in text_lower
            for word in ["order", "item", "validate"]
        )


# =============================================================================
# Token Estimation Tests
# =============================================================================

class TestResponseGeneratorTokenEstimation:
    """Test token estimation functionality."""

    @pytest.mark.asyncio
    async def test_estimate_tokens_empty_text(self, pipeline_config):
        """Test token estimation for empty text."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        generator = ResponseGenerator()
        tokens = generator._estimate_tokens("")

        assert tokens == 0

    @pytest.mark.asyncio
    async def test_estimate_tokens_short_text(self, pipeline_config):
        """Test token estimation for short text."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        generator = ResponseGenerator()
        text = "Hello world"
        tokens = generator._estimate_tokens(text)

        # ~2 words * 1.3 = ~3 tokens
        assert 2 <= tokens <= 5

    @pytest.mark.asyncio
    async def test_estimate_tokens_code_text(self, pipeline_config):
        """Test token estimation for code."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        generator = ResponseGenerator()
        code = """
        public void SaveBale(Bale bale)
        {
            _repository.Insert(bale);
        }
        """
        tokens = generator._estimate_tokens(code)

        # Should estimate reasonably for code
        assert tokens > 0


# =============================================================================
# Health Check Tests
# =============================================================================

class TestResponseGeneratorHealth:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, pipeline_config):
        """Test that health check returns status dict."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', False):
            with patch('services.llm_service.get_llm_service') as mock_get:
                mock_service = AsyncMock()
                mock_service.health_check.return_value = {
                    "healthy": True,
                    "endpoint": "http://localhost:8082",
                }
                mock_get.return_value = mock_service

                generator = ResponseGenerator()
                generator._use_traced = False
                await generator.initialize()

                health = await generator.health_check()

                assert "healthy" in health
                assert health["healthy"] is True


# =============================================================================
# Cache Tests
# =============================================================================

class TestResponseGeneratorCache:
    """Test cache functionality."""

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, pipeline_config):
        """Test getting cache statistics."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', False):
            with patch('services.llm_service.get_llm_service') as mock_get:
                mock_service = AsyncMock()
                mock_service.get_cache_stats.return_value = {
                    "size": 100,
                    "hits": 50,
                    "misses": 30,
                }
                mock_get.return_value = mock_service

                generator = ResponseGenerator()
                generator._use_traced = False
                await generator.initialize()

                stats = await generator.get_cache_stats()

                assert "size" in stats

    @pytest.mark.asyncio
    async def test_clear_cache(self, pipeline_config):
        """Test clearing the cache."""
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        with patch('code_assistance_pipeline.services.response_generator.TRACED_LLM_AVAILABLE', False):
            with patch('services.llm_service.get_llm_service') as mock_get:
                mock_service = AsyncMock()
                mock_get.return_value = mock_service

                generator = ResponseGenerator()
                generator._use_traced = False
                await generator.initialize()

                await generator.clear_cache()

                mock_service.clear_cache.assert_called_once()


# =============================================================================
# E2E Integration Tests with Real LLM
# =============================================================================

class TestResponseGeneratorE2E:
    """End-to-end tests with real LLM."""

    @pytest.mark.requires_llm
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_code_assistance_response(
        self, llm_client: LocalLLMClient, pipeline_config
    ):
        """Test complete code assistance response generation."""
        prompt = """You are a helpful code assistant.

CODE CONTEXT:
Method: BaleService.SaveBale
Class: BaleService
File: /src/Services/BaleService.cs:42
Signature: public void SaveBale(Bale bale)
Summary: Saves a bale record to the database
SQL Operations: 1 database call (INSERT INTO Bales)

Method: BaleService.ValidateBale
Class: BaleService
File: /src/Services/BaleService.cs:28
Signature: private bool ValidateBale(Bale bale)
Summary: Validates bale data before saving

Call Flow: btnSave_Click -> ValidateBale -> SaveBale

USER QUESTION: How do I save a bale in the system?

Please provide a clear answer that references the code elements.

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=400,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=50,
            must_contain=["SaveBale"],
        )

        # Should reference at least one of the context elements
        text_lower = response.text.lower()
        assert any(
            word in text_lower
            for word in ["baleservice", "savebale", "validate"]
        )
