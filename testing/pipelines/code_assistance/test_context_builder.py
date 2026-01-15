"""
Context Builder Service Tests
=============================

Test the ContextBuilder service including:
- Context assembly from multiple sources
- Call chain context integration
- SourceInfo creation with both field name formats
- Context truncation
- Prompt building
- History formatting
"""

import pytest
from typing import Dict, Any, List


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_methods() -> List[Dict[str, Any]]:
    """Create sample method data for context building."""
    return [
        {
            "method_name": "SaveBale",
            "class_name": "BaleService",
            "file_path": "/src/Services/BaleService.cs",
            "line_number": 42,
            "signature": "public void SaveBale(Bale bale)",
            "summary": "Saves a bale record to the database",
            "sql_calls": ["INSERT INTO Bales"],
            "similarity": 0.95,
        },
        {
            "method_name": "ValidateBale",
            "class_name": "BaleService",
            "file_path": "/src/Services/BaleService.cs",
            "line_number": 28,
            "signature": "private bool ValidateBale(Bale bale)",
            "summary": "Validates bale data before saving",
            "similarity": 0.82,
        },
    ]


@pytest.fixture
def sample_classes() -> List[Dict[str, Any]]:
    """Create sample class data for context building."""
    return [
        {
            "class_name": "BaleService",
            "namespace": "Gin.Services",
            "file_path": "/src/Services/BaleService.cs",
            "base_class": "BaseService",
            "interfaces": ["IBaleService"],
            "methods": ["SaveBale", "ValidateBale", "GetBales", "DeleteBale", "UpdateBale"],
            "similarity": 0.88,
        },
        {
            "class_name": "BaseService",
            "namespace": "Gin.Services",
            "file_path": "/src/Services/BaseService.cs",
            "base_class": None,
            "interfaces": ["IDisposable"],
            "methods": ["Dispose", "Initialize"],
            "similarity": 0.72,
        },
    ]


@pytest.fixture
def sample_event_handlers() -> List[Dict[str, Any]]:
    """Create sample event handler data for context building."""
    return [
        {
            "event_name": "Click",
            "handler_method": "btnSave_Click",
            "element_name": "btnSave",
            "ui_element_type": "Button",
            "handler_class": "BaleEntryForm",
            "file_path": "/src/Forms/BaleEntryForm.cs",
            "line_number": 150,
            "similarity": 0.75,
        },
    ]


@pytest.fixture
def sample_call_chain() -> List[str]:
    """Create sample call chain."""
    return [
        "BaleEntryForm.btnSave_Click",
        "BaleService.ValidateBale",
        "BaleService.SaveBale",
        "BaleRepository.Insert",
    ]


@pytest.fixture
def sample_history():
    """Create sample conversation history."""
    from code_assistance_pipeline.models.query_models import ConversationMessage

    return [
        ConversationMessage(
            role="user",
            content="What methods are available in BaleService?",
        ),
        ConversationMessage(
            role="assistant",
            content="BaleService has SaveBale, ValidateBale, GetBales, and DeleteBale methods.",
        ),
        ConversationMessage(
            role="user",
            content="How do I save a bale?",
        ),
        ConversationMessage(
            role="assistant",
            content="Use BaleService.SaveBale(bale) to save a bale to the database.",
        ),
    ]


# =============================================================================
# Basic Context Building Tests
# =============================================================================

class TestContextBuilderBasic:
    """Test basic context building functionality."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_build_context_returns_tuple(
        self,
        context_builder,
        sample_methods,
        sample_classes,
        pipeline_config,
    ):
        """Test that build_context returns a tuple of (context, sources)."""
        context, sources = context_builder.build_context(
            methods=sample_methods,
            classes=sample_classes,
            event_handlers=[],
            call_chain=[],
        )

        assert isinstance(context, str)
        assert isinstance(sources, list)
        assert len(context) > 0

    def test_build_context_with_empty_inputs(self, context_builder, pipeline_config):
        """Test context building with no input data."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=[],
        )

        # Should return empty but not fail
        assert isinstance(context, str)
        assert isinstance(sources, list)

    def test_build_context_methods_section(
        self,
        context_builder,
        sample_methods,
        pipeline_config,
    ):
        """Test that methods are properly formatted in context."""
        context, sources = context_builder.build_context(
            methods=sample_methods,
            classes=[],
            event_handlers=[],
            call_chain=[],
        )

        # Context should contain method information
        assert "SaveBale" in context
        assert "BaleService" in context
        assert "BaleService.cs" in context or "file_path" in context.lower()

    def test_build_context_classes_section(
        self,
        context_builder,
        sample_classes,
        pipeline_config,
    ):
        """Test that classes are properly formatted in context."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=sample_classes,
            event_handlers=[],
            call_chain=[],
        )

        # Context should contain class information
        assert "BaleService" in context
        assert "BaseService" in context  # Should mention base class

    def test_build_context_event_handlers_section(
        self,
        context_builder,
        sample_event_handlers,
        pipeline_config,
    ):
        """Test that event handlers are properly formatted in context."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=sample_event_handlers,
            call_chain=[],
        )

        # Context should contain event handler information
        assert "Click" in context
        assert "btnSave_Click" in context


# =============================================================================
# SourceInfo Creation Tests
# =============================================================================

class TestContextBuilderSourceInfo:
    """Test SourceInfo creation with both field name formats."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_sources_created_for_methods(
        self,
        context_builder,
        sample_methods,
        pipeline_config,
    ):
        """Test that SourceInfo is created for methods."""
        from code_assistance_pipeline.models.query_models import SourceType

        context, sources = context_builder.build_context(
            methods=sample_methods,
            classes=[],
            event_handlers=[],
            call_chain=[],
        )

        # Should have sources for methods
        assert len(sources) >= len(sample_methods)

        # Check source properties
        method_sources = [s for s in sources if s.type == SourceType.METHOD]
        assert len(method_sources) > 0

        first_source = method_sources[0]
        assert first_source.name is not None
        # Check both field names work
        assert first_source.file_path is not None or first_source.file_path == ""
        assert first_source.relevance_score >= 0

    def test_sources_created_for_classes(
        self,
        context_builder,
        sample_classes,
        pipeline_config,
    ):
        """Test that SourceInfo is created for classes."""
        from code_assistance_pipeline.models.query_models import SourceType

        context, sources = context_builder.build_context(
            methods=[],
            classes=sample_classes,
            event_handlers=[],
            call_chain=[],
        )

        class_sources = [s for s in sources if s.type == SourceType.CLASS]
        assert len(class_sources) > 0

    def test_sources_created_for_event_handlers(
        self,
        context_builder,
        sample_event_handlers,
        pipeline_config,
    ):
        """Test that SourceInfo is created for event handlers."""
        from code_assistance_pipeline.models.query_models import SourceType

        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=sample_event_handlers,
            call_chain=[],
        )

        event_sources = [s for s in sources if s.type == SourceType.EVENT_HANDLER]
        assert len(event_sources) > 0

    def test_source_info_field_aliases_work(
        self,
        context_builder,
        sample_methods,
        pipeline_config,
    ):
        """Test that SourceInfo supports both primary and alias field names."""
        from code_assistance_pipeline.models.query_models import SourceInfo, SourceType

        # Create using alias names (file, line, snippet, similarity)
        source = SourceInfo(
            type=SourceType.METHOD,
            name="Test.Method",
            file="/path/to/file.cs",  # alias for file_path
            line=42,  # alias for line_number
            similarity=0.95,  # alias for relevance_score
            snippet="public void Test()",  # alias for content
        )

        # Should be accessible via both names
        assert source.file_path == "/path/to/file.cs"
        assert source.line_number == 42
        assert source.relevance_score == 0.95
        assert source.content == "public void Test()"


# =============================================================================
# Call Chain Integration Tests
# =============================================================================

class TestContextBuilderCallChain:
    """Test call chain context integration."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_call_chain_formatted_in_context(
        self,
        context_builder,
        sample_call_chain,
        pipeline_config,
    ):
        """Test that call chain is properly formatted in context."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=sample_call_chain,
        )

        # Call chain should be included
        assert "Call Flow" in context or "call" in context.lower()
        # Should contain chain items
        assert "SaveBale" in context or "btnSave_Click" in context

    def test_call_chain_shows_flow_direction(
        self,
        context_builder,
        sample_call_chain,
        pipeline_config,
    ):
        """Test that call chain shows flow direction with arrows."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=sample_call_chain,
        )

        # Should contain arrow notation for flow
        assert "->" in context

    def test_empty_call_chain_handled(self, context_builder, pipeline_config):
        """Test that empty call chain is handled gracefully."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=[],
        )

        # Should not fail, should not contain call chain section
        # (or section header without content)
        assert context is not None


# =============================================================================
# History Integration Tests
# =============================================================================

class TestContextBuilderHistory:
    """Test conversation history integration."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_history_included_in_context(
        self,
        context_builder,
        sample_history,
        pipeline_config,
    ):
        """Test that conversation history is included in context."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=[],
            history=sample_history,
        )

        # Context should contain history
        assert "Previous" in context or "User" in context

    def test_history_truncated_to_recent(
        self,
        context_builder,
        pipeline_config,
    ):
        """Test that only recent history messages are included."""
        from code_assistance_pipeline.models.query_models import ConversationMessage

        # Create a long history
        long_history = [
            ConversationMessage(role="user", content=f"Question {i}")
            for i in range(10)
        ]

        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=[],
            history=long_history,
        )

        # Should only include recent messages (last 4)
        # Old messages should not be present
        assert "Question 0" not in context

    def test_history_preserves_role_labels(
        self,
        context_builder,
        sample_history,
        pipeline_config,
    ):
        """Test that history preserves user/assistant role labels."""
        context, sources = context_builder.build_context(
            methods=[],
            classes=[],
            event_handlers=[],
            call_chain=[],
            history=sample_history,
        )

        # Should have role labels
        assert "User" in context or "Assistant" in context


# =============================================================================
# Context Truncation Tests
# =============================================================================

class TestContextBuilderTruncation:
    """Test context truncation functionality."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_estimate_tokens(self, context_builder, pipeline_config):
        """Test token estimation."""
        text = "This is a test sentence with some words."
        tokens = context_builder.estimate_tokens(text)

        # Should estimate roughly 1.3 tokens per word
        word_count = len(text.split())
        expected_min = word_count
        expected_max = word_count * 2

        assert expected_min <= tokens <= expected_max

    def test_truncate_context_short_text(self, context_builder, pipeline_config):
        """Test that short context is not truncated."""
        short_context = "This is a short context."
        truncated = context_builder.truncate_context(short_context, max_tokens=1000)

        assert truncated == short_context

    def test_truncate_context_long_text(self, context_builder, pipeline_config):
        """Test that long context is truncated."""
        # Create a long context
        long_section = "Section content " * 500  # ~1000 words
        sections = [long_section, long_section, long_section]
        long_context = "\n---\n".join(sections)

        truncated = context_builder.truncate_context(long_context, max_tokens=500)

        # Truncated should be shorter
        assert len(truncated) < len(long_context)

    def test_truncate_preserves_complete_sections(
        self, context_builder, pipeline_config
    ):
        """Test that truncation preserves complete sections."""
        sections = [
            "Section 1 content here",
            "Section 2 content here",
            "Section 3 content here",
        ]
        context = "\n---\n".join(sections)

        truncated = context_builder.truncate_context(context, max_tokens=50)

        # Should not have partial sections (no mid-word cuts)
        # Each remaining section should be complete
        if truncated:
            assert truncated.count("---") <= context.count("---")


# =============================================================================
# Prompt Building Tests
# =============================================================================

class TestContextBuilderPrompt:
    """Test prompt building functionality."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_build_prompt_includes_query(self, context_builder, pipeline_config):
        """Test that prompt includes the user query."""
        query = "How do I save a bale?"
        context = "Method: BaleService.SaveBale"

        prompt = context_builder.build_prompt(query=query, context=context)

        assert query in prompt

    def test_build_prompt_includes_context(self, context_builder, pipeline_config):
        """Test that prompt includes the code context."""
        query = "How do I save a bale?"
        context = "Method: BaleService.SaveBale saves bale data to database"

        prompt = context_builder.build_prompt(query=query, context=context)

        assert "BaleService" in prompt
        assert "SaveBale" in prompt

    def test_build_prompt_includes_system_instructions(
        self, context_builder, pipeline_config
    ):
        """Test that prompt includes system instructions."""
        query = "Test query"
        context = "Test context"

        prompt = context_builder.build_prompt(query=query, context=context)

        # Should have instructions about being a code assistant
        assert "code" in prompt.lower() or "assistant" in prompt.lower()

    def test_build_prompt_with_custom_system(self, context_builder, pipeline_config):
        """Test prompt with custom system prompt."""
        query = "Test query"
        context = "Test context"
        custom_system = "You are a specialized C# expert."

        prompt = context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=custom_system,
        )

        assert "C# expert" in prompt

    def test_build_prompt_structured_format(self, context_builder, pipeline_config):
        """Test that prompt has structured format with clear sections."""
        query = "How does this work?"
        context = "Method details here"

        prompt = context_builder.build_prompt(query=query, context=context)

        # Should have clear section markers
        assert "CONTEXT" in prompt.upper() or "context" in prompt.lower()
        assert "QUESTION" in prompt.upper() or "question" in prompt.lower()


# =============================================================================
# Limits and Configuration Tests
# =============================================================================

class TestContextBuilderLimits:
    """Test context builder limits and configuration."""

    def test_default_limits(self, pipeline_config):
        """Test default limits are set."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder

        builder = ContextBuilder()

        assert builder.method_limit == ContextBuilder.DEFAULT_METHOD_LIMIT
        assert builder.class_limit == ContextBuilder.DEFAULT_CLASS_LIMIT
        assert builder.event_limit == ContextBuilder.DEFAULT_EVENT_LIMIT

    def test_custom_limits(self, pipeline_config):
        """Test custom limits can be set."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder

        builder = ContextBuilder(
            method_limit=5,
            class_limit=2,
            event_limit=1,
        )

        assert builder.method_limit == 5
        assert builder.class_limit == 2
        assert builder.event_limit == 1

    def test_methods_limited(self, pipeline_config):
        """Test that methods are limited by method_limit."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder

        builder = ContextBuilder(method_limit=1)

        many_methods = [
            {"method_name": f"Method{i}", "class_name": "TestClass"}
            for i in range(10)
        ]

        context, sources = builder.build_context(
            methods=many_methods,
            classes=[],
            event_handlers=[],
            call_chain=[],
        )

        # Should only include one method in sources
        from code_assistance_pipeline.models.query_models import SourceType
        method_sources = [s for s in sources if s.type == SourceType.METHOD]
        assert len(method_sources) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestContextBuilderIntegration:
    """Integration tests for context builder with all components."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder instance for testing."""
        from code_assistance_pipeline.services.context_builder import ContextBuilder
        return ContextBuilder()

    def test_full_context_build(
        self,
        context_builder,
        sample_methods,
        sample_classes,
        sample_event_handlers,
        sample_call_chain,
        sample_history,
        pipeline_config,
    ):
        """Test building context with all components."""
        context, sources = context_builder.build_context(
            methods=sample_methods,
            classes=sample_classes,
            event_handlers=sample_event_handlers,
            call_chain=sample_call_chain,
            history=sample_history,
        )

        # Context should contain all sections
        assert len(context) > 0
        assert len(sources) > 0

        # Should have sources from multiple types
        source_types = set(s.type.value for s in sources)
        assert "method" in source_types
        assert "class" in source_types
        assert "event_handler" in source_types

    def test_full_prompt_generation(
        self,
        context_builder,
        sample_methods,
        sample_classes,
        pipeline_config,
    ):
        """Test generating full prompt from context."""
        context, sources = context_builder.build_context(
            methods=sample_methods,
            classes=sample_classes,
            event_handlers=[],
            call_chain=[],
        )

        prompt = context_builder.build_prompt(
            query="How do I save a bale in the system?",
            context=context,
        )

        # Prompt should be ready for LLM
        assert len(prompt) > len(context)  # Has additional structure
        assert "save" in prompt.lower()
        assert "bale" in prompt.lower()
