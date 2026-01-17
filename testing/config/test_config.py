"""
Test Configuration Models
=========================

Pydantic models for test configuration.
All values are parameterized - NO HARDCODED VALUES.

Usage:
    config = get_test_config()
    # or
    config = TestConfig(mongodb_uri="mongodb://localhost:27017", ...)
"""

import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class MongoDBConfig(BaseModel):
    """MongoDB connection configuration."""

    uri: str = Field(
        default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        description="MongoDB connection URI"
    )
    database: str = Field(
        default_factory=lambda: os.getenv("MONGODB_DATABASE", "rag_server"),
        description="MongoDB database name"
    )
    timeout_ms: int = Field(
        default=30000,
        description="Connection timeout in milliseconds"
    )


class LLMConfig(BaseModel):
    """
    Local LLM endpoint configuration.

    CRITICAL: Only local llama.cpp endpoints are permitted.
    No external APIs (OpenAI, Anthropic, etc.) allowed.
    """

    sql_endpoint: str = Field(
        default_factory=lambda: os.getenv("LLAMACPP_SQL_HOST", "http://localhost:8080"),
        description="SQL LLM endpoint (llama.cpp) - port 8080"
    )
    general_endpoint: str = Field(
        default_factory=lambda: os.getenv("LLAMACPP_HOST", "http://localhost:8081"),
        description="General LLM endpoint (llama.cpp) - port 8081"
    )
    code_endpoint: str = Field(
        default_factory=lambda: os.getenv("LLAMACPP_CODE_HOST", "http://localhost:8082"),
        description="Code LLM endpoint (llama.cpp) - port 8082"
    )
    request_timeout: int = Field(
        default=120,
        description="LLM request timeout in seconds"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM response"
    )
    temperature: float = Field(
        default=0.0,
        description="LLM temperature (0.0 for deterministic)"
    )

    @field_validator("sql_endpoint", "general_endpoint", "code_endpoint")
    @classmethod
    def validate_local_endpoint(cls, v: str) -> str:
        """Ensure endpoint is local - no external APIs allowed."""
        if not v.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                f"Only local endpoints allowed. Got: {v}. "
                "External APIs (OpenAI, Anthropic, etc.) are not permitted."
            )
        return v


class TestConfig(BaseModel):
    """
    Base configuration for all pipeline tests.

    All values are configurable via environment variables or direct parameters.
    NO HARDCODED VALUES.
    """

    # MongoDB Configuration
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)

    # LLM Configuration (local llama.cpp only)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Test Behavior
    timeout_seconds: int = Field(
        default=300,
        description="Test timeout in seconds"
    )
    cleanup_after_test: bool = Field(
        default=True,
        description="Clean up test data after execution"
    )
    use_test_prefix: bool = Field(
        default=True,
        description="Prefix test data with 'test_' for isolation"
    )
    test_run_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this test run"
    )

    # Test Data Paths
    test_data_path: Optional[str] = Field(
        default=None,
        description="Path to test data files"
    )

    # Thresholds
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for matching tests"
    )
    confidence_threshold: float = Field(
        default=0.8,
        description="Minimum confidence score for generation tests"
    )


class PipelineTestConfig(TestConfig):
    """
    Extended configuration for pipeline-specific tests.

    Inherits from TestConfig and adds pipeline-specific settings.
    """

    # Pipeline identification
    pipeline_name: str = Field(
        default="unknown",
        description="Name of the pipeline being tested"
    )

    # Test execution flags
    run_storage_tests: bool = Field(
        default=True,
        description="Run storage validation tests"
    )
    run_retrieval_tests: bool = Field(
        default=True,
        description="Run retrieval validation tests"
    )
    run_generation_tests: bool = Field(
        default=True,
        description="Run LLM generation tests"
    )
    run_e2e_tests: bool = Field(
        default=True,
        description="Run end-to-end pipeline tests"
    )

    # Pipeline-specific settings
    max_results: int = Field(
        default=100,
        description="Maximum results to retrieve in tests"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for bulk operations"
    )


def get_test_config(**overrides) -> TestConfig:
    """
    Get test configuration with optional overrides.

    Args:
        **overrides: Key-value pairs to override default configuration

    Returns:
        TestConfig instance with applied overrides

    Example:
        config = get_test_config(
            timeout_seconds=600,
            cleanup_after_test=False
        )
    """
    return TestConfig(**overrides)


def get_pipeline_config(pipeline_name: str, **overrides) -> PipelineTestConfig:
    """
    Get pipeline-specific test configuration.

    Args:
        pipeline_name: Name of the pipeline (e.g., "sql", "audio", "git")
        **overrides: Key-value pairs to override default configuration

    Returns:
        PipelineTestConfig instance for the specified pipeline
    """
    return PipelineTestConfig(pipeline_name=pipeline_name, **overrides)
