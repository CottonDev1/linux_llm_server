"""
SQL Pipeline Package

This package provides a comprehensive SQL generation and validation pipeline
for natural language to SQL conversion with LLM support.

Components:
- query_pipeline: Main query processing pipeline
- training_pipeline: Training data generation pipeline
- services: Individual service modules (rules, schema, security, validation, execution)
- models: Pydantic data models
- utils: Utility functions for SQL parsing and parameter inference
"""

from sql_pipeline.query_pipeline import QueryPipeline, get_query_pipeline
# TrainingPipeline is Prefect-based, import the flow directly if needed
from sql_pipeline.training_pipeline import sql_training_flow
from sql_pipeline.models.query_models import (
    SQLCredentials,
    QueryOptions,
    SQLQueryRequest,
    SSEEvent,
    SQLQueryResult,
)
from sql_pipeline.models.rule_models import (
    AutoFix,
    RuleExample,
    SQLRule,
)
from sql_pipeline.models.validation_models import (
    ValidationResult,
    ExecutionResult,
)
from sql_pipeline.models.training_models import (
    TrainingPipelineConfig,
    GeneratedQuestion,
    SQLCandidate,
    TrainingExample,
    TrainingResult,
)

__all__ = [
    # Pipelines
    "QueryPipeline",
    "get_query_pipeline",
    "sql_training_flow",
    # Query Models
    "SQLCredentials",
    "QueryOptions",
    "SQLQueryRequest",
    "SSEEvent",
    "SQLQueryResult",
    # Rule Models
    "AutoFix",
    "RuleExample",
    "SQLRule",
    # Validation Models
    "ValidationResult",
    "ExecutionResult",
    # Training Models
    "TrainingPipelineConfig",
    "GeneratedQuestion",
    "SQLCandidate",
    "TrainingExample",
    "TrainingResult",
]

__version__ = "0.1.0"
