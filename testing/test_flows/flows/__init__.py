"""Consolidated Prefect test flows.

This module provides 4 main flows for testing:

1. pipeline_test_flow - Run tests for any combination of pipelines
2. full_test_flow - Complete test suite (all pipelines)
3. smoke_test_flow - Quick validation (SQL + Query only)
4. custom_test_flow - Fine-grained control with pytest filters

Usage in Prefect UI or code:
    from testing.test_flows.flows import pipeline_test_flow, full_test_flow

    # Run specific pipelines
    pipeline_test_flow(pipelines=["sql", "audio"])

    # Run all tests
    full_test_flow()
"""

from .pipeline_flows import (
    pipeline_test_flow,
    run_pipeline_tests,
    PipelineTestResult,
    PIPELINE_CONFIG,
)
from .combined_flow import (
    full_test_flow,
    smoke_test_flow,
)
from .granular_test_flow import (
    custom_test_flow,
    CustomTestResult,
)

__all__ = [
    # Main flows (4 total)
    "pipeline_test_flow",   # Parameterized - any/all pipelines
    "full_test_flow",       # All pipelines
    "smoke_test_flow",      # Quick validation
    "custom_test_flow",     # Fine-grained pytest control
    # Supporting types
    "PipelineTestResult",
    "CustomTestResult",
    "PIPELINE_CONFIG",
    "run_pipeline_tests",
]
