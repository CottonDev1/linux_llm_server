"""Prefect test flows package.

Run `python -m testing.test_flows.serve_flows` to serve flows to Prefect UI.
"""
from .flows import (
    pipeline_test_flow,
    full_test_flow,
    smoke_test_flow,
    custom_test_flow,
    PipelineTestResult,
    CustomTestResult,
    PIPELINE_CONFIG,
)

__all__ = [
    "pipeline_test_flow",
    "full_test_flow",
    "smoke_test_flow",
    "custom_test_flow",
    "PipelineTestResult",
    "CustomTestResult",
    "PIPELINE_CONFIG",
]
