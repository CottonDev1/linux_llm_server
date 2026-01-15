"""Prefect test flows for the LLM Website project."""

from .pipeline_flows import (
    sql_pipeline_test_flow,
    audio_pipeline_test_flow,
    query_pipeline_test_flow,
    git_pipeline_test_flow,
    code_flow_pipeline_test_flow,
    code_assistance_pipeline_test_flow,
    document_pipeline_test_flow,
    all_pipelines_test_flow,
    run_pipeline_tests,
    PipelineTestResult,
)
from .combined_flow import (
    api_test_flow,
    full_test_flow,
    quick_smoke_test_flow,
)
from .granular_test_flow import (
    run_specific_tests,
    run_tests_by_marker,
    run_tests_by_pattern,
    GranularTestResult,
    build_pytest_command,
    parse_pytest_output,
    execute_pytest,
    create_granular_test_artifact,
)

__all__ = [
    # Pipeline flows
    "sql_pipeline_test_flow",
    "audio_pipeline_test_flow",
    "query_pipeline_test_flow",
    "git_pipeline_test_flow",
    "code_flow_pipeline_test_flow",
    "code_assistance_pipeline_test_flow",
    "document_pipeline_test_flow",
    "all_pipelines_test_flow",
    "run_pipeline_tests",
    "PipelineTestResult",
    # Combined flows
    "api_test_flow",
    "full_test_flow",
    "quick_smoke_test_flow",
    # Granular test flows
    "run_specific_tests",
    "run_tests_by_marker",
    "run_tests_by_pattern",
    "GranularTestResult",
    "build_pytest_command",
    "parse_pytest_output",
    "execute_pytest",
    "create_granular_test_artifact",
]
