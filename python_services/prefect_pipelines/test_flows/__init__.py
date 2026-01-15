"""
Test Flows Package
==================

Prefect test flows for pipeline integration testing.

IMPORTANT:
- All tests are MANUALLY TRIGGERED ONLY - no automatic scheduling
- All LLM calls use local llama.cpp endpoints only (ports 8080, 8081, 8082)
- No external APIs (OpenAI, Anthropic, etc.) permitted
- All configuration via Prefect parameters - no hardcoded values

Available Flows:
- sql_pipeline_tests: SQL generation pipeline tests
- audio_pipeline_tests: Audio transcription pipeline tests
- query_pipeline_tests: RAG/Query retrieval pipeline tests
- git_pipeline_tests: Git analysis pipeline tests
- code_flow_pipeline_tests: Code flow analysis pipeline tests
- code_assistance_pipeline_tests: Code assistance pipeline tests
- document_pipeline_tests: Document processing pipeline tests
- all_pipelines_tests: Master flow to run all pipeline tests

Shared Utilities:
- test_flow_utils: Utilities for event emission, metrics, and artifacts

Usage:
    # Run via Prefect CLI (manual trigger)
    prefect deployment run 'sql-pipeline-tests/sql-pipeline-tests' \\
        --param mongodb_uri="mongodb://EWRSPT-AI:27018"

    # Or via Python
    from prefect_pipelines.test_flows.sql_pipeline_test_flow import sql_pipeline_test_flow
    sql_pipeline_test_flow(mongodb_uri="mongodb://EWRSPT-AI:27018")
"""

from .test_flow_utils import (
    TestStatus,
    EventType,
    TestResult,
    TestMetrics,
    AgentActivity,
    emit_test_event,
    emit_agent_activity,
    emit_metrics_event,
    create_test_summary_table,
    create_test_results_table,
    create_test_report_markdown,
    ProgressTracker,
    TestTimer,
    parse_pytest_output,
    create_metrics_from_results,
)

__all__ = [
    # Enums
    "TestStatus",
    "EventType",
    # Data Models
    "TestResult",
    "TestMetrics",
    "AgentActivity",
    # Event Functions
    "emit_test_event",
    "emit_agent_activity",
    "emit_metrics_event",
    # Artifact Functions
    "create_test_summary_table",
    "create_test_results_table",
    "create_test_report_markdown",
    # Utilities
    "ProgressTracker",
    "TestTimer",
    "parse_pytest_output",
    "create_metrics_from_results",
]
