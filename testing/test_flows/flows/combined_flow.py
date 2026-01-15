"""High-level test flows for common scenarios.

These flows provide convenient entry points for common testing scenarios,
built on top of the core pipeline_test_flow.
"""
from typing import Dict, Any
from prefect import flow, get_run_logger

from .pipeline_flows import pipeline_test_flow


@flow(name="full-test-suite", log_prints=True)
def full_test_flow() -> Dict[str, Any]:
    """
    Run the complete test suite - all pipelines.

    This is the most comprehensive test run, covering:
    - SQL Pipeline
    - Audio Pipeline
    - Query/RAG Pipeline
    - Git Pipeline
    - Code Flow Pipeline
    - Code Assistance Pipeline
    - Document Pipeline
    - Shared/Cross-cutting tests

    Returns:
        Dict with success status and detailed results for all pipelines
    """
    logger = get_run_logger()
    logger.info("Starting full test suite...")
    return pipeline_test_flow(pipelines=None, parallel=False)


@flow(name="smoke-test", log_prints=True)
def smoke_test_flow() -> Dict[str, Any]:
    """
    Quick smoke test - SQL and Query pipelines only.

    Use this for fast validation of core functionality.
    Runs the two most critical pipelines to catch major issues quickly.

    Returns:
        Dict with success status and results
    """
    logger = get_run_logger()
    logger.info("Running quick smoke test (SQL + Query)...")
    return pipeline_test_flow(pipelines=["sql", "query"], parallel=False)


__all__ = [
    "full_test_flow",
    "smoke_test_flow",
]
