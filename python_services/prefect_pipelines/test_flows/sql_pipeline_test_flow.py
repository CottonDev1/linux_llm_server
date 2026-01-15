"""
SQL Pipeline Test Flow
======================

Prefect flow for testing the SQL generation pipeline.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (port 8080)
- All configuration via Prefect parameters - no hardcoded values

Tests:
1. Storage: agent_learning collection operations
2. Retrieval: Cache hits, rule matching
3. Generation: SQL generation using local llama.cpp
4. E2E: Question → Cache → Rules → Schema → LLM → Validation
"""

import os
import sys
from typing import Dict, Any

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prefect_pipelines.test_flows.base_test_flow import (
    TestFlowConfig,
    run_pytest_module,
    run_test_category,
    create_test_artifact,
)
from prefect_pipelines.test_flows.test_flow_utils import (
    TestStatus,
    TestMetrics,
    TestResult,
    ProgressTracker,
    create_test_report_markdown,
)


@flow(
    name="sql-pipeline-tests",
    description="Run SQL pipeline tests - manual trigger only",
    log_prints=True,
)
def sql_pipeline_test_flow(
    mongodb_uri: str,
    llm_endpoint: str = "http://localhost:8080",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
) -> Dict[str, Any]:
    """
    Run SQL pipeline tests.

    Args:
        mongodb_uri: MongoDB connection URI (required)
        llm_endpoint: Local SQL LLM endpoint (llama.cpp port 8080)
        timeout_seconds: Test timeout per category
        run_storage: Run storage tests
        run_retrieval: Run retrieval tests
        run_generation: Run generation tests
        run_e2e: Run end-to-end tests
        cleanup_after_test: Clean up test data

    Returns:
        Dict with test results and metrics
    """
    logger = get_run_logger()
    logger.info("Starting SQL Pipeline Tests")

    # Validate local endpoint
    if not llm_endpoint.startswith(("http://localhost", "http://127.0.0.1")):
        raise ValueError(f"Only local LLM endpoints allowed. Got: {llm_endpoint}")

    # Create config
    config = TestFlowConfig(
        mongodb_uri=mongodb_uri,
        llm_sql_endpoint=llm_endpoint,
        timeout_seconds=timeout_seconds,
        cleanup_after_test=cleanup_after_test,
        run_storage_tests=run_storage,
        run_retrieval_tests=run_retrieval,
        run_generation_tests=run_generation,
        run_e2e_tests=run_e2e,
    )

    results = []
    metrics = TestMetrics()

    # Run enabled test categories
    if run_storage:
        logger.info("Running SQL storage tests...")
        result = run_test_category("sql", "storage", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Storage: {result.status.value}")

    if run_retrieval:
        logger.info("Running SQL retrieval tests...")
        result = run_test_category("sql", "retrieval", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Retrieval: {result.status.value}")

    if run_generation:
        logger.info("Running SQL generation tests...")
        result = run_test_category("sql", "generation", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Generation: {result.status.value}")

    if run_e2e:
        logger.info("Running SQL E2E tests...")
        result = run_test_category("sql", "e2e", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  E2E: {result.status.value}")

    # Create artifact
    import asyncio
    asyncio.run(create_test_artifact(metrics, results, "sql"))

    success = metrics.failed == 0 and metrics.errors == 0
    logger.info(f"SQL Pipeline Tests Complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": "sql",
        "success": success,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    # For local testing only
    import argparse

    parser = argparse.ArgumentParser(description="Run SQL Pipeline Tests")
    parser.add_argument("--mongodb-uri", required=True, help="MongoDB URI")
    parser.add_argument("--llm-endpoint", default="http://localhost:8080")
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    result = sql_pipeline_test_flow(
        mongodb_uri=args.mongodb_uri,
        llm_endpoint=args.llm_endpoint,
        timeout_seconds=args.timeout,
    )

    print(f"Result: {result}")
