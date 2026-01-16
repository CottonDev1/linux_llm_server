"""
Git Analysis Pipeline Test Flow
===============================

Prefect flow for testing the git analysis pipeline.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (port 8082)
- All configuration via Prefect parameters - no hardcoded values

Tests:
1. Storage: code_methods, code_classes collection operations
2. Retrieval: Method/class lookup, semantic code search
3. Generation: Code analysis using local llama.cpp
4. E2E: Parse → Extract → Store → Analyze
"""

import os
import sys
from typing import Dict, Any

from prefect import flow, get_run_logger
from prefect.variables import Variable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prefect_pipelines.test_flows.base_test_flow import (
    TestFlowConfig,
    run_test_category,
    create_test_artifact,
)
from prefect_pipelines.test_flows.test_flow_utils import (
    TestMetrics,
)


async def load_git_test_config() -> Dict[str, Any]:
    """
    Load git test configuration from Prefect Variables.

    Returns:
        Dict with configuration values loaded from Variables with defaults.
    """
    mongodb_uri = await Variable.get("mongodb_uri", default="mongodb://localhost:27017")
    llm_code_endpoint = await Variable.get("llm_code_endpoint", default="http://localhost:8082")
    test_timeout = await Variable.get("test_timeout_seconds", default="300")

    return {
        "mongodb_uri": mongodb_uri,
        "llm_endpoint": llm_code_endpoint,
        "timeout_seconds": int(test_timeout),
    }


@flow(
    name="git-pipeline-tests",
    description="Run git analysis pipeline tests - manual trigger only",
    log_prints=True,
)
async def git_pipeline_test_flow(
    mongodb_uri: str = "mongodb://localhost:27017",
    llm_endpoint: str = "http://localhost:8082",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
) -> Dict[str, Any]:
    """
    Run git analysis pipeline tests.

    Args:
        mongodb_uri: MongoDB connection URI (default: mongodb://localhost:27017)
        llm_endpoint: Local Code LLM endpoint (llama.cpp port 8082)
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
    logger.info("Starting Git Analysis Pipeline Tests")

    # Load configuration from Prefect Variables (overrides defaults)
    var_config = await load_git_test_config()
    mongodb_uri = var_config.get("mongodb_uri", mongodb_uri)
    llm_endpoint = var_config.get("llm_endpoint", llm_endpoint)
    timeout_seconds = var_config.get("timeout_seconds", timeout_seconds)

    logger.info(f"Using MongoDB URI: {mongodb_uri}")
    logger.info(f"Using LLM endpoint: {llm_endpoint}")

    if not llm_endpoint.startswith(("http://localhost", "http://127.0.0.1")):
        raise ValueError(f"Only local LLM endpoints allowed. Got: {llm_endpoint}")

    config = TestFlowConfig(
        mongodb_uri=mongodb_uri,
        llm_code_endpoint=llm_endpoint,
        timeout_seconds=timeout_seconds,
        cleanup_after_test=cleanup_after_test,
        run_storage_tests=run_storage,
        run_retrieval_tests=run_retrieval,
        run_generation_tests=run_generation,
        run_e2e_tests=run_e2e,
    )

    results = []
    metrics = TestMetrics()

    if run_storage:
        logger.info("Running git storage tests...")
        result = run_test_category("git", "storage", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Storage: {result.status.value}")

    if run_retrieval:
        logger.info("Running git retrieval tests...")
        result = run_test_category("git", "retrieval", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Retrieval: {result.status.value}")

    if run_generation:
        logger.info("Running git generation tests...")
        result = run_test_category("git", "generation", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Generation: {result.status.value}")

    if run_e2e:
        logger.info("Running git E2E tests...")
        result = run_test_category("git", "e2e", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  E2E: {result.status.value}")

    await create_test_artifact(metrics, results, "git")

    success = metrics.failed == 0 and metrics.errors == 0
    logger.info(f"Git Pipeline Tests Complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": "git",
        "success": success,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run Git Analysis Pipeline Tests")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--llm-endpoint", default="http://localhost:8082")
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    result = asyncio.run(git_pipeline_test_flow(
        mongodb_uri=args.mongodb_uri,
        llm_endpoint=args.llm_endpoint,
        timeout_seconds=args.timeout,
    ))

    print(f"Result: {result}")
