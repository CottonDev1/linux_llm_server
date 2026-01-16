"""
Query/RAG Pipeline Test Flow
============================

Prefect flow for testing the query/RAG retrieval pipeline.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (port 8081)
- All configuration via Prefect parameters - no hardcoded values

Tests:
1. Storage: Document embedding storage in MongoDB
2. Retrieval: Semantic search and context retrieval
3. Generation: RAG response generation using local llama.cpp
4. E2E: Query → Embed → Search → Retrieve → Generate
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


async def load_query_test_config() -> Dict[str, Any]:
    """
    Load query test configuration from Prefect Variables.

    Returns:
        Dict containing configuration values
    """
    mongodb_uri = await Variable.get("mongodb_uri", default="mongodb://localhost:27017")
    llm_endpoint = await Variable.get("llm_general_endpoint", default="http://localhost:8081")
    timeout_seconds = await Variable.get("query_test_timeout_seconds", default="300")

    return {
        "mongodb_uri": mongodb_uri,
        "llm_endpoint": llm_endpoint,
        "timeout_seconds": int(timeout_seconds),
    }


@flow(
    name="query-pipeline-tests",
    description="Run query/RAG pipeline tests - manual trigger only",
    log_prints=True,
)
async def query_pipeline_test_flow(
    mongodb_uri: str = "mongodb://localhost:27017",
    llm_endpoint: str = "http://localhost:8081",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
) -> Dict[str, Any]:
    """
    Run query/RAG pipeline tests.

    Args:
        mongodb_uri: MongoDB connection URI (default: mongodb://localhost:27017)
        llm_endpoint: Local General LLM endpoint (llama.cpp port 8081)
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
    logger.info("Starting Query/RAG Pipeline Tests")

    # Load configuration from Prefect Variables (overrides defaults if set)
    var_config = await load_query_test_config()

    # Use Variable values if parameters are at their defaults
    if mongodb_uri == "mongodb://localhost:27017":
        mongodb_uri = var_config["mongodb_uri"]
    if llm_endpoint == "http://localhost:8081":
        llm_endpoint = var_config["llm_endpoint"]
    if timeout_seconds == 300:
        timeout_seconds = var_config["timeout_seconds"]

    if not llm_endpoint.startswith(("http://localhost", "http://127.0.0.1")):
        raise ValueError(f"Only local LLM endpoints allowed. Got: {llm_endpoint}")

    config = TestFlowConfig(
        mongodb_uri=mongodb_uri,
        llm_general_endpoint=llm_endpoint,
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
        logger.info("Running query storage tests...")
        result = run_test_category("query", "storage", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Storage: {result.status.value}")

    if run_retrieval:
        logger.info("Running query retrieval tests...")
        result = run_test_category("query", "retrieval", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Retrieval: {result.status.value}")

    if run_generation:
        logger.info("Running query generation tests...")
        result = run_test_category("query", "generation", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Generation: {result.status.value}")

    if run_e2e:
        logger.info("Running query E2E tests...")
        result = run_test_category("query", "e2e", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  E2E: {result.status.value}")

    await create_test_artifact(metrics, results, "query")

    success = metrics.failed == 0 and metrics.errors == 0
    logger.info(f"Query Pipeline Tests Complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": "query",
        "success": success,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run Query Pipeline Tests")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--llm-endpoint", default="http://localhost:8081")
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    result = asyncio.run(query_pipeline_test_flow(
        mongodb_uri=args.mongodb_uri,
        llm_endpoint=args.llm_endpoint,
        timeout_seconds=args.timeout,
    ))

    print(f"Result: {result}")
