"""
Document Agent Pipeline Test Flow
=================================

Prefect flow for testing the document processing pipeline.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (port 8081)
- All configuration via Prefect parameters - no hardcoded values

Tests:
1. Storage: documents collection operations
2. Retrieval: Document search, chunk retrieval
3. Generation: Document Q&A using local llama.cpp
4. E2E: Upload → Parse → Chunk → Embed → Query → Generate
"""

import asyncio
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


# =============================================================================
# Prefect Variables Support
# =============================================================================

async def load_document_test_config() -> Dict[str, Any]:
    """
    Load document test configuration from Prefect Variables.

    Loads test-related settings with sensible defaults.

    Returns:
        Dict with mongodb_uri, llm_endpoint, timeout settings
    """
    mongodb_uri = await Variable.get(
        "mongodb_uri",
        default="mongodb://localhost:27017/?directConnection=true"
    )
    llm_endpoint = await Variable.get(
        "llm_general_endpoint",
        default="http://localhost:8081"
    )
    timeout_seconds = await Variable.get(
        "test_timeout_seconds",
        default=300
    )

    return {
        "mongodb_uri": mongodb_uri,
        "llm_endpoint": llm_endpoint,
        "timeout_seconds": int(timeout_seconds) if timeout_seconds else 300,
    }


@flow(
    name="document-pipeline-tests",
    description="Run document agent pipeline tests - manual trigger only",
    log_prints=True,
)
async def document_pipeline_test_flow(
    mongodb_uri: str = "mongodb://localhost:27017/?directConnection=true",
    llm_endpoint: str = "http://localhost:8081",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
) -> Dict[str, Any]:
    """
    Run document agent pipeline tests.

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
    logger.info("Starting Document Agent Pipeline Tests")

    # Load configuration from Prefect Variables (overrides defaults if set)
    test_config = await load_document_test_config()

    # Use Prefect Variables values if parameters are at their defaults
    if mongodb_uri == "mongodb://localhost:27017/?directConnection=true":
        mongodb_uri = test_config["mongodb_uri"]
    if llm_endpoint == "http://localhost:8081":
        llm_endpoint = test_config["llm_endpoint"]
    if timeout_seconds == 300:
        timeout_seconds = test_config["timeout_seconds"]

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
        logger.info("Running document_agent storage tests...")
        result = run_test_category("document_agent", "storage", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Storage: {result.status.value}")

    if run_retrieval:
        logger.info("Running document_agent retrieval tests...")
        result = run_test_category("document_agent", "retrieval", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Retrieval: {result.status.value}")

    if run_generation:
        logger.info("Running document_agent generation tests...")
        result = run_test_category("document_agent", "generation", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Generation: {result.status.value}")

    if run_e2e:
        logger.info("Running document_agent E2E tests...")
        result = run_test_category("document_agent", "e2e", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  E2E: {result.status.value}")

    await create_test_artifact(metrics, results, "document_agent")

    success = metrics.failed == 0 and metrics.errors == 0
    logger.info(f"Document Agent Pipeline Tests Complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": "document_agent",
        "success": success,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Document Agent Pipeline Tests")
    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://localhost:27017/?directConnection=true",
        help="MongoDB URI (default: mongodb://localhost:27017)"
    )
    parser.add_argument("--llm-endpoint", default="http://localhost:8081")
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    result = asyncio.run(document_pipeline_test_flow(
        mongodb_uri=args.mongodb_uri,
        llm_endpoint=args.llm_endpoint,
        timeout_seconds=args.timeout,
    ))

    print(f"Result: {result}")
