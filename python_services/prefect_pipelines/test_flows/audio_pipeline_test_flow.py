"""
Audio Pipeline Test Flow
========================

Prefect flow for testing the audio transcription pipeline.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (port 8081)
- All configuration via Prefect parameters - no hardcoded values

Tests:
1. Storage: audio_analyses collection operations
2. Retrieval: Audio analysis lookup, call metadata
3. Generation: Transcription summarization using local llama.cpp
4. E2E: Audio → Transcribe → Analyze → Summarize
"""

import os
import sys
from typing import Dict, Any

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.variables import Variable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prefect_pipelines.test_flows.base_test_flow import (
    TestFlowConfig,
    run_test_category,
    create_test_artifact,
)
from prefect_pipelines.test_flows.test_flow_utils import (
    TestMetrics,
    TestResult,
)


async def load_audio_test_config() -> Dict[str, Any]:
    """
    Load audio test configuration from Prefect Variables.

    Returns:
        Dict with configuration values from Prefect Variables
    """
    config = {
        "audio_file_path": await Variable.get("audio_file_path", default=""),
        "audio_customer_name": await Variable.get("audio_customer_name", default="Customer"),
        "audio_support_name": await Variable.get("audio_support_name", default="Support"),
        "default_max_tokens": await Variable.get("default_max_tokens", default="2048"),
    }
    return config


@flow(
    name="audio-pipeline-tests",
    description="Run audio pipeline tests - manual trigger only",
    log_prints=True,
)
async def audio_pipeline_test_flow(
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
    Run audio pipeline tests.

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
    logger.info("Starting Audio Pipeline Tests")

    # Load configuration from Prefect Variables
    audio_config = await load_audio_test_config()
    logger.info(f"Loaded audio test config: {audio_config}")

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
        logger.info("Running audio storage tests...")
        result = run_test_category("audio", "storage", config)
        results.append(result)
        metrics.add_result(result)

    if run_retrieval:
        logger.info("Running audio retrieval tests...")
        result = run_test_category("audio", "retrieval", config)
        results.append(result)
        metrics.add_result(result)

    if run_generation:
        logger.info("Running audio generation tests...")
        result = run_test_category("audio", "generation", config)
        results.append(result)
        metrics.add_result(result)

    if run_e2e:
        logger.info("Running audio E2E tests...")
        result = run_test_category("audio", "e2e", config)
        results.append(result)
        metrics.add_result(result)

    await create_test_artifact(metrics, results, "audio")

    success = metrics.failed == 0 and metrics.errors == 0
    logger.info(f"Audio Pipeline Tests Complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": "audio",
        "success": success,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Run audio pipeline tests")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017")
    parser.add_argument("--llm-endpoint", default="http://localhost:8081")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-storage", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")

    args = parser.parse_args()

    asyncio.run(audio_pipeline_test_flow(
        mongodb_uri=args.mongodb_uri,
        llm_endpoint=args.llm_endpoint,
        timeout_seconds=args.timeout,
        run_storage=not args.skip_storage,
        run_retrieval=not args.skip_retrieval,
        run_generation=not args.skip_generation,
        run_e2e=not args.skip_e2e,
        cleanup_after_test=not args.no_cleanup,
    ))
