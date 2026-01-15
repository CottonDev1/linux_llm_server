"""
Base Test Flow
==============

Base classes and shared components for Prefect test flows.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only
- All configuration via parameters - no hardcoded values
"""

import os
import sys
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from pydantic import BaseModel, Field, field_validator

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prefect_pipelines.test_flows.test_flow_utils import (
    TestStatus,
    TestResult,
    TestMetrics,
    ProgressTracker,
    TestTimer,
    parse_pytest_output,
    create_test_report_markdown,
    create_metrics_from_results,
)


class TestFlowConfig(BaseModel):
    """
    Configuration for test flows.

    All values are configurable via Prefect parameters.
    NO HARDCODED VALUES.
    """

    # MongoDB Configuration
    mongodb_uri: str = Field(
        default="",
        description="MongoDB connection URI (required)"
    )
    mongodb_database: str = Field(
        default="llm_website",
        description="MongoDB database name"
    )

    # Local LLM Endpoints ONLY
    llm_sql_endpoint: str = Field(
        default="http://localhost:8080",
        description="SQL LLM endpoint (llama.cpp port 8080)"
    )
    llm_general_endpoint: str = Field(
        default="http://localhost:8081",
        description="General LLM endpoint (llama.cpp port 8081)"
    )
    llm_code_endpoint: str = Field(
        default="http://localhost:8082",
        description="Code LLM endpoint (llama.cpp port 8082)"
    )

    # Test Execution
    timeout_seconds: int = Field(
        default=300,
        description="Test timeout in seconds"
    )
    cleanup_after_test: bool = Field(
        default=True,
        description="Clean up test data after execution"
    )

    # Test Categories
    run_storage_tests: bool = Field(
        default=True,
        description="Run storage validation tests"
    )
    run_retrieval_tests: bool = Field(
        default=True,
        description="Run retrieval validation tests"
    )
    run_generation_tests: bool = Field(
        default=True,
        description="Run LLM generation tests"
    )
    run_e2e_tests: bool = Field(
        default=True,
        description="Run end-to-end tests"
    )

    @field_validator("llm_sql_endpoint", "llm_general_endpoint", "llm_code_endpoint")
    @classmethod
    def validate_local_endpoint(cls, v: str) -> str:
        """Ensure endpoint is local - no external APIs."""
        if not v.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                f"Only local endpoints allowed. Got: {v}. "
                "External APIs (OpenAI, Anthropic, etc.) are not permitted."
            )
        return v


@task(name="run_pytest_module", log_prints=True)
def run_pytest_module(
    module_path: str,
    config: TestFlowConfig,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run pytest on a specific test module.

    Args:
        module_path: Path to the test module
        config: Test configuration
        timeout: Timeout in seconds

    Returns:
        Dict with test results
    """
    logger = get_run_logger()

    # Set environment variables for tests
    env = os.environ.copy()
    env["MONGODB_URI"] = config.mongodb_uri
    env["MONGODB_DATABASE"] = config.mongodb_database
    env["LLAMACPP_SQL_HOST"] = config.llm_sql_endpoint
    env["LLAMACPP_HOST"] = config.llm_general_endpoint
    env["LLAMACPP_CODE_HOST"] = config.llm_code_endpoint
    env["TEST_CLEANUP"] = str(config.cleanup_after_test).lower()

    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        module_path,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]

    logger.info(f"Running pytest: {' '.join(cmd)}")

    timer = TestTimer()
    timer.start()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )

        duration_ms = timer.stop()

        # Parse output
        output = result.stdout + result.stderr
        logger.info(f"Pytest output:\n{output}")

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": duration_ms,
            "module": module_path,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Pytest timeout after {timeout}s")
        return {
            "success": False,
            "error": f"Timeout after {timeout}s",
            "module": module_path,
            "duration_ms": timer.stop(),
        }
    except Exception as e:
        logger.error(f"Pytest error: {e}")
        return {
            "success": False,
            "error": str(e),
            "module": module_path,
            "duration_ms": timer.stop(),
        }


@task(name="run_test_category", log_prints=True)
def run_test_category(
    pipeline: str,
    category: str,
    config: TestFlowConfig,
) -> TestResult:
    """
    Run tests for a specific pipeline category.

    Args:
        pipeline: Pipeline name (sql, audio, git, etc.)
        category: Test category (storage, retrieval, generation, e2e)
        config: Test configuration

    Returns:
        TestResult for the category
    """
    logger = get_run_logger()

    module_path = f"tests/pipelines/{pipeline}/test_{pipeline}_{category}.py"
    logger.info(f"Running {pipeline}/{category} tests: {module_path}")

    result = run_pytest_module(module_path, config, config.timeout_seconds)

    status = TestStatus.PASSED if result.get("success") else TestStatus.FAILED

    return TestResult(
        name=f"{pipeline}_{category}",
        status=status,
        duration_ms=result.get("duration_ms", 0),
        error=result.get("error") or result.get("stderr", "")[:500] if not result.get("success") else None,
        pipeline=pipeline,
        category=category,
        details={
            "stdout": result.get("stdout", "")[:2000],
            "returncode": result.get("returncode"),
        },
    )


@task(name="create_test_artifact", log_prints=True)
async def create_test_artifact(
    metrics: TestMetrics,
    results: List[TestResult],
    pipeline: str,
):
    """
    Create a Prefect artifact with test results.

    Args:
        metrics: Aggregated test metrics
        results: List of test results
        pipeline: Pipeline name
    """
    report = create_test_report_markdown(
        metrics=metrics,
        results=results,
        title=f"{pipeline.title()} Pipeline Test Report",
    )

    await create_markdown_artifact(
        key=f"{pipeline}-test-results",
        markdown=report,
        description=f"Test results for {pipeline} pipeline",
    )


@flow(name="pipeline_test_base", log_prints=True)
def pipeline_test_base_flow(
    pipeline: str,
    mongodb_uri: str,
    llm_endpoint: str = "http://localhost:8081",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
) -> Dict[str, Any]:
    """
    Base flow for running pipeline tests.

    This is a template flow - specific pipeline flows should call this
    or implement similar logic.

    Args:
        pipeline: Pipeline name
        mongodb_uri: MongoDB connection URI
        llm_endpoint: LLM endpoint URL
        timeout_seconds: Test timeout
        run_storage: Run storage tests
        run_retrieval: Run retrieval tests
        run_generation: Run generation tests
        run_e2e: Run end-to-end tests
        cleanup_after_test: Clean up test data

    Returns:
        Dict with test results and metrics
    """
    logger = get_run_logger()
    logger.info(f"Starting {pipeline} pipeline tests")

    # Validate LLM endpoint is local
    if not llm_endpoint.startswith(("http://localhost", "http://127.0.0.1")):
        raise ValueError(f"Only local LLM endpoints allowed. Got: {llm_endpoint}")

    # Create config
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
    metrics.start_time = datetime.utcnow().isoformat()

    # Run enabled test categories
    categories = []
    if run_storage:
        categories.append("storage")
    if run_retrieval:
        categories.append("retrieval")
    if run_generation:
        categories.append("generation")
    if run_e2e:
        categories.append("e2e")

    for category in categories:
        logger.info(f"Running {pipeline}/{category} tests...")
        result = run_test_category(pipeline, category, config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Result: {result.status.value} ({result.duration_ms}ms)")

    metrics.end_time = datetime.utcnow().isoformat()

    # Create artifact
    import asyncio
    asyncio.run(create_test_artifact(metrics, results, pipeline))

    logger.info(f"Pipeline tests complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": pipeline,
        "success": metrics.failed == 0 and metrics.errors == 0,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }
