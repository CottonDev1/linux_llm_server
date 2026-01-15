"""Consolidated Prefect flows for pipeline testing.

This module provides a single parameterized flow that can run tests for any
combination of pipelines, replacing multiple single-purpose flows.
"""
import subprocess
import time
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
from pathlib import Path
import sys
import re

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

TESTING_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TESTING_ROOT))

from config.settings import settings


# Valid pipeline names
PIPELINE_NAMES = Literal[
    "sql", "audio", "query", "git",
    "code_flow", "code_assistance", "document", "shared"
]

PIPELINE_CONFIG = {
    "sql": {"dir": "sql", "display": "SQL"},
    "audio": {"dir": "audio", "display": "Audio"},
    "query": {"dir": "query", "display": "Query/RAG"},
    "git": {"dir": "git", "display": "Git"},
    "code_flow": {"dir": "code_flow", "display": "Code Flow"},
    "code_assistance": {"dir": "code_assistance", "display": "Code Assistance"},
    "document": {"dir": "document_agent", "display": "Document"},
    "shared": {"dir": "shared", "display": "Shared/Cross-cutting"},
}


@dataclass
class PipelineTestResult:
    """Result from a pipeline test run."""
    pipeline_name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    error_message: str = ""


@task(name="run_pipeline_tests", retries=1, tags=["pipeline", "pytest"])
def run_pipeline_tests(pipeline_name: str, test_dir: str) -> PipelineTestResult:
    """Run pytest for a specific pipeline."""
    logger = get_run_logger()
    start_time = time.time()

    cmd = [
        "python", "-m", "pytest",
        str(TESTING_ROOT / "pipelines" / test_dir),
        "-v", "--tb=short",
        f"--timeout={settings.pytest_timeout}"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.pytest_timeout + 60,
            cwd=str(TESTING_ROOT),
        )

        test_result = PipelineTestResult(
            pipeline_name=pipeline_name,
            exit_code=result.returncode,
            duration_seconds=time.time() - start_time,
        )

        output = result.stdout + result.stderr
        if "passed" in output:
            match = re.search(r'(\d+) passed', output)
            if match:
                test_result.passed = int(match.group(1))
            match = re.search(r'(\d+) failed', output)
            if match:
                test_result.failed = int(match.group(1))
            match = re.search(r'(\d+) skipped', output)
            if match:
                test_result.skipped = int(match.group(1))

        logger.info(f"{pipeline_name}: {test_result.passed} passed, {test_result.failed} failed")
        return test_result

    except subprocess.TimeoutExpired:
        return PipelineTestResult(
            pipeline_name=pipeline_name,
            failed=1,
            error_message="Timeout",
            exit_code=124,
            duration_seconds=time.time() - start_time,
        )
    except Exception as e:
        return PipelineTestResult(
            pipeline_name=pipeline_name,
            failed=1,
            error_message=str(e),
            exit_code=1,
            duration_seconds=time.time() - start_time,
        )


@task(name="create_pipeline_report", log_prints=True)
async def create_pipeline_report(results: List[PipelineTestResult], title: str = "Pipeline Test Results"):
    """Create a markdown artifact with test results."""
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_duration = sum(r.duration_seconds for r in results)

    success_rate = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0

    markdown = f"""# {title}

## Summary
| Metric | Value |
|--------|-------|
| Total Passed | {total_passed} |
| Total Failed | {total_failed} |
| Total Skipped | {total_skipped} |
| Success Rate | {success_rate:.1f}% |
| Duration | {total_duration:.1f}s |

## Pipeline Results

| Pipeline | Passed | Failed | Skipped | Duration | Status |
|----------|--------|--------|---------|----------|--------|
"""

    for r in results:
        status = "PASS" if r.failed == 0 else "FAIL"
        markdown += f"| {r.pipeline_name} | {r.passed} | {r.failed} | {r.skipped} | {r.duration_seconds:.1f}s | {status} |\n"

    if any(r.error_message for r in results):
        markdown += "\n## Errors\n\n"
        for r in results:
            if r.error_message:
                markdown += f"### {r.pipeline_name}\n```\n{r.error_message}\n```\n\n"

    await create_markdown_artifact(
        key="pipeline-test-results",
        markdown=markdown,
        description=f"Test results: {total_passed} passed, {total_failed} failed",
    )


@flow(name="pipeline-tests", log_prints=True)
def pipeline_test_flow(
    pipelines: Optional[List[str]] = None,
    parallel: bool = False,
) -> Dict[str, Any]:
    """
    Run tests for specified pipelines.

    This is the main entry point for pipeline testing. It can run tests for
    any combination of pipelines, either sequentially or in parallel.

    Args:
        pipelines: List of pipeline names to test. If None or empty, runs ALL pipelines.
                   Valid values: sql, audio, query, git, code_flow, code_assistance, document, shared
        parallel: If True, run pipeline tests in parallel (faster but uses more resources)

    Returns:
        Dict with success status and detailed results

    Examples:
        # Run all pipelines
        pipeline_test_flow()

        # Run specific pipelines
        pipeline_test_flow(pipelines=["sql", "query"])

        # Run in parallel for speed
        pipeline_test_flow(pipelines=["sql", "audio", "query"], parallel=True)
    """
    logger = get_run_logger()

    # Determine which pipelines to run
    if not pipelines:
        pipelines_to_run = list(PIPELINE_CONFIG.keys())
    else:
        # Validate pipeline names
        invalid = [p for p in pipelines if p not in PIPELINE_CONFIG]
        if invalid:
            logger.warning(f"Invalid pipeline names ignored: {invalid}")
        pipelines_to_run = [p for p in pipelines if p in PIPELINE_CONFIG]

    if not pipelines_to_run:
        return {"success": False, "error": "No valid pipelines specified"}

    logger.info("=" * 60)
    logger.info(f"Running tests for pipelines: {', '.join(pipelines_to_run)}")
    logger.info("=" * 60)

    results = []

    if parallel:
        # Run tests in parallel using Prefect's task submission
        from prefect import task
        futures = []
        for pipeline in pipelines_to_run:
            config = PIPELINE_CONFIG[pipeline]
            future = run_pipeline_tests.submit(
                f"{config['display']} Pipeline",
                config["dir"]
            )
            futures.append(future)

        # Collect results
        for future in futures:
            results.append(future.result())
    else:
        # Run sequentially
        for pipeline in pipelines_to_run:
            config = PIPELINE_CONFIG[pipeline]
            logger.info(f"Running {config['display']} Pipeline tests...")
            result = run_pipeline_tests(f"{config['display']} Pipeline", config["dir"])
            results.append(result)

    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)

    # Create report artifact
    import asyncio
    asyncio.run(create_pipeline_report(results))

    logger.info("=" * 60)
    logger.info("Pipeline Tests Complete")
    logger.info(f"Total: {total_passed} passed, {total_failed} failed")
    logger.info("=" * 60)

    return {
        "success": total_failed == 0,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "pipelines_tested": pipelines_to_run,
        "results": [
            {
                "pipeline": r.pipeline_name,
                "passed": r.passed,
                "failed": r.failed,
                "skipped": r.skipped,
                "duration": r.duration_seconds,
                "success": r.failed == 0,
            }
            for r in results
        ],
    }


__all__ = [
    "pipeline_test_flow",
    "run_pipeline_tests",
    "PipelineTestResult",
    "PIPELINE_CONFIG",
]
