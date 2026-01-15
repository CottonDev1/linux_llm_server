"""
Prefect Pipeline Test Orchestration Flow

Orchestrates Python pipeline tests with:
1. Individual test suite execution as separate tasks
2. Parallel/sequential test execution
3. Result aggregation and reporting
4. Artifact creation for Prefect dashboard

Pipeline Tests:
- SQL Pipeline: Rules matching, schema context, SQL generation
- Git Pipeline: Repository operations, commit analysis
- RAG Pipeline: Query understanding, retrieval, response generation
- Audio Pipeline: File discovery, transcription, analysis

Run with:
    python prefect_pipelines/pytest_pipeline_flow.py

Or via Prefect:
    prefect deployment run 'pytest-pipeline-flow/default'
"""

import asyncio
import subprocess
import json
import time
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.events import emit_event


# Test suite definitions
PIPELINE_TEST_SUITES = {
    "sql-pipeline": {
        "file": "tests/test_sql_pipeline.py",
        "description": "SQL generation pipeline: rules, schema, generation, validation",
        "category": "sql",
        "has_llm_calls": True,
        "markers": ["sql"]
    },
    "git-pipeline": {
        "file": "tests/test_git_pipeline.py",
        "description": "Git analysis pipeline: repo validation, pull, commit analysis",
        "category": "git",
        "has_llm_calls": False,
        "markers": ["git"]
    },
    "rag-pipeline": {
        "file": "tests/test_rag_pipeline.py",
        "description": "RAG/Knowledge Base pipeline: retrieval, reranking, response generation",
        "category": "rag",
        "has_llm_calls": True,
        "markers": ["rag", "knowledge-base"]
    },
    "audio-pipeline": {
        "file": "tests/test_audio_pipeline.py",
        "description": "Audio pipeline: file discovery, transcription, LLM analysis",
        "category": "audio",
        "has_llm_calls": True,
        "markers": ["audio"]
    },
    "prefect-pipelines": {
        "file": "tests/test_prefect_pipelines.py",
        "description": "Prefect workflow infrastructure tests",
        "category": "infrastructure",
        "has_llm_calls": False,
        "markers": ["prefect"]
    },
    "hybrid-retrieval": {
        "file": "tests/test_sql_hybrid_retrieval.py",
        "description": "SQL hybrid retrieval: RRF algorithm, keyword extraction",
        "category": "sql",
        "has_llm_calls": False,
        "markers": ["sql", "retrieval"]
    }
}

# Project paths
PYTHON_SERVICES_ROOT = Path(__file__).parent.parent
VENV_PYTHON = PYTHON_SERVICES_ROOT / "venv" / "Scripts" / "python.exe"


@dataclass
class PipelineTestResult:
    """Result from a single test suite execution."""
    suite_name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    llm_calls_tracked: int = 0


@dataclass
class PipelineTestSummary:
    """Summary of all pipeline test runs."""
    total_suites: int = 0
    suites_passed: int = 0
    suites_failed: int = 0
    total_tests: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_errors: int = 0
    duration_seconds: float = 0.0
    results: List[PipelineTestResult] = field(default_factory=list)


def parse_pytest_output(stdout: str) -> Dict[str, int]:
    """Parse pytest output to extract test counts."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    # Look for summary line like "5 passed, 2 failed, 1 skipped in 1.23s"
    import re

    # Match patterns like "39 passed" or "2 failed"
    for status in ["passed", "failed", "skipped", "error"]:
        match = re.search(rf"(\d+)\s+{status}", stdout)
        if match:
            key = "errors" if status == "error" else status
            counts[key] = int(match.group(1))

    return counts


@task(
    name="run_pipeline_test_suite",
    description="Run a single Python pipeline test suite",
    retries=1,
    retry_delay_seconds=10,
    tags=["pytest", "pipeline-tests"]
)
async def run_pipeline_test_suite(
    suite_name: str,
    verbose: bool = True,
    timeout: int = 300
) -> PipelineTestResult:
    """
    Run a single pipeline test suite using pytest.

    Args:
        suite_name: Name of the test suite to run
        verbose: Whether to run pytest with verbose output
        timeout: Test timeout in seconds

    Returns:
        PipelineTestResult with execution details
    """
    logger = get_run_logger()
    start_time = time.time()
    result = PipelineTestResult(suite_name=suite_name)

    if suite_name not in PIPELINE_TEST_SUITES:
        result.error_message = f"Unknown test suite: {suite_name}"
        result.exit_code = 1
        return result

    suite_info = PIPELINE_TEST_SUITES[suite_name]
    test_file = PYTHON_SERVICES_ROOT / suite_info["file"]

    if not test_file.exists():
        result.error_message = f"Test file not found: {test_file}"
        result.exit_code = 1
        return result

    logger.info(f"Running test suite: {suite_name}")
    logger.info(f"  Description: {suite_info['description']}")
    logger.info(f"  File: {suite_info['file']}")
    logger.info(f"  LLM calls: {'Yes' if suite_info['has_llm_calls'] else 'No'}")

    try:
        # Build pytest command
        cmd = [
            str(VENV_PYTHON), "-m", "pytest",
            str(test_file),
            "-v" if verbose else "-q",
            f"--timeout={timeout}",
            "--tb=short"
        ]

        # Set environment
        env = os.environ.copy()

        # Run pytest
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PYTHON_SERVICES_ROOT),
            env=env
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout + 60  # Add buffer
        )

        result.stdout = stdout.decode("utf-8", errors="replace")
        result.stderr = stderr.decode("utf-8", errors="replace")
        result.exit_code = process.returncode

        # Parse test counts from output
        counts = parse_pytest_output(result.stdout)
        result.passed = counts["passed"]
        result.failed = counts["failed"]
        result.skipped = counts["skipped"]
        result.errors = counts["errors"]

        # Log result
        status = "PASSED" if result.exit_code == 0 else "FAILED"
        logger.info(f"Suite {suite_name}: {status}")
        logger.info(f"  Tests: {result.passed} passed, {result.failed} failed, {result.skipped} skipped")

        # Emit Prefect event
        emit_event(
            event=f"pipeline-test.{suite_name}.{'completed' if result.exit_code == 0 else 'failed'}",
            resource={"prefect.resource.id": f"pipeline-test.{suite_name}"},
            payload={
                "suite": suite_name,
                "passed": result.passed,
                "failed": result.failed,
                "duration": time.time() - start_time
            }
        )

    except asyncio.TimeoutError:
        result.error_message = f"Test suite timed out after {timeout}s"
        result.exit_code = 124
        result.failed = 1
        logger.error(result.error_message)

    except Exception as e:
        result.error_message = str(e)
        result.exit_code = 1
        result.failed = 1
        logger.error(f"Error running {suite_name}: {e}")

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="create_pipeline_test_artifact",
    description="Create Prefect artifact with pipeline test results",
    tags=["pytest", "reporting"]
)
async def create_pipeline_test_artifact(summary: PipelineTestSummary) -> str:
    """
    Create a markdown artifact with test results for Prefect dashboard.

    Args:
        summary: PipelineTestSummary with all test results

    Returns:
        Markdown report content
    """
    logger = get_run_logger()
    logger.info("Creating pipeline test report artifact...")

    # Calculate pass rate
    pass_rate = (summary.tests_passed / summary.total_tests * 100) if summary.total_tests > 0 else 0

    # Build status
    if summary.tests_failed == 0 and summary.tests_errors == 0:
        status = "PASSED"
        status_color = "green"
    elif summary.tests_passed > summary.tests_failed:
        status = "PARTIAL"
        status_color = "orange"
    else:
        status = "FAILED"
        status_color = "red"

    report = f"""# Pipeline Test Report

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Overall Status: {status}

| Metric | Value |
|--------|-------|
| **Total Test Suites** | {summary.total_suites} |
| **Suites Passed** | {summary.suites_passed} |
| **Suites Failed** | {summary.suites_failed} |
| **Total Tests** | {summary.total_tests} |
| **Tests Passed** | {summary.tests_passed} |
| **Tests Failed** | {summary.tests_failed} |
| **Tests Skipped** | {summary.tests_skipped} |
| **Pass Rate** | {pass_rate:.1f}% |
| **Total Duration** | {summary.duration_seconds:.1f}s |

## Suite Results

| Suite | Category | Status | Passed | Failed | Skipped | Duration |
|-------|----------|--------|--------|--------|---------|----------|
"""

    for result in summary.results:
        suite_info = PIPELINE_TEST_SUITES.get(result.suite_name, {})
        category = suite_info.get("category", "unknown")
        suite_status = "PASS" if result.exit_code == 0 and result.failed == 0 else "FAIL"
        report += f"| {result.suite_name} | {category} | {suite_status} | {result.passed} | {result.failed} | {result.skipped} | {result.duration_seconds:.1f}s |\n"

    # Add failures section if any
    failures = [r for r in summary.results if r.failed > 0 or r.error_message]
    if failures:
        report += "\n## Failures\n\n"
        for result in failures:
            report += f"### {result.suite_name}\n\n"
            if result.error_message:
                report += f"**Error:** {result.error_message}\n\n"
            if result.stderr:
                report += f"```\n{result.stderr[:1000]}{'...' if len(result.stderr) > 1000 else ''}\n```\n\n"

    # Add pipeline descriptions
    report += "\n## Pipeline Test Suites\n\n"
    for name, info in PIPELINE_TEST_SUITES.items():
        llm_indicator = " (LLM)" if info.get("has_llm_calls") else ""
        report += f"- **{name}**{llm_indicator}: {info['description']}\n"

    # Create artifact
    await create_markdown_artifact(
        key="pipeline-test-report",
        markdown=report,
        description="Python Pipeline Test Results"
    )

    logger.info("Pipeline test report artifact created")
    return report


@flow(
    name="pytest-pipeline-flow",
    description="Run Python pipeline tests with Prefect orchestration",
    retries=0
)
async def run_pipeline_tests(
    suites: Optional[List[str]] = None,
    parallel: bool = True,
    verbose: bool = True,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Main flow for running Python pipeline tests.

    Args:
        suites: List of suite names to run (None = all)
        parallel: Whether to run suites in parallel
        verbose: Whether to run pytest with verbose output
        timeout: Test timeout in seconds

    Returns:
        Dict with test results summary
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Pipeline Test Flow")
    logger.info("=" * 60)

    # Determine which suites to run
    if suites:
        suites_to_run = [s for s in suites if s in PIPELINE_TEST_SUITES]
        invalid = set(suites) - set(suites_to_run)
        if invalid:
            logger.warning(f"Unknown suites ignored: {invalid}")
    else:
        suites_to_run = list(PIPELINE_TEST_SUITES.keys())

    logger.info(f"Running {len(suites_to_run)} test suites: {', '.join(suites_to_run)}")
    logger.info(f"Mode: {'parallel' if parallel else 'sequential'}")

    summary = PipelineTestSummary(
        total_suites=len(suites_to_run)
    )

    # Run tests
    if parallel:
        # Run all suites in parallel
        tasks = [
            run_pipeline_test_suite(
                suite,
                verbose=verbose,
                timeout=timeout
            )
            for suite in suites_to_run
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = PipelineTestResult(
                    suite_name=suites_to_run[i],
                    failed=1,
                    error_message=str(result)
                )
                summary.results.append(error_result)
            else:
                summary.results.append(result)
    else:
        # Run suites sequentially
        for suite in suites_to_run:
            result = await run_pipeline_test_suite(
                suite,
                verbose=verbose,
                timeout=timeout
            )
            summary.results.append(result)

    # Aggregate results
    for result in summary.results:
        summary.tests_passed += result.passed
        summary.tests_failed += result.failed
        summary.tests_skipped += result.skipped
        summary.tests_errors += result.errors
        summary.total_tests += result.passed + result.failed + result.skipped

        if result.exit_code == 0 and result.failed == 0:
            summary.suites_passed += 1
        else:
            summary.suites_failed += 1

    summary.duration_seconds = time.time() - start_time

    # Create Prefect artifact
    await create_pipeline_test_artifact(summary)

    # Emit summary event
    emit_event(
        event="pipeline-tests.completed",
        resource={"prefect.resource.id": "pipeline-tests"},
        payload={
            "total_suites": summary.total_suites,
            "suites_passed": summary.suites_passed,
            "suites_failed": summary.suites_failed,
            "total_tests": summary.total_tests,
            "tests_passed": summary.tests_passed,
            "tests_failed": summary.tests_failed,
            "duration": summary.duration_seconds
        }
    )

    # Log summary
    logger.info("=" * 60)
    logger.info("Pipeline Test Flow Complete")
    logger.info(f"Suites: {summary.suites_passed}/{summary.total_suites} passed")
    logger.info(f"Tests: {summary.tests_passed} passed, {summary.tests_failed} failed, {summary.tests_skipped} skipped")
    logger.info(f"Duration: {summary.duration_seconds:.1f}s")
    logger.info("=" * 60)

    return {
        "success": summary.tests_failed == 0 and summary.tests_errors == 0,
        "total_suites": summary.total_suites,
        "suites_passed": summary.suites_passed,
        "suites_failed": summary.suites_failed,
        "total_tests": summary.total_tests,
        "tests_passed": summary.tests_passed,
        "tests_failed": summary.tests_failed,
        "tests_skipped": summary.tests_skipped,
        "duration_seconds": summary.duration_seconds
    }


@flow(
    name="pytest-quick-pipeline",
    description="Quick smoke test of SQL and RAG pipelines"
)
async def run_quick_pipeline_test() -> Dict[str, Any]:
    """Run quick smoke test - SQL and RAG pipelines only."""
    return await run_pipeline_tests(
        suites=["sql-pipeline", "rag-pipeline"],
        parallel=True,
        verbose=True
    )


@flow(
    name="pytest-llm-pipelines",
    description="Run all LLM-calling pipeline tests"
)
async def run_llm_pipeline_tests() -> Dict[str, Any]:
    """Run tests that exercise LLM functionality."""
    llm_suites = [
        name for name, info in PIPELINE_TEST_SUITES.items()
        if info.get("has_llm_calls")
    ]
    return await run_pipeline_tests(
        suites=llm_suites,
        parallel=True,
        verbose=True
    )


@flow(
    name="pytest-full-pipeline",
    description="Run all pipeline tests"
)
async def run_full_pipeline_tests() -> Dict[str, Any]:
    """Run all pipeline tests sequentially with full reporting."""
    return await run_pipeline_tests(
        suites=None,  # All suites
        parallel=False,
        verbose=True
    )


def run_tests_sync(
    suites: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for running pipeline tests.

    Example:
        from prefect_pipelines.pytest_pipeline_flow import run_tests_sync
        result = run_tests_sync(suites=["sql-pipeline"])
    """
    return asyncio.run(run_pipeline_tests(
        suites=suites,
        parallel=parallel
    ))


# Export
__all__ = [
    "run_pipeline_tests",
    "run_quick_pipeline_test",
    "run_llm_pipeline_tests",
    "run_full_pipeline_tests",
    "run_tests_sync",
    "PIPELINE_TEST_SUITES"
]


if __name__ == "__main__":
    # Parse arguments
    suites = None

    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            suites = arg.split(",")

    print("=" * 60)
    print("Pipeline Test Flow")
    print("=" * 60)

    result = run_tests_sync(suites=suites)

    print("\nResult:")
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["success"] else 1)
