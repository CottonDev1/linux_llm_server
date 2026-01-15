"""Combined flow for running all tests."""
from typing import Dict, Any, List, Optional
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

from .pipeline_flows import (
    all_pipelines_test_flow,
    sql_pipeline_test_flow,
    audio_pipeline_test_flow,
    query_pipeline_test_flow,
    git_pipeline_test_flow,
    code_flow_pipeline_test_flow,
    code_assistance_pipeline_test_flow,
    document_pipeline_test_flow,
    run_pipeline_tests,
    PipelineTestResult,
)


@task(name="create_test_report", log_prints=True)
async def create_test_report(results: List[PipelineTestResult], title: str = "Test Results"):
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
        status = "✅" if r.failed == 0 else "❌"
        markdown += f"| {r.pipeline_name} | {r.passed} | {r.failed} | {r.skipped} | {r.duration_seconds:.1f}s | {status} |\n"

    if any(r.error_message for r in results):
        markdown += "\n## Errors\n\n"
        for r in results:
            if r.error_message:
                markdown += f"### {r.pipeline_name}\n```\n{r.error_message}\n```\n\n"

    await create_markdown_artifact(
        key="test-results",
        markdown=markdown,
        description=f"Test results: {total_passed} passed, {total_failed} failed",
    )


@flow(name="api-tests", log_prints=True)
def api_test_flow(
    pipelines: Optional[List[str]] = None,
    include_shared: bool = True,
) -> Dict[str, Any]:
    """
    Run API tests for specified pipelines.

    Args:
        pipelines: List of pipeline names to test. If None, runs all.
                   Options: sql, audio, query, git, code_flow, code_assistance, document_agent
        include_shared: Whether to include shared/cross-cutting tests

    Returns:
        Dict with success status and results
    """
    logger = get_run_logger()

    all_pipeline_dirs = [
        ("sql", "SQL"),
        ("audio", "Audio"),
        ("query", "Query"),
        ("git", "Git"),
        ("code_flow", "Code Flow"),
        ("code_assistance", "Code Assistance"),
        ("document_agent", "Document"),
    ]

    if pipelines:
        pipeline_dirs = [(d, n) for d, n in all_pipeline_dirs if d in pipelines]
    else:
        pipeline_dirs = all_pipeline_dirs

    results = []

    # Run pipeline-specific tests
    for test_dir, name in pipeline_dirs:
        logger.info(f"Running {name} Pipeline API tests...")
        result = run_pipeline_tests(f"{name} Pipeline", test_dir)
        results.append(result)

    # Run shared/cross-cutting tests
    if include_shared:
        logger.info("Running shared/cross-cutting tests...")
        shared_result = run_pipeline_tests("Shared Tests", "shared")
        results.append(shared_result)

    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)

    # Create report artifact
    import asyncio
    asyncio.run(create_test_report(results, "API Test Results"))

    logger.info("=" * 60)
    logger.info("API Tests Complete")
    logger.info(f"Total: {total_passed} passed, {total_failed} failed")
    logger.info("=" * 60)

    return {
        "success": total_failed == 0,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "results": [
            {
                "pipeline": r.pipeline_name,
                "passed": r.passed,
                "failed": r.failed,
                "skipped": r.skipped,
                "duration": r.duration_seconds,
            }
            for r in results
        ],
    }


@flow(name="full-test-suite", log_prints=True)
def full_test_flow(
    run_backend_tests: bool = True,
    pipelines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run complete test suite - all backend API tests.

    Args:
        run_backend_tests: Whether to run backend tests (default True)
        pipelines: Optional list of specific pipelines to test

    Returns:
        Dict with success status and results
    """
    logger = get_run_logger()

    results = {
        "backend": None,
        "success": True,
    }

    if run_backend_tests:
        logger.info("Running backend API tests...")
        if pipelines:
            results["backend"] = api_test_flow(pipelines=pipelines)
        else:
            results["backend"] = all_pipelines_test_flow()

        if not results["backend"]["success"]:
            results["success"] = False

    return results


@flow(name="quick-smoke-test", log_prints=True)
def quick_smoke_test_flow() -> Dict[str, Any]:
    """
    Run a quick smoke test - SQL and Query pipelines only.
    Good for validating basic functionality.
    """
    logger = get_run_logger()
    logger.info("Running quick smoke test...")

    return api_test_flow(pipelines=["sql", "query"], include_shared=False)


__all__ = [
    "api_test_flow",
    "full_test_flow",
    "quick_smoke_test_flow",
    "create_test_report",
]
