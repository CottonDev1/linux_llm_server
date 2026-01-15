"""Custom test flow for fine-grained test execution.

This flow provides flexible test selection using pytest patterns, markers, and paths.
Use this when you need more control than the standard pipeline flows offer.
"""
import subprocess
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

TESTING_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TESTING_ROOT))

from config.settings import settings


@dataclass
class CustomTestResult:
    """Result from a custom test execution."""
    test_path: Optional[str] = None
    test_pattern: Optional[str] = None
    marker: Optional[str] = None
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    failed_tests: List[str] = field(default_factory=list)
    passed_tests: List[str] = field(default_factory=list)


@task(name="build_pytest_command", tags=["pytest"])
def build_pytest_command(
    test_path: Optional[str] = None,
    test_pattern: Optional[str] = None,
    marker: Optional[str] = None,
    verbose: bool = True,
    timeout: int = 300,
) -> List[str]:
    """Build the pytest command with specified options."""
    logger = get_run_logger()

    cmd = ["python", "-m", "pytest"]

    if test_path:
        full_path = TESTING_ROOT / "pipelines" / test_path
        if not full_path.exists():
            logger.warning(f"Test path does not exist: {full_path}")
        cmd.append(str(full_path))
    else:
        cmd.append(str(TESTING_ROOT / "pipelines"))

    if verbose:
        cmd.append("-v")
    cmd.append("--tb=short")
    cmd.append(f"--timeout={timeout}")

    if test_pattern:
        cmd.extend(["-k", test_pattern])
    if marker:
        cmd.extend(["-m", marker])

    logger.info(f"Built pytest command: {' '.join(cmd)}")
    return cmd


@task(name="parse_pytest_output", tags=["pytest"])
def parse_pytest_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """Parse pytest output to extract test counts and results."""
    output = stdout + stderr

    result = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "passed_tests": [],
        "failed_tests": [],
    }

    # Parse summary line
    summary_match = re.search(r'=+\s*(.*?)\s*=+\s*$', output, re.MULTILINE)
    if summary_match:
        summary = summary_match.group(1)
        for metric, pattern in [("passed", r'(\d+)\s+passed'),
                                 ("failed", r'(\d+)\s+failed'),
                                 ("skipped", r'(\d+)\s+skipped'),
                                 ("errors", r'(\d+)\s+error')]:
            match = re.search(pattern, summary)
            if match:
                result[metric] = int(match.group(1))

    # Parse individual test results
    result["passed_tests"] = re.findall(r'(\S+::\S+)\s+PASSED', output)
    result["failed_tests"] = re.findall(r'(\S+::\S+)\s+FAILED', output)

    return result


@task(name="create_custom_test_artifact", tags=["artifact"])
async def create_custom_test_artifact(result: CustomTestResult, title: str = "Custom Test Results"):
    """Create a markdown artifact with detailed test results."""
    total_tests = result.passed + result.failed + result.skipped + result.errors
    success_rate = (result.passed / total_tests * 100) if total_tests > 0 else 0

    filters = []
    if result.test_path:
        filters.append(f"Path: `{result.test_path}`")
    if result.test_pattern:
        filters.append(f"Pattern (-k): `{result.test_pattern}`")
    if result.marker:
        filters.append(f"Marker (-m): `{result.marker}`")

    filter_text = "\n".join(f"- {f}" for f in filters) if filters else "- None (all tests)"

    markdown = f"""# {title}

## Test Selection
{filter_text}

## Summary
| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Passed | {result.passed} |
| Failed | {result.failed} |
| Skipped | {result.skipped} |
| Errors | {result.errors} |
| Success Rate | {success_rate:.1f}% |
| Duration | {result.duration_seconds:.2f}s |

## Status
{"PASSED" if result.exit_code == 0 else "FAILED"}
"""

    if result.failed_tests:
        markdown += "\n## Failed Tests\n"
        for test in result.failed_tests[:20]:
            markdown += f"- `{test}`\n"
        if len(result.failed_tests) > 20:
            markdown += f"\n... and {len(result.failed_tests) - 20} more\n"

    if result.error_message:
        markdown += f"\n## Error\n```\n{result.error_message}\n```\n"

    await create_markdown_artifact(
        key="custom-test-results",
        markdown=markdown,
        description=f"Test results: {result.passed} passed, {result.failed} failed",
    )


@task(name="execute_pytest", retries=1, tags=["pytest"])
def execute_pytest(cmd: List[str], timeout: int = 300) -> CustomTestResult:
    """Execute the pytest command and capture results."""
    logger = get_run_logger()
    start_time = time.time()

    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        process_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
            cwd=str(TESTING_ROOT),
        )

        duration = time.time() - start_time
        parsed = parse_pytest_output(process_result.stdout, process_result.stderr)

        result = CustomTestResult(
            passed=parsed["passed"],
            failed=parsed["failed"],
            skipped=parsed["skipped"],
            errors=parsed["errors"],
            duration_seconds=duration,
            exit_code=process_result.returncode,
            stdout=process_result.stdout,
            stderr=process_result.stderr,
            passed_tests=parsed["passed_tests"],
            failed_tests=parsed["failed_tests"],
        )

        logger.info(f"Completed: {result.passed} passed, {result.failed} failed in {duration:.2f}s")
        return result

    except subprocess.TimeoutExpired as e:
        return CustomTestResult(
            failed=1,
            error_message=f"Timeout after {timeout} seconds",
            exit_code=124,
            duration_seconds=time.time() - start_time,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        )
    except Exception as e:
        return CustomTestResult(
            failed=1,
            error_message=str(e),
            exit_code=1,
            duration_seconds=time.time() - start_time,
        )


@flow(name="custom-tests", log_prints=True)
def custom_test_flow(
    test_path: Optional[str] = None,
    pattern: Optional[str] = None,
    marker: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run tests with custom filters for fine-grained control.

    Use this flow when you need to run a specific subset of tests that
    doesn't fit the standard pipeline groupings.

    Args:
        test_path: Test file/directory relative to testing/pipelines/
                   Example: "sql/test_sql_generation.py" or "audio"
        pattern: pytest -k pattern for test name filtering
                 Example: "test_query and not slow"
        marker: pytest -m marker expression
                Example: "smoke" or "not integration"
        timeout: Timeout per test in seconds (default: 300)

    Returns:
        Dict with success status and detailed results

    Examples:
        # Run a specific test file
        custom_test_flow(test_path="sql/test_sql_generation.py")

        # Run tests matching a name pattern
        custom_test_flow(pattern="test_embedding")

        # Run tests with a specific marker
        custom_test_flow(marker="smoke")

        # Combine filters
        custom_test_flow(test_path="sql", pattern="test_query", marker="not slow")
    """
    logger = get_run_logger()

    logger.info("=" * 60)
    logger.info("Starting Custom Test Execution")
    if test_path:
        logger.info(f"  Path: {test_path}")
    if pattern:
        logger.info(f"  Pattern (-k): {pattern}")
    if marker:
        logger.info(f"  Marker (-m): {marker}")
    logger.info("=" * 60)

    cmd = build_pytest_command(
        test_path=test_path,
        test_pattern=pattern,
        marker=marker,
        verbose=True,
        timeout=timeout,
    )

    result = execute_pytest(cmd, timeout=timeout)
    result.test_path = test_path
    result.test_pattern = pattern
    result.marker = marker

    import asyncio
    asyncio.run(create_custom_test_artifact(result))

    logger.info("=" * 60)
    logger.info(f"Complete: {result.passed} passed, {result.failed} failed")
    logger.info("=" * 60)

    return {
        "success": result.exit_code == 0,
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "errors": result.errors,
        "duration": result.duration_seconds,
        "failed_tests": result.failed_tests,
    }


__all__ = [
    "custom_test_flow",
    "CustomTestResult",
]
