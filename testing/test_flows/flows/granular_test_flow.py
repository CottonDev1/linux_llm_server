"""Prefect flow for granular test execution with fine-grained control.

This flow allows running specific tests with pytest patterns, markers, and paths.
Provides detailed output parsing and artifact creation for test results.
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
class GranularTestResult:
    """Result from a granular test execution."""
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


@task(name="build_pytest_command", tags=["pytest", "command"])
def build_pytest_command(
    test_path: Optional[str] = None,
    test_pattern: Optional[str] = None,
    marker: Optional[str] = None,
    verbose: bool = True,
    timeout: int = 300,
) -> List[str]:
    """
    Build the pytest command with all specified options.

    Args:
        test_path: Test file/directory relative to testing/pipelines/
        test_pattern: pytest -k pattern for test selection
        marker: pytest -m marker expression
        verbose: Whether to use verbose output
        timeout: Timeout for each test in seconds

    Returns:
        List of command arguments for subprocess
    """
    logger = get_run_logger()

    cmd = ["python", "-m", "pytest"]

    # Determine test path
    if test_path:
        full_path = TESTING_ROOT / "pipelines" / test_path
        if not full_path.exists():
            logger.warning(f"Test path does not exist: {full_path}")
        cmd.append(str(full_path))
    else:
        # Default to all pipeline tests
        cmd.append(str(TESTING_ROOT / "pipelines"))

    # Add verbosity
    if verbose:
        cmd.append("-v")

    # Add short traceback
    cmd.append("--tb=short")

    # Add timeout
    cmd.append(f"--timeout={timeout}")

    # Add pattern filter (-k)
    if test_pattern:
        cmd.extend(["-k", test_pattern])

    # Add marker filter (-m)
    if marker:
        cmd.extend(["-m", marker])

    logger.info(f"Built pytest command: {' '.join(cmd)}")
    return cmd


@task(name="parse_pytest_output", tags=["pytest", "parsing"])
def parse_pytest_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Parse pytest output to extract test counts and individual test results.

    Args:
        stdout: Standard output from pytest
        stderr: Standard error from pytest

    Returns:
        Dictionary with parsed test results
    """
    logger = get_run_logger()
    output = stdout + stderr

    result = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "passed_tests": [],
        "failed_tests": [],
        "skipped_tests": [],
        "error_tests": [],
    }

    # Parse summary line (e.g., "5 passed, 2 failed, 1 skipped in 10.5s")
    summary_match = re.search(
        r'=+\s*(.*?)\s*=+\s*$',
        output,
        re.MULTILINE
    )
    if summary_match:
        summary = summary_match.group(1)

        passed_match = re.search(r'(\d+)\s+passed', summary)
        if passed_match:
            result["passed"] = int(passed_match.group(1))

        failed_match = re.search(r'(\d+)\s+failed', summary)
        if failed_match:
            result["failed"] = int(failed_match.group(1))

        skipped_match = re.search(r'(\d+)\s+skipped', summary)
        if skipped_match:
            result["skipped"] = int(skipped_match.group(1))

        error_match = re.search(r'(\d+)\s+error', summary)
        if error_match:
            result["errors"] = int(error_match.group(1))

    # Parse individual test results from verbose output
    # Pattern: test_file.py::test_name PASSED/FAILED/SKIPPED
    test_patterns = [
        (r'(\S+::\S+)\s+PASSED', "passed_tests"),
        (r'(\S+::\S+)\s+FAILED', "failed_tests"),
        (r'(\S+::\S+)\s+SKIPPED', "skipped_tests"),
        (r'(\S+::\S+)\s+ERROR', "error_tests"),
    ]

    for pattern, key in test_patterns:
        matches = re.findall(pattern, output)
        result[key] = matches

    logger.info(
        f"Parsed results: {result['passed']} passed, "
        f"{result['failed']} failed, {result['skipped']} skipped"
    )

    return result


@task(name="create_granular_test_artifact", tags=["artifact", "report"])
async def create_granular_test_artifact(
    result: GranularTestResult,
    title: str = "Granular Test Results",
) -> None:
    """
    Create a markdown artifact with detailed test results.

    Args:
        result: GranularTestResult from the test run
        title: Title for the artifact
    """
    total_tests = result.passed + result.failed + result.skipped + result.errors
    success_rate = (
        (result.passed / total_tests * 100) if total_tests > 0 else 0
    )

    # Build filter description
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
| Exit Code | {result.exit_code} |

## Status
{"PASSED" if result.exit_code == 0 else "FAILED"}
"""

    # Add failed tests section if any
    if result.failed_tests:
        markdown += "\n## Failed Tests\n"
        for test in result.failed_tests[:20]:  # Limit to 20
            markdown += f"- `{test}`\n"
        if len(result.failed_tests) > 20:
            markdown += f"\n... and {len(result.failed_tests) - 20} more\n"

    # Add passed tests section (collapsed by default in many viewers)
    if result.passed_tests and len(result.passed_tests) <= 50:
        markdown += "\n## Passed Tests\n"
        for test in result.passed_tests:
            markdown += f"- `{test}`\n"
    elif result.passed_tests:
        markdown += f"\n## Passed Tests\n{len(result.passed_tests)} tests passed (list truncated)\n"

    # Add error message if present
    if result.error_message:
        markdown += f"\n## Error\n```\n{result.error_message}\n```\n"

    await create_markdown_artifact(
        key="granular-test-results",
        markdown=markdown,
        description=f"Test results: {result.passed} passed, {result.failed} failed",
    )


@task(name="execute_pytest", retries=1, tags=["pytest", "execution"])
def execute_pytest(
    cmd: List[str],
    timeout: int = 300,
) -> GranularTestResult:
    """
    Execute the pytest command and capture results.

    Args:
        cmd: List of command arguments
        timeout: Maximum execution time in seconds

    Returns:
        GranularTestResult with execution details
    """
    logger = get_run_logger()
    start_time = time.time()

    try:
        logger.info(f"Executing: {' '.join(cmd)}")

        process_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Buffer for cleanup
            cwd=str(TESTING_ROOT),
        )

        duration = time.time() - start_time

        # Parse the output
        parsed = parse_pytest_output(
            process_result.stdout,
            process_result.stderr
        )

        result = GranularTestResult(
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

        logger.info(
            f"Test execution completed: {result.passed} passed, "
            f"{result.failed} failed in {duration:.2f}s"
        )

        return result

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        logger.error(f"Test execution timed out after {timeout}s")
        return GranularTestResult(
            failed=1,
            error_message=f"Timeout after {timeout} seconds",
            exit_code=124,
            duration_seconds=duration,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Test execution failed: {e}")
        return GranularTestResult(
            failed=1,
            error_message=str(e),
            exit_code=1,
            duration_seconds=duration,
        )


@flow(name="run-specific-tests", log_prints=True)
def run_specific_tests(
    test_path: Optional[str] = None,
    test_pattern: Optional[str] = None,
    marker: Optional[str] = None,
    verbose: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run specific tests with granular control over test selection.

    This flow provides fine-grained control over which tests to run,
    supporting path-based selection, pattern matching (-k), and
    marker filtering (-m).

    Args:
        test_path: Test file/directory relative to testing/pipelines/
                   Example: "sql/test_query.py" or "audio"
        test_pattern: pytest -k pattern for test selection
                      Example: "test_query and not slow"
        marker: pytest -m marker expression
                Example: "smoke" or "not integration"
        verbose: Whether to use verbose output (default: True)
        timeout: Timeout for test execution in seconds (default: 300)

    Returns:
        Dict containing:
            - success: bool indicating if all tests passed
            - passed: number of passed tests
            - failed: number of failed tests
            - skipped: number of skipped tests
            - duration: execution time in seconds
            - result: GranularTestResult object with full details

    Examples:
        # Run a specific test file
        run_specific_tests(test_path="sql/test_rules.py")

        # Run tests matching a pattern
        run_specific_tests(test_pattern="test_query")

        # Run tests with a marker
        run_specific_tests(marker="smoke")

        # Combine filters
        run_specific_tests(
            test_path="sql",
            test_pattern="test_query",
            marker="not slow"
        )
    """
    logger = get_run_logger()

    logger.info("=" * 60)
    logger.info("Starting Granular Test Execution")
    logger.info("=" * 60)

    if test_path:
        logger.info(f"Test path: {test_path}")
    if test_pattern:
        logger.info(f"Test pattern (-k): {test_pattern}")
    if marker:
        logger.info(f"Marker (-m): {marker}")
    logger.info(f"Timeout: {timeout}s")

    # Build the pytest command
    cmd = build_pytest_command(
        test_path=test_path,
        test_pattern=test_pattern,
        marker=marker,
        verbose=verbose,
        timeout=timeout,
    )

    # Execute pytest
    result = execute_pytest(cmd, timeout=timeout)

    # Store filter info in result for artifact creation
    result.test_path = test_path
    result.test_pattern = test_pattern
    result.marker = marker

    # Create artifact
    import asyncio
    asyncio.run(create_granular_test_artifact(result))

    # Log final summary
    logger.info("=" * 60)
    logger.info("Test Execution Complete")
    logger.info(f"Passed: {result.passed}")
    logger.info(f"Failed: {result.failed}")
    logger.info(f"Skipped: {result.skipped}")
    logger.info(f"Duration: {result.duration_seconds:.2f}s")
    logger.info(f"Exit code: {result.exit_code}")
    logger.info("=" * 60)

    return {
        "success": result.exit_code == 0,
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "errors": result.errors,
        "duration": result.duration_seconds,
        "exit_code": result.exit_code,
        "failed_tests": result.failed_tests,
        "result": result,
    }


@flow(name="run-tests-by-marker", log_prints=True)
def run_tests_by_marker(
    marker: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Convenience flow to run tests filtered by a specific marker.

    Args:
        marker: pytest marker expression (e.g., "smoke", "integration", "slow")
        timeout: Timeout for test execution in seconds

    Returns:
        Dict with test results
    """
    return run_specific_tests(marker=marker, timeout=timeout)


@flow(name="run-tests-by-pattern", log_prints=True)
def run_tests_by_pattern(
    pattern: str,
    test_path: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Convenience flow to run tests filtered by a name pattern.

    Args:
        pattern: pytest -k pattern (e.g., "test_query", "test_audio and not slow")
        test_path: Optional path to narrow scope
        timeout: Timeout for test execution in seconds

    Returns:
        Dict with test results
    """
    return run_specific_tests(
        test_path=test_path,
        test_pattern=pattern,
        timeout=timeout,
    )


__all__ = [
    "run_specific_tests",
    "run_tests_by_marker",
    "run_tests_by_pattern",
    "GranularTestResult",
    "build_pytest_command",
    "parse_pytest_output",
    "execute_pytest",
    "create_granular_test_artifact",
]
