"""
Prefect Playwright Test Orchestration Flow

Orchestrates all Playwright E2E tests with:
1. Individual test suite execution as separate tasks
2. Parallel test execution when possible
3. Result aggregation and reporting
4. Allure report generation
5. Artifact creation for Prefect dashboard

Features:
- Run all tests or specific test suites
- Configurable parallelism
- Automatic retry on flaky tests
- Rich reporting via Prefect artifacts
"""

import asyncio
import subprocess
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# Test suite definitions
TEST_SUITES = {
    "admin-pages": {
        "file": "admin-pages.spec.js",
        "description": "Admin page loading and RBAC tests",
        "category": "admin"
    },
    "sidebar-links": {
        "file": "sidebar-links.spec.js",
        "description": "Sidebar navigation and link verification tests",
        "category": "admin"
    },
    "user-management": {
        "file": "user-management.spec.js",
        "description": "User creation, monitoring access, and staff dashboard tests",
        "category": "admin"
    },
    "sql-query": {
        "file": "sql-query.spec.js",
        "description": "SQL query generation tests",
        "category": "sql"
    },
    "sql-agent": {
        "file": "sql-agent.spec.js",
        "description": "SQL agent tests",
        "category": "sql"
    },
    "audio-bulk": {
        "file": "audio-bulk.spec.js",
        "description": "Bulk audio processing tests (uses bundled fixtures, max 2 files)",
        "category": "audio"
    },
    "audio-single": {
        "file": "audio-single.spec.js",
        "description": "Single audio file processing tests (uses bundled fixtures)",
        "category": "audio"
    },
    "audio-analysis": {
        "file": "audio-analysis.spec.js",
        "description": "Audio analysis workflow tests (staff monitoring, search, view/edit)",
        "category": "audio"
    },
    "knowledge-base-chat": {
        "file": "knowledge-base-chat.spec.js",
        "description": "Knowledge base chat interface tests",
        "category": "knowledge-base"
    },
    "document-agent": {
        "file": "document-agent.spec.js",
        "description": "Document agent tests",
        "category": "documents"
    },
    "git-analysis": {
        "file": "git-analysis.spec.js",
        "description": "Git analysis tests",
        "category": "git"
    },
    "semantic-ticket-match": {
        "file": "semantic-ticket-match.spec.js",
        "description": "Semantic ticket matching tests",
        "category": "tickets"
    },
    "cotton-provider-doc": {
        "file": "cotton-provider-doc.spec.js",
        "description": "Cotton Provider document Q&A validation tests",
        "category": "knowledge-base"
    }
}

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class TestResult:
    """Result from a single test suite execution"""
    suite_name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class TestRunSummary:
    """Summary of all test runs"""
    total_suites: int = 0
    suites_passed: int = 0
    suites_failed: int = 0
    total_tests: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    duration_seconds: float = 0.0
    results: List[TestResult] = field(default_factory=list)


def check_env_requirements(suite_info: dict) -> tuple[bool, str]:
    """Check if required environment variables are set for a test suite."""
    required = suite_info.get("requires_env", [])
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    return True, ""


@task(
    name="run_playwright_suite",
    description="Run a single Playwright test suite",
    retries=1,
    retry_delay_seconds=30,
    tags=["playwright", "e2e-tests"]
)
async def run_playwright_suite(
    suite_name: str,
    headless: bool = True,
    timeout: int = 300000
) -> TestResult:
    """
    Run a single Playwright test suite.

    Args:
        suite_name: Name of the test suite to run
        headless: Whether to run in headless mode
        timeout: Test timeout in milliseconds

    Returns:
        TestResult with execution details
    """
    logger = get_run_logger()
    start_time = time.time()
    result = TestResult(suite_name=suite_name)

    if suite_name not in TEST_SUITES:
        result.error_message = f"Unknown test suite: {suite_name}"
        result.exit_code = 1
        return result

    suite_info = TEST_SUITES[suite_name]

    # Check environment requirements
    env_ok, env_error = check_env_requirements(suite_info)
    if not env_ok:
        logger.warning(f"Skipping {suite_name}: {env_error}")
        result.skipped = 1
        result.error_message = env_error
        result.duration_seconds = time.time() - start_time
        return result

    logger.info(f"Running test suite: {suite_name} ({suite_info['description']})")

    try:
        # Build command
        cmd = [
            "npx", "playwright", "test",
            suite_info["file"],
            "--reporter=json",
            f"--timeout={timeout}"
        ]

        if headless:
            cmd.append("--headed=false")

        # Set environment
        env = os.environ.copy()
        env["HEADLESS"] = "true" if headless else "false"

        # Run test
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=env
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout / 1000 + 60  # Add buffer
        )

        result.stdout = stdout.decode("utf-8", errors="replace")
        result.stderr = stderr.decode("utf-8", errors="replace")
        result.exit_code = process.returncode

        # Parse JSON output
        try:
            # Find JSON in output (Playwright outputs JSON followed by summary)
            json_start = result.stdout.find("{")
            json_end = result.stdout.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result.stdout[json_start:json_end]
                report = json.loads(json_str)

                # Extract test counts from Playwright JSON report
                if "suites" in report:
                    for suite in report.get("suites", []):
                        for spec in suite.get("specs", []):
                            for test in spec.get("tests", []):
                                status = test.get("status", "")
                                if status == "expected":
                                    result.passed += 1
                                elif status == "unexpected":
                                    result.failed += 1
                                elif status == "skipped":
                                    result.skipped += 1
        except json.JSONDecodeError:
            # Fallback: parse from exit code
            if result.exit_code == 0:
                result.passed = 1
            else:
                result.failed = 1

        if result.exit_code != 0 and not result.failed:
            result.failed = 1

        logger.info(f"Suite {suite_name}: {result.passed} passed, {result.failed} failed, {result.skipped} skipped")

    except asyncio.TimeoutError:
        result.error_message = f"Test suite timed out after {timeout/1000}s"
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
    name="generate_allure_report",
    description="Generate Allure test report",
    retries=1,
    tags=["playwright", "reporting"]
)
async def generate_allure_report() -> bool:
    """
    Generate Allure report from test results.

    Returns:
        True if report generation succeeded
    """
    logger = get_run_logger()
    logger.info("Generating Allure report...")

    try:
        # Generate report
        process = await asyncio.create_subprocess_exec(
            "npx", "allure", "generate", "allure-results",
            "--clean", "-o", "allure-report",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT)
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info("Allure report generated successfully")
            return True
        else:
            logger.warning(f"Allure report generation failed: {stderr.decode()}")
            return False

    except Exception as e:
        logger.error(f"Error generating Allure report: {e}")
        return False


@task(
    name="create_test_report_artifact",
    description="Create Prefect artifact with test results",
    tags=["playwright", "reporting"]
)
async def create_test_report_artifact(summary: TestRunSummary) -> str:
    """
    Create a markdown artifact with test results for Prefect dashboard.

    Args:
        summary: TestRunSummary with all test results

    Returns:
        Markdown report content
    """
    logger = get_run_logger()
    logger.info("Creating test report artifact...")

    # Calculate pass rate
    pass_rate = (summary.tests_passed / summary.total_tests * 100) if summary.total_tests > 0 else 0

    # Build status emoji
    if summary.tests_failed == 0:
        status_emoji = "PASSED"
        status_color = "green"
    elif summary.tests_passed > summary.tests_failed:
        status_emoji = "PARTIAL"
        status_color = "orange"
    else:
        status_emoji = "FAILED"
        status_color = "red"

    report = f"""# Playwright E2E Test Report

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Overall Status: {status_emoji}

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

| Suite | Status | Passed | Failed | Skipped | Duration |
|-------|--------|--------|--------|---------|----------|
"""

    for result in summary.results:
        suite_status = "PASS" if result.exit_code == 0 and result.failed == 0 else "FAIL" if result.failed > 0 else "SKIP"
        report += f"| {result.suite_name} | {suite_status} | {result.passed} | {result.failed} | {result.skipped} | {result.duration_seconds:.1f}s |\n"

    # Add failures section if any
    failures = [r for r in summary.results if r.failed > 0 or r.error_message]
    if failures:
        report += "\n## Failures\n\n"
        for result in failures:
            report += f"### {result.suite_name}\n\n"
            if result.error_message:
                report += f"**Error:** {result.error_message}\n\n"
            if result.stderr:
                report += f"```\n{result.stderr[:500]}{'...' if len(result.stderr) > 500 else ''}\n```\n\n"

    # Add links
    report += """
## Reports

- [Allure Report](/allure-report/index.html)
- [Prefect Dashboard](http://localhost:4200)

## Test Suites Included

"""

    for name, info in TEST_SUITES.items():
        report += f"- **{name}**: {info['description']}\n"

    # Create artifact
    await create_markdown_artifact(
        key="playwright-test-report",
        markdown=report,
        description="Playwright E2E Test Results"
    )

    logger.info("Test report artifact created")
    return report


@flow(
    name="playwright_test_flow",
    description="Run all Playwright E2E tests",
    retries=0
)
async def run_playwright_tests(
    suites: Optional[List[str]] = None,
    parallel: bool = False,
    headless: bool = True,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Main flow for running Playwright E2E tests.

    Args:
        suites: List of suite names to run (None = all)
        parallel: Whether to run suites in parallel
        headless: Whether to run in headless mode
        generate_report: Whether to generate Allure report

    Returns:
        Dict with test results summary
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting Playwright E2E Test Flow")
    logger.info("=" * 60)

    # Determine which suites to run
    if suites:
        suites_to_run = [s for s in suites if s in TEST_SUITES]
        invalid = set(suites) - set(suites_to_run)
        if invalid:
            logger.warning(f"Unknown suites ignored: {invalid}")
    else:
        suites_to_run = list(TEST_SUITES.keys())

    logger.info(f"Running {len(suites_to_run)} test suites: {', '.join(suites_to_run)}")
    logger.info(f"Mode: {'parallel' if parallel else 'sequential'}, Headless: {headless}")

    summary = TestRunSummary(total_suites=len(suites_to_run))

    # Run tests
    if parallel:
        # Run all suites in parallel
        tasks = [
            run_playwright_suite(suite, headless=headless)
            for suite in suites_to_run
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                error_result = TestResult(
                    suite_name="unknown",
                    failed=1,
                    error_message=str(result)
                )
                summary.results.append(error_result)
            else:
                summary.results.append(result)
    else:
        # Run suites sequentially
        for suite in suites_to_run:
            result = await run_playwright_suite(suite, headless=headless)
            summary.results.append(result)

    # Aggregate results
    for result in summary.results:
        summary.tests_passed += result.passed
        summary.tests_failed += result.failed
        summary.tests_skipped += result.skipped
        summary.total_tests += result.passed + result.failed + result.skipped

        if result.exit_code == 0 and result.failed == 0:
            summary.suites_passed += 1
        elif result.skipped > 0 and result.passed == 0 and result.failed == 0:
            pass  # Skipped suite
        else:
            summary.suites_failed += 1

    summary.duration_seconds = time.time() - start_time

    # Generate reports
    if generate_report:
        await generate_allure_report()

    # Create Prefect artifact
    await create_test_report_artifact(summary)

    # Log summary
    logger.info("=" * 60)
    logger.info("Test Flow Complete")
    logger.info(f"Suites: {summary.suites_passed}/{summary.total_suites} passed")
    logger.info(f"Tests: {summary.tests_passed} passed, {summary.tests_failed} failed, {summary.tests_skipped} skipped")
    logger.info(f"Duration: {summary.duration_seconds:.1f}s")
    logger.info("=" * 60)

    return {
        "success": summary.tests_failed == 0,
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
    name="playwright_quick_test",
    description="Run a quick smoke test of core functionality"
)
async def run_quick_test() -> Dict[str, Any]:
    """
    Run a quick smoke test - admin pages, sidebar links, and knowledge base.
    """
    return await run_playwright_tests(
        suites=["admin-pages", "sidebar-links", "knowledge-base-chat"],
        parallel=True,
        headless=True,
        generate_report=False
    )


@flow(
    name="playwright_user_management_test",
    description="Run user management and staff dashboard E2E tests"
)
async def run_user_management_test() -> Dict[str, Any]:
    """
    Run user management tests including:
    - User creation with email and non-active status
    - Monitoring category assignment
    - Staff dashboard card verification
    """
    return await run_playwright_tests(
        suites=["user-management"],
        parallel=False,
        headless=True,
        generate_report=True
    )


@flow(
    name="playwright_full_test",
    description="Run full E2E test suite"
)
async def run_full_test() -> Dict[str, Any]:
    """
    Run the full E2E test suite sequentially with all reports.
    """
    return await run_playwright_tests(
        suites=None,  # All suites
        parallel=False,
        headless=True,
        generate_report=True
    )


@flow(
    name="playwright_audio_test",
    description="Run audio processing E2E tests (single + bulk + analysis workflow)"
)
async def run_audio_test() -> Dict[str, Any]:
    """
    Run audio processing tests - single file, bulk processing, and analysis workflow.
    Uses bundled test fixtures in tests/fixtures/audio/.
    Bulk test is limited to 2 files for faster execution.
    """
    return await run_playwright_tests(
        suites=["audio-single", "audio-bulk", "audio-analysis"],
        parallel=False,  # Run sequentially to avoid resource contention
        headless=True,
        generate_report=True
    )


@flow(
    name="playwright_audio_analysis_test",
    description="Run audio analysis workflow E2E tests"
)
async def run_audio_analysis_test() -> Dict[str, Any]:
    """
    Run audio analysis workflow tests only - staff monitoring, search, view/edit.
    Tests the ProcessAudio pages and functionality.
    """
    return await run_playwright_tests(
        suites=["audio-analysis"],
        parallel=False,
        headless=True,
        generate_report=True
    )


def run_tests_sync(
    suites: Optional[List[str]] = None,
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Synchronous wrapper for running tests.

    Example:
        from prefect_pipelines.playwright_test_flow import run_tests_sync
        result = run_tests_sync(suites=["admin-pages"])
    """
    return asyncio.run(run_playwright_tests(
        suites=suites,
        parallel=parallel
    ))


# Export
__all__ = [
    "run_playwright_tests",
    "run_quick_test",
    "run_full_test",
    "run_audio_test",
    "run_audio_analysis_test",
    "run_user_management_test",
    "run_tests_sync",
    "TEST_SUITES"
]


if __name__ == "__main__":
    import sys

    # Parse arguments
    suites = None
    if len(sys.argv) > 1:
        suites = sys.argv[1].split(",")

    print("=" * 60)
    print("Playwright Test Flow")
    print("=" * 60)

    result = run_tests_sync(suites=suites)

    print("\nResult:")
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["success"] else 1)
