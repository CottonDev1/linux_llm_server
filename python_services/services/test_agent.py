"""
Test Agent - Playwright Test Generation and Execution
=====================================================

An agentic test agent that can:
1. Generate Playwright tests for pipeline stages
2. Execute tests with self-correction (MAGIC pattern)
3. Start required services via Environment Agent
4. Record webstack errors for analysis
5. Delegate code fixes to Code Agent

This agent uses conversation to determine what pipeline stages
need testing and generates comprehensive E2E tests.
"""

import asyncio
import logging
import os
import re
import json
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of a test execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ErrorType(str, Enum):
    """Types of test errors."""
    SELECTOR_NOT_FOUND = "selector_not_found"
    TIMEOUT = "timeout"
    ASSERTION_FAILED = "assertion_failed"
    NAVIGATION_FAILED = "navigation_failed"
    WEBSTACK_ERROR = "webstack_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SCRIPT_ERROR = "script_error"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    """Stages of the SQL pipeline that can be tested."""
    DATABASE_CONNECTION = "database_connection"
    DATABASE_SELECTION = "database_selection"
    QUERY_INPUT = "query_input"
    SQL_GENERATION = "sql_generation"
    QUERY_EXECUTION = "query_execution"
    RESULTS_DISPLAY = "results_display"
    FEEDBACK_SUBMISSION = "feedback_submission"
    SCHEMA_EXPLORATION = "schema_exploration"
    TRAINING_DATA = "training_data"


@dataclass
class TestStep:
    """A single step in a Playwright test."""
    action: str  # e.g., "click", "fill", "wait", "assert"
    selector: str
    value: Optional[str] = None
    description: str = ""
    timeout: int = 30000  # milliseconds


@dataclass
class GeneratedTest:
    """A generated Playwright test."""
    name: str
    description: str
    stages: List[PipelineStage]
    steps: List[TestStep]
    setup: List[str] = field(default_factory=list)
    teardown: List[str] = field(default_factory=list)
    script: str = ""
    file_path: Optional[str] = None


@dataclass
class TestError:
    """Error encountered during test execution."""
    error_type: ErrorType
    message: str
    selector: Optional[str] = None
    line_number: Optional[int] = None
    screenshot_path: Optional[str] = None
    stack_trace: Optional[str] = None
    is_webstack_error: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CorrectionAttempt:
    """An attempt to correct a failing test."""
    attempt_number: int
    original_error: TestError
    fix_applied: str
    success: bool
    new_error: Optional[TestError] = None


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    status: TestStatus
    duration_ms: int = 0
    error: Optional[TestError] = None
    correction_attempts: List[CorrectionAttempt] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


@dataclass
class TestPlan:
    """Plan for testing a set of pipeline stages."""
    stages: List[PipelineStage]
    tests: List[GeneratedTest]
    prerequisites: List[str]
    estimated_duration: int  # seconds


class TestAgent:
    """
    Agentic test agent for Playwright test generation and execution.

    Capabilities:
    - TEST_GENERATE: Generate Playwright test scripts
    - TEST_EXECUTE: Execute Playwright tests
    - TEST_FIX: Fix failing test scripts (self-correction)
    - SERVICE_MANAGE: Ensure services via Environment Agent
    - PIPELINE_ANALYZE: Analyze pipeline for test generation
    - ERROR_RECORD: Record webstack errors
    """

    # Existing test templates for reference (from .env or default)
    REFERENCE_TESTS_DIR = os.environ.get(
        "TEST_AGENT_REFERENCE_DIR",
        str(Path(__file__).parent.parent.parent.parent / "tests" / "e2e")
    )

    # Pipeline stage selectors (based on existing tests)
    STAGE_SELECTORS = {
        PipelineStage.DATABASE_CONNECTION: {
            "test_connection_btn": "[data-testid='test-connection']",
            "connection_status": "[data-testid='connection-status']",
            "server_input": "[data-testid='server-input']",
        },
        PipelineStage.DATABASE_SELECTION: {
            "database_dropdown": "[data-testid='database-select']",
            "database_option": "[data-testid='database-option']",
        },
        PipelineStage.QUERY_INPUT: {
            "query_input": "[data-testid='query-input']",
            "submit_btn": "[data-testid='submit-query']",
        },
        PipelineStage.SQL_GENERATION: {
            "sql_output": "[data-testid='sql-output']",
            "loading_indicator": "[data-testid='loading']",
        },
        PipelineStage.QUERY_EXECUTION: {
            "execute_btn": "[data-testid='execute-sql']",
            "execution_status": "[data-testid='execution-status']",
        },
        PipelineStage.RESULTS_DISPLAY: {
            "results_table": "[data-testid='results-table']",
            "row_count": "[data-testid='row-count']",
            "export_btn": "[data-testid='export-results']",
        },
        PipelineStage.FEEDBACK_SUBMISSION: {
            "thumbs_up": "[data-testid='feedback-positive']",
            "thumbs_down": "[data-testid='feedback-negative']",
            "correction_input": "[data-testid='correction-input']",
        },
    }

    # API endpoints discovered from python_services
    API_ENDPOINTS = {
        "query": "POST /api/sql/query",
        "query_stream": "POST /api/sql/query-stream",
        "execute": "POST /api/sql/execute",
        "test_connection": "POST /api/sql/test-connection",
        "databases": "GET /api/sql/databases",
        "feedback": "POST /api/sql/feedback",
        "health": "GET /api/sql/health",
        "schema_check": "POST /api/sql/schema/check",
    }

    MAX_CORRECTION_ATTEMPTS = 3

    # Default test output directory (from .env or relative to project root)
    DEFAULT_TEST_OUTPUT_DIR = os.environ.get(
        "TEST_AGENT_OUTPUT_DIR",
        str(Path(__file__).parent.parent.parent.parent / "agents" / "tests" / "generated")
    )

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        test_output_dir: str = None,
    ):
        """
        Initialize the Test Agent.

        Args:
            base_url: Base URL for the application under test
            test_output_dir: Directory for generated test files (defaults to TEST_AGENT_OUTPUT_DIR env var)
        """
        self.base_url = base_url
        self.test_output_dir = Path(test_output_dir or self.DEFAULT_TEST_OUTPUT_DIR)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        self._environment_agent = None
        self._recorded_errors: List[TestError] = []

    async def _get_environment_agent(self):
        """Lazy load the Environment Agent."""
        if self._environment_agent is None:
            from .environment_agent import EnvironmentAgent
            self._environment_agent = EnvironmentAgent()
        return self._environment_agent

    # =========================================================================
    # Pipeline Analysis
    # =========================================================================

    async def analyze_pipeline_stages(
        self,
        target_stages: Optional[List[str]] = None,
    ) -> TestPlan:
        """
        Analyze which pipeline stages need testing.

        Args:
            target_stages: Specific stages to test (all if None)

        Returns:
            TestPlan with tests for each stage
        """
        # Convert string stages to enum
        if target_stages:
            stages = [PipelineStage(s) for s in target_stages if hasattr(PipelineStage, s.upper())]
        else:
            stages = list(PipelineStage)

        logger.info(f"Analyzing pipeline stages: {[s.value for s in stages]}")

        # Generate tests for each stage
        tests = []
        for stage in stages:
            test = await self._generate_stage_test(stage)
            if test:
                tests.append(test)

        # Determine prerequisites
        prerequisites = []
        if PipelineStage.DATABASE_CONNECTION in stages:
            prerequisites.append("Node.js server running on port 3000")
            prerequisites.append("Python API running on port 8001")
        if PipelineStage.QUERY_EXECUTION in stages:
            prerequisites.append("LLM service running on port 8080")
            prerequisites.append("Database server accessible")

        return TestPlan(
            stages=stages,
            tests=tests,
            prerequisites=prerequisites,
            estimated_duration=len(tests) * 30,  # ~30s per test
        )

    async def _generate_stage_test(self, stage: PipelineStage) -> Optional[GeneratedTest]:
        """Generate a test for a specific pipeline stage."""
        selectors = self.STAGE_SELECTORS.get(stage, {})

        if not selectors:
            logger.warning(f"No selectors defined for stage: {stage.value}")
            return None

        steps = []
        description = f"Test {stage.value.replace('_', ' ').title()}"

        if stage == PipelineStage.DATABASE_CONNECTION:
            steps = [
                TestStep("goto", self.base_url, description="Navigate to application"),
                TestStep("wait", selectors["server_input"], description="Wait for server input"),
                TestStep("fill", selectors["server_input"], "NCSQLTEST", "Enter server name"),
                TestStep("click", selectors["test_connection_btn"], description="Click test connection"),
                TestStep("wait", selectors["connection_status"], description="Wait for connection status"),
                TestStep("assert", selectors["connection_status"], "connected", "Verify connected status"),
            ]

        elif stage == PipelineStage.DATABASE_SELECTION:
            steps = [
                TestStep("wait", selectors["database_dropdown"], description="Wait for database dropdown"),
                TestStep("click", selectors["database_dropdown"], description="Open database dropdown"),
                TestStep("click", selectors["database_option"], description="Select a database"),
            ]

        elif stage == PipelineStage.QUERY_INPUT:
            steps = [
                TestStep("wait", selectors["query_input"], description="Wait for query input"),
                TestStep("fill", selectors["query_input"], "Show all tickets from today", "Enter test query"),
                TestStep("click", selectors["submit_btn"], description="Submit query"),
            ]

        elif stage == PipelineStage.SQL_GENERATION:
            steps = [
                TestStep("wait", selectors["loading_indicator"], description="Wait for loading"),
                TestStep("wait_hidden", selectors["loading_indicator"], description="Wait for loading to complete"),
                TestStep("assert_visible", selectors["sql_output"], description="Verify SQL output visible"),
            ]

        elif stage == PipelineStage.RESULTS_DISPLAY:
            steps = [
                TestStep("wait", selectors["results_table"], description="Wait for results table"),
                TestStep("assert_visible", selectors["row_count"], description="Verify row count displayed"),
            ]

        elif stage == PipelineStage.FEEDBACK_SUBMISSION:
            steps = [
                TestStep("wait", selectors["thumbs_up"], description="Wait for feedback buttons"),
                TestStep("click", selectors["thumbs_up"], description="Click positive feedback"),
            ]

        return GeneratedTest(
            name=f"test_{stage.value}",
            description=description,
            stages=[stage],
            steps=steps,
        )

    # =========================================================================
    # Test Generation
    # =========================================================================

    async def generate_test(
        self,
        stages: List[PipelineStage],
        test_name: str = None,
    ) -> GeneratedTest:
        """
        Generate a comprehensive Playwright test.

        Args:
            stages: Pipeline stages to include in the test
            test_name: Custom test name (auto-generated if None)

        Returns:
            GeneratedTest with complete script
        """
        if not test_name:
            test_name = f"test_{'_'.join(s.value for s in stages)}"

        logger.info(f"Generating test: {test_name} for stages: {[s.value for s in stages]}")

        # Collect steps from all stages
        all_steps = []
        for stage in stages:
            stage_test = await self._generate_stage_test(stage)
            if stage_test:
                all_steps.extend(stage_test.steps)

        # Generate the script
        script = self._generate_playwright_script(test_name, all_steps)

        test = GeneratedTest(
            name=test_name,
            description=f"E2E test covering: {', '.join(s.value for s in stages)}",
            stages=stages,
            steps=all_steps,
            script=script,
        )

        # Save to file
        file_path = self.test_output_dir / f"{test_name}.spec.js"
        file_path.write_text(script)
        test.file_path = str(file_path)

        logger.info(f"Generated test saved to: {file_path}")
        return test

    def _generate_playwright_script(
        self,
        test_name: str,
        steps: List[TestStep],
    ) -> str:
        """Generate a Playwright test script from steps."""
        script_lines = [
            "// @ts-check",
            "const { test, expect } = require('@playwright/test');",
            "",
            f"test.describe('{test_name.replace('_', ' ').title()}', () => {{",
            "",
            "  test.beforeEach(async ({ page }) => {",
            f"    await page.goto('{self.base_url}');",
            "  });",
            "",
            f"  test('{test_name}', async ({{ page }}) => {{",
        ]

        for step in steps:
            script_lines.append(f"    // {step.description or step.action}")

            if step.action == "goto":
                script_lines.append(f"    await page.goto('{step.selector}');")

            elif step.action == "wait":
                script_lines.append(f"    await page.waitForSelector('{step.selector}', {{ timeout: {step.timeout} }});")

            elif step.action == "wait_hidden":
                script_lines.append(f"    await page.waitForSelector('{step.selector}', {{ state: 'hidden', timeout: {step.timeout} }});")

            elif step.action == "click":
                script_lines.append(f"    await page.click('{step.selector}');")

            elif step.action == "fill":
                script_lines.append(f"    await page.fill('{step.selector}', '{step.value}');")

            elif step.action == "assert":
                script_lines.append(f"    await expect(page.locator('{step.selector}')).toContainText('{step.value}');")

            elif step.action == "assert_visible":
                script_lines.append(f"    await expect(page.locator('{step.selector}')).toBeVisible();")

            script_lines.append("")

        script_lines.append("  });")
        script_lines.append("});")

        return "\n".join(script_lines)

    # =========================================================================
    # Test Execution
    # =========================================================================

    async def execute_test(
        self,
        test: GeneratedTest,
        ensure_environment: bool = True,
    ) -> TestResult:
        """
        Execute a generated test with self-correction.

        Args:
            test: The test to execute
            ensure_environment: Whether to ensure services are running

        Returns:
            TestResult with execution details
        """
        logger.info(f"Executing test: {test.name}")

        # Ensure environment is ready
        if ensure_environment:
            env_agent = await self._get_environment_agent()
            env_status = await env_agent.ensure_test_environment()

            if not env_status.healthy:
                return TestResult(
                    test_name=test.name,
                    status=TestStatus.ERROR,
                    error=TestError(
                        error_type=ErrorType.SERVICE_UNAVAILABLE,
                        message=f"Environment not healthy: {env_status.warnings}",
                        is_webstack_error=True,
                    ),
                )

        # Execute with correction loop
        result = await self._execute_with_correction(test)

        return result

    async def _execute_with_correction(
        self,
        test: GeneratedTest,
    ) -> TestResult:
        """Execute test with MAGIC self-correction pattern."""
        correction_attempts = []
        current_script = test.script
        current_file = test.file_path

        for attempt in range(1, self.MAX_CORRECTION_ATTEMPTS + 1):
            logger.info(f"Execution attempt {attempt}/{self.MAX_CORRECTION_ATTEMPTS}")

            # Run the test
            result = await self._run_playwright_test(current_file)

            if result.status == TestStatus.PASSED:
                result.correction_attempts = correction_attempts
                return result

            if result.error is None:
                continue

            # Analyze the error
            error = result.error

            # Record webstack errors
            if error.is_webstack_error:
                self._recorded_errors.append(error)
                logger.warning(f"Webstack error recorded: {error.message}")
                break  # Don't try to fix webstack errors

            # Try to fix the error
            fix_result = await self._attempt_fix(
                current_script,
                error,
                correction_attempts,
            )

            if fix_result is None:
                logger.warning("Could not generate fix for error")
                break

            new_script, fix_description = fix_result

            # Save the fixed script
            current_script = new_script
            fixed_path = Path(current_file).with_suffix(".fixed.spec.js")
            fixed_path.write_text(new_script)
            current_file = str(fixed_path)

            correction_attempts.append(CorrectionAttempt(
                attempt_number=attempt,
                original_error=error,
                fix_applied=fix_description,
                success=False,  # Will be updated if next attempt passes
            ))

        # Final result after all attempts
        result.correction_attempts = correction_attempts
        return result

    async def _run_playwright_test(self, test_file: str) -> TestResult:
        """Run a Playwright test file and capture results."""
        try:
            # Run Playwright
            cmd = [
                "npx", "playwright", "test", test_file,
                "--reporter=json",
                "--output=/tmp/playwright-results",
            ]

            start_time = asyncio.get_event_loop().time()

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/mnt/c/Projects/llm_website",
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120,
            )

            duration = int((asyncio.get_event_loop().time() - start_time) * 1000)

            if process.returncode == 0:
                return TestResult(
                    test_name=Path(test_file).stem,
                    status=TestStatus.PASSED,
                    duration_ms=duration,
                )

            # Parse error from output
            error = self._parse_playwright_error(
                stdout.decode(),
                stderr.decode(),
            )

            return TestResult(
                test_name=Path(test_file).stem,
                status=TestStatus.FAILED,
                duration_ms=duration,
                error=error,
            )

        except asyncio.TimeoutError:
            return TestResult(
                test_name=Path(test_file).stem,
                status=TestStatus.ERROR,
                error=TestError(
                    error_type=ErrorType.TIMEOUT,
                    message="Test execution timed out after 120 seconds",
                ),
            )
        except Exception as e:
            return TestResult(
                test_name=Path(test_file).stem,
                status=TestStatus.ERROR,
                error=TestError(
                    error_type=ErrorType.UNKNOWN,
                    message=str(e),
                ),
            )

    def _parse_playwright_error(
        self,
        stdout: str,
        stderr: str,
    ) -> TestError:
        """Parse Playwright output to extract error details."""
        full_output = stdout + stderr

        # Pattern matching for common errors
        if "waiting for selector" in full_output.lower():
            selector_match = re.search(r'selector\s+"([^"]+)"', full_output)
            return TestError(
                error_type=ErrorType.SELECTOR_NOT_FOUND,
                message="Element not found",
                selector=selector_match.group(1) if selector_match else None,
            )

        if "timeout" in full_output.lower():
            return TestError(
                error_type=ErrorType.TIMEOUT,
                message="Operation timed out",
            )

        if "expect" in full_output.lower() and "received" in full_output.lower():
            return TestError(
                error_type=ErrorType.ASSERTION_FAILED,
                message="Assertion failed",
            )

        if "net::err" in full_output.lower() or "connection refused" in full_output.lower():
            return TestError(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                message="Could not connect to service",
                is_webstack_error=True,
            )

        if "500" in full_output or "internal server error" in full_output.lower():
            return TestError(
                error_type=ErrorType.WEBSTACK_ERROR,
                message="Server returned 500 error",
                is_webstack_error=True,
            )

        return TestError(
            error_type=ErrorType.UNKNOWN,
            message=full_output[:500],  # First 500 chars of output
        )

    # =========================================================================
    # Self-Correction (MAGIC Pattern)
    # =========================================================================

    async def _attempt_fix(
        self,
        script: str,
        error: TestError,
        previous_attempts: List[CorrectionAttempt],
    ) -> Optional[Tuple[str, str]]:
        """
        Attempt to fix a test script based on the error.

        Uses pattern-based fixes for common issues.
        Delegates complex fixes to Code Agent (TODO).

        Args:
            script: Current test script
            error: The error that occurred
            previous_attempts: Previous fix attempts for context

        Returns:
            Tuple of (fixed_script, fix_description) or None if no fix possible
        """
        logger.info(f"Attempting fix for error type: {error.error_type}")

        # Pattern-based fixes (high confidence, no LLM needed)
        if error.error_type == ErrorType.SELECTOR_NOT_FOUND and error.selector:
            return self._fix_selector_not_found(script, error.selector)

        if error.error_type == ErrorType.TIMEOUT:
            return self._fix_timeout(script)

        # For complex fixes, delegate to Code Agent
        if self._code_agent_available():
            logger.info(f"Delegating complex fix to Code Agent for {error.error_type}")
            return await self._delegate_to_code_agent(script, error, previous_attempts)

        logger.warning(f"No pattern-based fix available for {error.error_type}")
        return None

    def _code_agent_available(self) -> bool:
        """Check if Code Agent is available for delegation."""
        # Check if ewr_code_agent is importable
        try:
            import sys
            agents_path = "/mnt/c/Projects/llm_website/agents"
            if agents_path not in sys.path:
                sys.path.insert(0, agents_path)
            from ewr_code_agent.src.ewr_code_agent import CodeAgent
            return True
        except ImportError:
            return False

    async def _delegate_to_code_agent(
        self,
        script: str,
        error: TestError,
        previous_attempts: List[CorrectionAttempt],
    ) -> Optional[Tuple[str, str]]:
        """
        Delegate complex test fix to the Code Agent.

        The Code Agent uses LLM to analyze the error and generate a fix.

        Args:
            script: Current test script
            error: The error that occurred
            previous_attempts: Previous fix attempts

        Returns:
            Tuple of (fixed_script, fix_description) or None
        """
        try:
            # Build context for the Code Agent
            context = {
                "script": script,
                "error_type": error.error_type.value,
                "error_message": error.message,
                "selector": error.selector,
                "stack_trace": error.stack_trace,
                "previous_attempts": [
                    {
                        "fix": a.fix_applied,
                        "error": a.original_error.message,
                    }
                    for a in previous_attempts[-3:]  # Last 3 attempts
                ],
            }

            # Import and use Code Agent
            import sys
            agents_path = "/mnt/c/Projects/llm_website/agents"
            if agents_path not in sys.path:
                sys.path.insert(0, agents_path)

            from ewr_code_agent.src.ewr_code_agent import CodeAgent

            # Create agent instance
            code_agent = CodeAgent()

            # Request code fix
            fix_request = f"""Fix this Playwright test script that is failing with error:

Error Type: {error.error_type.value}
Error Message: {error.message}
{f'Failing Selector: {error.selector}' if error.selector else ''}

The test script:
```javascript
{script}
```

{self._format_previous_attempts(previous_attempts)}

Provide the corrected script and a brief description of what you changed.
Focus on:
1. Fixing the specific error
2. Using more robust selectors
3. Adding appropriate waits
4. Handling edge cases

Return the complete fixed script."""

            # Use the code agent's analyze/fix capability
            result = await code_agent.handle_task({
                "task_id": f"fix-test-{error.error_type.value}",
                "task_type": "code_refactor",
                "params": {
                    "code": script,
                    "instruction": fix_request,
                    "language": "javascript",
                }
            })

            if result.status.value == "completed" and result.result:
                fixed_code = result.result.get("refactored_code", "")
                explanation = result.result.get("explanation", "Code Agent fix applied")

                if fixed_code and fixed_code != script:
                    logger.info(f"Code Agent provided fix: {explanation[:100]}...")
                    return (fixed_code, f"Code Agent: {explanation}")

            logger.warning("Code Agent did not provide a valid fix")
            return None

        except Exception as e:
            logger.error(f"Code Agent delegation failed: {e}")
            return None

    def _format_previous_attempts(self, attempts: List[CorrectionAttempt]) -> str:
        """Format previous fix attempts for context."""
        if not attempts:
            return ""

        lines = ["Previous fix attempts that failed:"]
        for i, attempt in enumerate(attempts[-3:], 1):
            lines.append(f"{i}. Fix: {attempt.fix_applied}")
            lines.append(f"   Error: {attempt.original_error.message}")
        return "\n".join(lines)

    def _fix_selector_not_found(
        self,
        script: str,
        selector: str,
    ) -> Optional[Tuple[str, str]]:
        """Fix a selector not found error."""
        # Try alternative selector strategies
        alternatives = []

        if selector.startswith("[data-testid="):
            # Try without data-testid
            testid = re.search(r"data-testid='([^']+)'", selector)
            if testid:
                # Try CSS class
                alternatives.append(f".{testid.group(1)}")
                # Try id
                alternatives.append(f"#{testid.group(1)}")
                # Try text
                alternatives.append(f"text={testid.group(1).replace('-', ' ')}")

        if not alternatives:
            return None

        # Replace the first occurrence with the first alternative
        new_script = script.replace(selector, alternatives[0], 1)

        return (
            new_script,
            f"Changed selector from '{selector}' to '{alternatives[0]}'",
        )

    def _fix_timeout(self, script: str) -> Optional[Tuple[str, str]]:
        """Fix timeout errors by increasing timeout values."""
        # Double all timeout values
        new_script = re.sub(
            r"timeout:\s*(\d+)",
            lambda m: f"timeout: {int(m.group(1)) * 2}",
            script,
        )

        if new_script == script:
            # Add explicit wait before problematic actions
            new_script = script.replace(
                "await page.click",
                "await page.waitForTimeout(1000);\n    await page.click",
            )

        return (
            new_script,
            "Increased timeout values and added explicit waits",
        )

    # =========================================================================
    # Error Recording
    # =========================================================================

    def get_recorded_errors(self) -> List[TestError]:
        """Get all recorded webstack errors."""
        return self._recorded_errors.copy()

    def clear_recorded_errors(self) -> None:
        """Clear recorded errors."""
        self._recorded_errors.clear()

    async def export_error_report(self, output_path: str) -> str:
        """Export recorded errors to a JSON file."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "error_count": len(self._recorded_errors),
            "errors": [
                {
                    "type": e.error_type.value,
                    "message": e.message,
                    "selector": e.selector,
                    "is_webstack_error": e.is_webstack_error,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self._recorded_errors
            ],
        }

        Path(output_path).write_text(json.dumps(report, indent=2))
        return output_path


# Convenience functions
async def generate_pipeline_tests(
    stages: Optional[List[str]] = None,
) -> TestPlan:
    """Generate tests for specified pipeline stages."""
    agent = TestAgent()
    return await agent.analyze_pipeline_stages(stages)


async def run_generated_tests(test_plan: TestPlan) -> List[TestResult]:
    """Execute all tests in a test plan."""
    agent = TestAgent()
    results = []

    for test in test_plan.tests:
        result = await agent.execute_test(test)
        results.append(result)

    return results
