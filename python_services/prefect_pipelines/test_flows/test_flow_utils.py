"""
Test Flow Utilities
===================

Shared utilities for Prefect test flows:
- Event emission for real-time updates
- Metrics tracking and reporting
- Test result formatting
- Progress tracking

All test flows are MANUAL TRIGGER ONLY - no automatic scheduling.
All LLM calls use local llama.cpp endpoints only - no external APIs.
"""

import json
import time
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class EventType(str, Enum):
    """Event types for test flow updates."""
    TEST_START = "test_start"
    TEST_PROGRESS = "test_progress"
    TEST_COMPLETE = "test_complete"
    TEST_FAILED = "test_failed"
    SUITE_START = "suite_start"
    SUITE_COMPLETE = "suite_complete"
    METRICS = "metrics"
    ARTIFACT = "artifact"


@dataclass
class TestResult:
    """Result from a single test."""
    name: str
    status: TestStatus
    duration_ms: int = 0
    error: Optional[str] = None
    assertions: int = 0
    pipeline: str = ""
    category: str = ""  # storage, retrieval, generation, e2e
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "assertions": self.assertions,
            "pipeline": self.pipeline,
            "category": self.category,
            "details": self.details,
        }


@dataclass
class TestMetrics:
    """Aggregated metrics for a test run."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    pipelines: Dict[str, Dict[str, int]] = field(default_factory=dict)
    categories: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_result(self, result: TestResult):
        """Add a test result to metrics."""
        self.total += 1

        if result.status == TestStatus.PASSED:
            self.passed += 1
        elif result.status == TestStatus.FAILED:
            self.failed += 1
        elif result.status == TestStatus.SKIPPED:
            self.skipped += 1
        elif result.status == TestStatus.ERROR:
            self.errors += 1

        self.duration_ms += result.duration_ms

        # Track by pipeline
        if result.pipeline:
            if result.pipeline not in self.pipelines:
                self.pipelines[result.pipeline] = {"passed": 0, "failed": 0, "total": 0}
            self.pipelines[result.pipeline]["total"] += 1
            if result.status == TestStatus.PASSED:
                self.pipelines[result.pipeline]["passed"] += 1
            elif result.status in (TestStatus.FAILED, TestStatus.ERROR):
                self.pipelines[result.pipeline]["failed"] += 1

        # Track by category
        if result.category:
            if result.category not in self.categories:
                self.categories[result.category] = {"passed": 0, "failed": 0, "total": 0}
            self.categories[result.category]["total"] += 1
            if result.status == TestStatus.PASSED:
                self.categories[result.category]["passed"] += 1
            elif result.status in (TestStatus.FAILED, TestStatus.ERROR):
                self.categories[result.category]["failed"] += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "success_rate": round(self.success_rate, 2),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "pipelines": self.pipelines,
            "categories": self.categories,
        }


@dataclass
class AgentActivity:
    """Activity record for an agent."""
    agent_id: str
    pipeline: str
    action: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """
    Track progress of test execution.

    Provides real-time progress updates for Prefect UI.
    """

    def __init__(self, total_tests: int = 0):
        self.total = total_tests
        self.completed = 0
        self.current_test: Optional[str] = None
        self.current_pipeline: Optional[str] = None
        self.start_time = time.time()
        self.results: List[TestResult] = []
        self.metrics = TestMetrics()

    def start_test(self, name: str, pipeline: str, category: str):
        """Mark a test as started."""
        self.current_test = name
        self.current_pipeline = pipeline

    def complete_test(self, result: TestResult):
        """Mark a test as completed."""
        self.completed += 1
        self.results.append(result)
        self.metrics.add_result(result)
        self.current_test = None

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        elapsed = int((time.time() - self.start_time) * 1000)
        return {
            "total": self.total,
            "completed": self.completed,
            "remaining": self.total - self.completed,
            "current_test": self.current_test,
            "current_pipeline": self.current_pipeline,
            "elapsed_ms": elapsed,
            "percent": round((self.completed / self.total * 100) if self.total > 0 else 0, 1),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get final summary."""
        self.metrics.end_time = datetime.utcnow().isoformat()
        return {
            "progress": self.get_progress(),
            "metrics": self.metrics.to_dict(),
        }


class TestTimer:
    """Timer for tracking test execution time."""

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self):
        """Start the timer."""
        self._start = time.perf_counter()

    def stop(self) -> int:
        """Stop the timer and return elapsed milliseconds."""
        self._end = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self._start is None:
            return 0
        end = self._end or time.perf_counter()
        return int((end - self._start) * 1000)


def emit_test_event(
    event_type: EventType,
    data: Dict[str, Any],
    logger=None,
) -> Dict[str, Any]:
    """
    Emit a test event for Prefect tracking.

    Args:
        event_type: Type of event
        data: Event data
        logger: Optional logger

    Returns:
        Formatted event dictionary
    """
    event = {
        "type": event_type.value,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }

    if logger:
        logger.info(f"[{event_type.value}] {json.dumps(data)}")

    return event


def emit_agent_activity(
    agent_id: str,
    pipeline: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    logger=None,
) -> AgentActivity:
    """
    Emit an agent activity record.

    Args:
        agent_id: Identifier for the agent
        pipeline: Pipeline being tested
        action: Action being performed
        details: Optional additional details
        logger: Optional logger

    Returns:
        AgentActivity record
    """
    activity = AgentActivity(
        agent_id=agent_id,
        pipeline=pipeline,
        action=action,
        details=details or {},
    )

    if logger:
        logger.info(f"[Agent:{agent_id}] {pipeline}/{action}")

    return activity


def emit_metrics_event(
    metrics: TestMetrics,
    logger=None,
) -> Dict[str, Any]:
    """
    Emit a metrics event.

    Args:
        metrics: Test metrics
        logger: Optional logger

    Returns:
        Formatted metrics event
    """
    return emit_test_event(
        EventType.METRICS,
        metrics.to_dict(),
        logger=logger,
    )


def create_test_summary_table(results: List[TestResult]) -> str:
    """
    Create a markdown table summarizing test results.

    Args:
        results: List of test results

    Returns:
        Markdown table string
    """
    if not results:
        return "No test results"

    lines = [
        "| Test | Pipeline | Category | Status | Duration |",
        "|------|----------|----------|--------|----------|",
    ]

    for r in results:
        status_icon = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.ERROR: "💥",
        }.get(r.status, "❓")

        lines.append(
            f"| {r.name} | {r.pipeline} | {r.category} | "
            f"{status_icon} {r.status.value} | {r.duration_ms}ms |"
        )

    return "\n".join(lines)


def create_test_results_table(
    metrics: TestMetrics,
    by: str = "pipeline"
) -> str:
    """
    Create a markdown table of results by pipeline or category.

    Args:
        metrics: Test metrics
        by: Group by 'pipeline' or 'category'

    Returns:
        Markdown table string
    """
    data = metrics.pipelines if by == "pipeline" else metrics.categories

    if not data:
        return f"No {by} results"

    lines = [
        f"| {by.title()} | Passed | Failed | Total | Rate |",
        "|---------|--------|--------|-------|------|",
    ]

    for name, counts in data.items():
        total = counts.get("total", 0)
        passed = counts.get("passed", 0)
        rate = round((passed / total * 100) if total > 0 else 0, 1)
        lines.append(
            f"| {name} | {passed} | {counts.get('failed', 0)} | {total} | {rate}% |"
        )

    return "\n".join(lines)


def create_test_report_markdown(
    metrics: TestMetrics,
    results: List[TestResult],
    title: str = "Pipeline Test Report",
) -> str:
    """
    Create a full markdown test report.

    Args:
        metrics: Test metrics
        results: List of test results
        title: Report title

    Returns:
        Full markdown report
    """
    sections = [
        f"# {title}",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Total Tests:** {metrics.total}",
        f"- **Passed:** {metrics.passed} ✅",
        f"- **Failed:** {metrics.failed} ❌",
        f"- **Skipped:** {metrics.skipped} ⏭️",
        f"- **Errors:** {metrics.errors} 💥",
        f"- **Success Rate:** {metrics.success_rate:.1f}%",
        f"- **Duration:** {metrics.duration_ms}ms",
        "",
        "## Results by Pipeline",
        "",
        create_test_results_table(metrics, "pipeline"),
        "",
        "## Results by Category",
        "",
        create_test_results_table(metrics, "category"),
        "",
        "## Test Details",
        "",
        create_test_summary_table(results),
    ]

    # Add failed test details
    failed = [r for r in results if r.status in (TestStatus.FAILED, TestStatus.ERROR)]
    if failed:
        sections.extend([
            "",
            "## Failed Tests",
            "",
        ])
        for r in failed:
            sections.extend([
                f"### {r.name}",
                "",
                f"- **Pipeline:** {r.pipeline}",
                f"- **Category:** {r.category}",
                f"- **Error:** {r.error or 'Unknown'}",
                "",
            ])

    return "\n".join(sections)


def parse_pytest_output(output: str) -> List[TestResult]:
    """
    Parse pytest output to extract test results.

    Args:
        output: Pytest stdout output

    Returns:
        List of TestResult objects
    """
    import re

    results = []

    # Pattern for test results: tests/pipelines/sql/test_storage.py::TestClass::test_name PASSED/FAILED
    pattern = r'(tests/pipelines/\w+/test_\w+\.py::[\w:]+)\s+(PASSED|FAILED|SKIPPED|ERROR)'

    for match in re.finditer(pattern, output):
        full_name = match.group(1)
        status_str = match.group(2)

        # Extract pipeline and test name
        parts = full_name.split("::")
        module_path = parts[0]
        test_name = parts[-1] if len(parts) > 1 else full_name

        # Extract pipeline from path
        pipeline_match = re.search(r'pipelines/(\w+)/', module_path)
        pipeline = pipeline_match.group(1) if pipeline_match else "unknown"

        # Determine category from filename
        category = "unknown"
        if "storage" in module_path:
            category = "storage"
        elif "retrieval" in module_path:
            category = "retrieval"
        elif "generation" in module_path:
            category = "generation"
        elif "e2e" in module_path:
            category = "e2e"

        status = TestStatus[status_str]

        results.append(TestResult(
            name=test_name,
            status=status,
            pipeline=pipeline,
            category=category,
        ))

    return results


def create_metrics_from_results(results: List[TestResult]) -> TestMetrics:
    """
    Create TestMetrics from a list of TestResults.

    Args:
        results: List of test results

    Returns:
        Aggregated TestMetrics
    """
    metrics = TestMetrics()
    metrics.start_time = datetime.utcnow().isoformat()

    for result in results:
        metrics.add_result(result)

    metrics.end_time = datetime.utcnow().isoformat()
    return metrics
