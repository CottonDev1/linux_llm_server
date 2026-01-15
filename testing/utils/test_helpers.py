"""
Test Helper Utilities
=====================

Shared utility functions for pipeline tests.
"""

import os
import uuid
import json
import time
import asyncio
import tempfile
from typing import Callable, Any, Optional, Dict
from datetime import datetime
from functools import wraps
from contextlib import contextmanager


def generate_test_id(prefix: str = "test") -> str:
    """
    Generate a unique test ID.

    Args:
        prefix: Prefix for the ID

    Returns:
        Unique test identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique}"


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 30.0,
    interval: float = 0.5,
    error_message: str = "Condition not met within timeout"
) -> bool:
    """
    Wait for a condition to become true.

    Args:
        condition: Callable that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
        error_message: Error message if timeout occurs

    Returns:
        True if condition was met

    Raises:
        TimeoutError: If condition not met within timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            if condition():
                return True
        except Exception:
            pass
        await asyncio.sleep(interval)

    raise TimeoutError(f"{error_message} (waited {timeout}s)")


async def retry_async(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of the function

    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception


@contextmanager
def measure_time(label: str = "Operation"):
    """
    Context manager to measure execution time.

    Args:
        label: Label for the operation

    Yields:
        Dict with timing information (updated on exit)

    Example:
        with measure_time("Database query") as timing:
            result = db.query(...)
        print(f"Query took {timing['elapsed_ms']}ms")
    """
    timing = {"start": None, "end": None, "elapsed_ms": 0}
    timing["start"] = time.perf_counter()

    try:
        yield timing
    finally:
        timing["end"] = time.perf_counter()
        timing["elapsed_ms"] = int((timing["end"] - timing["start"]) * 1000)
        print(f"{label}: {timing['elapsed_ms']}ms")


def create_temp_file(
    content: str = "",
    suffix: str = ".txt",
    prefix: str = "test_",
    cleanup: bool = True,
) -> str:
    """
    Create a temporary file for testing.

    Args:
        content: Content to write to the file
        suffix: File suffix/extension
        prefix: File name prefix
        cleanup: Whether to register for cleanup

    Returns:
        Path to the temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)

    if content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    return path


def load_test_data(filename: str, base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load test data from a JSON file.

    Args:
        filename: Name of the test data file
        base_path: Base path for test data files

    Returns:
        Loaded test data dictionary

    Raises:
        FileNotFoundError: If test data file not found
    """
    if base_path is None:
        # Default to tests/data directory
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )

    file_path = os.path.join(base_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestTimer:
    """
    Timer for tracking test execution time.

    Usage:
        timer = TestTimer()
        timer.start("step1")
        # do work
        timer.stop("step1")
        timer.start("step2")
        # do work
        timer.stop("step2")
        print(timer.get_summary())
    """

    def __init__(self):
        self._timers: Dict[str, Dict[str, Any]] = {}

    def start(self, name: str):
        """Start a named timer."""
        self._timers[name] = {
            "start": time.perf_counter(),
            "end": None,
            "elapsed_ms": None,
        }

    def stop(self, name: str) -> int:
        """Stop a named timer and return elapsed time in ms."""
        if name not in self._timers:
            raise KeyError(f"Timer '{name}' was not started")

        timer = self._timers[name]
        timer["end"] = time.perf_counter()
        timer["elapsed_ms"] = int((timer["end"] - timer["start"]) * 1000)
        return timer["elapsed_ms"]

    def get_elapsed(self, name: str) -> Optional[int]:
        """Get elapsed time for a timer in milliseconds."""
        if name in self._timers and self._timers[name]["elapsed_ms"]:
            return self._timers[name]["elapsed_ms"]
        return None

    def get_summary(self) -> Dict[str, int]:
        """Get summary of all timers."""
        return {
            name: timer["elapsed_ms"]
            for name, timer in self._timers.items()
            if timer["elapsed_ms"] is not None
        }

    def get_total(self) -> int:
        """Get total time across all timers."""
        return sum(self.get_summary().values())


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.

    - Converts to lowercase
    - Removes extra whitespace
    - Removes trailing semicolons

    Args:
        sql: SQL query string

    Returns:
        Normalized SQL string
    """
    import re

    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

    # Normalize whitespace
    sql = ' '.join(sql.split())

    # Lowercase and strip
    sql = sql.lower().strip()

    # Remove trailing semicolon
    if sql.endswith(';'):
        sql = sql[:-1]

    return sql
