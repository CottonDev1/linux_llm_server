"""
Logging Utilities

Provides consistent formatted logging with colored tags and timestamps.
Also writes to pipeline log files for the admin dashboard.

Format: HH:MM:SS AM/PM | Tag Name          |  Message

Thread-safe: Uses a lock to prevent concurrent stdout writes.
"""

import os
import threading
from datetime import datetime
from typing import Optional

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Tag column width for alignment (longest tag "Embedding Service" = 17, +1 = 18)
TAG_WIDTH = 18

# Thread lock for safe concurrent logging
_log_lock = threading.Lock()

# Log directory and files (matching config.py)
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "logs")
_LOG_PIPELINE_FILE = "pipeline.log"
_LOG_SERVICE_FILE = "python-service.log"


def _ensure_log_dir() -> None:
    """Ensure log directory exists."""
    if not os.path.exists(_LOG_DIR):
        os.makedirs(_LOG_DIR, exist_ok=True)


# Track the last date we wrote to each file for daily rotation
_last_write_dates: dict = {}


def _check_daily_rotation(filepath: str) -> None:
    """Clear log file if it's from a previous day."""
    try:
        if os.path.exists(filepath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            today = datetime.now().date()
            if file_mtime.date() < today:
                # File is from a previous day, clear it
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("")
    except Exception:
        pass  # Silently fail to avoid log loops


def _write_to_file(filename: str, message: str) -> None:
    """Write a log entry to a file. Clears file if from previous day."""
    try:
        _ensure_log_dir()
        filepath = os.path.join(_LOG_DIR, filename)

        # Check if we need to rotate (clear) the file for a new day
        today = datetime.now().date()
        last_date = _last_write_dates.get(filename)
        if last_date != today:
            _check_daily_rotation(filepath)
            _last_write_dates[filename] = today

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    except Exception:
        pass  # Silently fail file writes to avoid log loops




def log_message(
    tag: str,
    message: str,
    color: str = GREEN,
    level: str = "INFO",
    write_to_pipeline: bool = True,
    write_to_service: bool = True
) -> None:
    """
    Print a formatted log message and write to log files.

    Format: HH:MM:SS AM/PM | Tag Name          |  message

    Args:
        tag: The tag to display (e.g., "Health Monitor")
        message: The log message
        color: ANSI color code for the tag (default: GREEN)
        level: Log level for file output (INFO, WARNING, ERROR)
        write_to_pipeline: Write to pipeline.log
        write_to_service: Write to python-service.log
    """
    timestamp = datetime.now().strftime("%I:%M:%S %p")

    # Pad tag to fixed width for alignment
    padded_tag = tag[:TAG_WIDTH].ljust(TAG_WIDTH)

    # Console output with color (two spaces after second |)
    console_msg = f"{timestamp} {color}| {padded_tag}|{RESET}  {message}"

    # File output without color codes (two spaces after second |)
    file_msg = f"{timestamp} | {padded_tag}|  {level}: {message}"

    # Use lock for thread-safe output
    with _log_lock:
        print(console_msg)

        if write_to_pipeline:
            _write_to_file(_LOG_PIPELINE_FILE, file_msg)

        if write_to_service:
            _write_to_file(_LOG_SERVICE_FILE, file_msg)


def log_health(message: str) -> None:
    """Log a Health Monitor message in green (service log only)."""
    log_message("Health Monitor", message, GREEN, "INFO", write_to_pipeline=False)


def log_process(message: str) -> None:
    """Log a Process Manager message in green (service log only)."""
    log_message("Process Manager", message, GREEN, "INFO", write_to_pipeline=False)


def log_info(pipeline_name: str, message: str) -> None:
    """Log an Info message with pipeline name in cyan (pipeline log only)."""
    log_message(pipeline_name, message, CYAN, "INFO", write_to_service=False)


def log_warning(tag: str, message: str) -> None:
    """Log a warning message in yellow (both logs)."""
    log_message(tag, message, YELLOW, "WARNING")


def log_error(tag: str, message: str) -> None:
    """Log an error message in red (both logs)."""
    log_message(tag, message, RED, "ERROR")


def log_debug(tag: str, message: str) -> None:
    """Log a debug message (no color, service log only)."""
    log_message(tag, message, RESET, "DEBUG", write_to_pipeline=False)


def log_service(tag: str, message: str) -> None:
    """Log a service message in green (service log only, for Python tab)."""
    log_message(tag, message, GREEN, "INFO", write_to_pipeline=False)
