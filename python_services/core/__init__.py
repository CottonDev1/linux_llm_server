"""
Core utilities for the Python Data Services.

This module provides platform-aware utilities for process management,
port handling, and system detection.
"""
from .platform_utils import is_windows_environment
from .port_utils import get_pids_on_port, is_port_available
from .process_manager import (
    write_pid_file,
    remove_pid_file,
    kill_existing_service,
    setup_worker_signal_handlers,
    setup_parent_signal_handlers,
)

__all__ = [
    "is_windows_environment",
    "get_pids_on_port",
    "is_port_available",
    "write_pid_file",
    "remove_pid_file",
    "kill_existing_service",
    "setup_worker_signal_handlers",
    "setup_parent_signal_handlers",
]
