"""
Platform detection utilities.

Provides cross-platform detection for Windows and Linux environments,
including proper WSL (Windows Subsystem for Linux) handling.
"""
import sys


def is_windows_environment() -> bool:
    """
    Detect if we're managing Windows processes.
    Works correctly in both native Windows and WSL environments.

    Returns True if:
    - Running on native Windows (sys.platform == 'win32')
    - Running in WSL (detected via /proc/version containing 'microsoft')
    """
    # Native Windows
    if sys.platform == 'win32':
        return True

    # Check if running in WSL (Linux kernel but Windows processes)
    if sys.platform == 'linux':
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return True
        except:
            pass

    return False
