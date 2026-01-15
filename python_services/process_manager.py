"""
Process Management - Platform-aware PID tracking and cleanup

This module handles process management for both Windows and Linux environments.
Key features:
- Tracks both parent (reloader) and worker PIDs when using uvicorn reload mode
- Uses process tree killing on Windows (/T flag) to ensure children die with parent
- Uses process group killing on Linux (killpg) for the same purpose
- Properly detects WSL environment running Windows processes
- Includes emergency cleanup via atexit for crash scenarios
"""

import os
import sys
import socket
import subprocess
import signal
import atexit
import time
import re
from pathlib import Path
from typing import Optional

from core.log_utils import log_process, log_warning

# Configuration
PROCESS_KILL_WAIT_SECONDS = 5  # Windows needs longer wait times
PROCESS_KILL_MAX_RETRIES = 5
PORT_CHECK_RETRIES = 10
MIN_SAFE_PID = 1000  # Never kill PIDs below this threshold (system processes)

# Process names that are safe to kill (our service processes)
SAFE_PROCESS_NAMES = [
    'python', 'python.exe', 'python3', 'python3.exe',
    'uvicorn', 'uvicorn.exe',
    'gunicorn', 'gunicorn.exe',
]

# Global flags
_cleanup_registered = False


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


def get_pid_file_path(port: int = 8001, is_parent: bool = False) -> Path:
    """
    Get path to the PID file for tracking the running service.

    Args:
        port: The port number the service runs on
        is_parent: If True, returns path for parent (reloader) PID file
                   If False, returns path for worker PID file
    """
    suffix = "_parent" if is_parent else "_worker"
    return Path(__file__).parent / f".python_service_{port}{suffix}.pid"


def write_pid_file(port: int = 8001, is_parent: bool = False):
    """
    Write current process ID to PID file.
    Also registers emergency cleanup handler on first write.

    Args:
        port: The port number
        is_parent: True if writing parent PID, False for worker PID
    """
    global _cleanup_registered

    pid_file = get_pid_file_path(port, is_parent)
    current_pid = os.getpid()

    try:
        pid_file.write_text(str(current_pid))
        file_type = "Parent" if is_parent else "Worker"
        log_process(f"{file_type} PID file created: {pid_file} (PID: {current_pid})")

        # Register emergency cleanup only once
        if not _cleanup_registered:
            atexit.register(emergency_cleanup, port)
            _cleanup_registered = True

    except Exception as e:
        log_warning("Process Manager", f"Could not write PID file: {e}")


def remove_pid_file(port: int = 8001, is_parent: bool = False):
    """Remove PID file on shutdown."""
    pid_file = get_pid_file_path(port, is_parent)
    try:
        if pid_file.exists():
            pid_file.unlink()
            file_type = "Parent" if is_parent else "Worker"
            log_process(f"{file_type} PID file removed: {pid_file}")
    except Exception as e:
        log_warning("Process Manager", f"Could not remove PID file: {e}")


def emergency_cleanup(port: int = 8001):
    """
    Emergency cleanup that runs on ANY exit (crash, kill, normal exit).
    Registered via atexit as a fallback when graceful shutdown fails.
    """
    try:
        for is_parent in [True, False]:
            pid_file = get_pid_file_path(port, is_parent)
            if pid_file.exists():
                pid_file.unlink()
    except:
        pass  # Silent fail - this is emergency cleanup


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if pid <= 0:
        return False

    if is_windows_environment():
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return str(pid) in result.stdout
        except:
            return False
    else:
        # Linux: Use kill -0 to check process existence
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def get_process_name(pid: int) -> Optional[str]:
    """
    Get the process name for a given PID.
    Returns None if process doesn't exist or can't be determined.
    """
    if pid <= 0:
        return None

    if is_windows_environment():
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Output format: "process.exe","PID","Session Name","Session#","Mem Usage"
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            for line in lines:
                if line.startswith('"'):
                    parts = line.split('","')
                    if len(parts) >= 2:
                        process_name = parts[0].strip('"').lower()
                        return process_name
            return None
        except Exception:
            return None
    else:
        # Linux: Read from /proc/{pid}/comm
        try:
            with open(f'/proc/{pid}/comm', 'r') as f:
                return f.read().strip().lower()
        except (FileNotFoundError, PermissionError):
            return None


def is_safe_to_kill(pid: int) -> tuple[bool, str]:
    """
    Validate that a PID is safe to kill.

    Returns:
        Tuple of (is_safe: bool, reason: str)

    Safety checks:
    1. PID must be above MIN_SAFE_PID (protects system processes)
    2. Process must be a known safe process type (Python, uvicorn, etc.)
    3. PID must not be the current process
    """
    current_pid = os.getpid()

    # Check 1: Never kill current process
    if pid == current_pid:
        return False, f"PID {pid} is current process"

    # Check 2: Never kill low PIDs (system processes)
    if pid < MIN_SAFE_PID:
        return False, f"PID {pid} is below safe threshold ({MIN_SAFE_PID})"

    # Check 3: Verify process name is in safe list
    process_name = get_process_name(pid)
    if process_name is None:
        return False, f"PID {pid} - could not determine process name"

    # Check if process name matches any safe process
    is_safe = any(safe_name in process_name for safe_name in SAFE_PROCESS_NAMES)

    if is_safe:
        return True, f"PID {pid} ({process_name}) is a safe Python process"
    else:
        return False, f"PID {pid} ({process_name}) is NOT a safe process type"


def kill_process_tree(pid: int, force_tree: bool = False) -> bool:
    """
    Kill a process, optionally including its child processes.

    Args:
        pid: Process ID to kill
        force_tree: If True, kill entire process tree (parent + all children)
                    If False, only kill the single process (safer default)

    On Windows: Uses taskkill with optional /T flag for tree killing
    On Linux: Uses SIGKILL, optionally with process group killing

    SAFETY: Tree killing (/T or killpg) should only be used when force_tree=True,
    which should only happen after validating the process is safe to kill.

    Returns True if successful.
    """
    if pid <= 0:
        return False

    process_name = get_process_name(pid)
    process_info = f"{pid} ({process_name})" if process_name else str(pid)

    if is_windows_environment():
        try:
            # Build taskkill command - only use /T (tree) if explicitly requested
            cmd = ['taskkill', '/F', '/PID', str(pid)]
            kill_mode = "single"

            if force_tree:
                cmd = ['taskkill', '/F', '/T', '/PID', str(pid)]
                kill_mode = "tree"

            log_process(f"Killing process {process_info} (mode: {kill_mode})")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0:
                log_process(f"Process {process_info} terminated (Windows, {kill_mode})")
                return True
            else:
                # Check if process already dead (common race condition)
                if "not found" in result.stderr.lower() or not is_process_running(pid):
                    log_process(f"Process {process_info} already terminated")
                    return True
                log_warning("Process Manager", f"taskkill error for {process_info}: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            log_warning("Process Manager", f"Timeout killing process {process_info}")
            return False
        except Exception as e:
            log_warning("Process Manager", f"Error killing process {process_info}: {e}")
            return False
    else:
        # Linux: Only use process group killing if force_tree is True
        try:
            if force_tree:
                # Kill entire process group (dangerous - only for validated processes)
                try:
                    pgid = os.getpgid(pid)
                    # Extra safety: don't kill process group 1 (init)
                    if pgid <= 1:
                        log_process(f"SAFETY: Refusing to kill process group {pgid}")
                        return False
                    os.killpg(pgid, signal.SIGKILL)
                    log_process(f"Process group {pgid} ({process_info}) terminated (Linux, tree)")
                    return True
                except OSError:
                    # Fallback to single process kill
                    pass

            # Single process kill (safer default)
            os.kill(pid, signal.SIGKILL)
            log_process(f"Process {process_info} terminated (Linux, single)")
            return True

        except ProcessLookupError:
            log_process(f"Process {process_info} already terminated")
            return True
        except OSError as e:
            log_warning("Process Manager", f"Error killing process {process_info}: {e}")
            return False


def get_pids_on_port(port: int) -> list:
    """
    Get all process IDs LISTENING on a local port.

    IMPORTANT: Only matches processes with the port in LOCAL address column,
    NOT in foreign/remote address. This prevents killing unrelated processes
    that happen to have connections TO the port.

    Returns list of unique PIDs.
    """
    pids = set()

    if is_windows_environment():
        try:
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # netstat output format:
            # Proto  Local Address          Foreign Address        State           PID
            # TCP    0.0.0.0:8001           0.0.0.0:0              LISTENING       12345
            # TCP    127.0.0.1:8001         192.168.1.5:54321      ESTABLISHED     12345
            #
            # We ONLY want to match the LOCAL address column (index 1), NOT foreign (index 2)
            for line in result.stdout.split('\n'):
                parts = line.split()
                # Need at least: Proto, Local, Foreign, State, PID
                if len(parts) >= 5:
                    local_address = parts[1]
                    pid_str = parts[-1]

                    # Check if LOCAL address contains our port
                    # Match patterns like :8001, 0.0.0.0:8001, 127.0.0.1:8001, [::]:8001
                    if local_address.endswith(f':{port}') and pid_str.isdigit():
                        pid = int(pid_str)
                        if pid > 0:
                            pids.add(pid)
                            log_process(f"Found PID {pid} listening on local port {port}")

        except Exception as e:
            log_warning("Process Manager", f"Error scanning port {port}: {e}")
    else:
        # Linux: Try lsof first, fall back to ss
        # lsof -ti :{port} only returns processes listening on the local port
        try:
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            for pid_str in result.stdout.strip().split('\n'):
                if pid_str and pid_str.isdigit():
                    pids.add(int(pid_str))

        except FileNotFoundError:
            # lsof not available, try ss (sport = source port, i.e., local listening port)
            try:
                result = subprocess.run(
                    ['ss', '-tlnp', f'sport = :{port}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                for match in re.findall(r'pid=(\d+)', result.stdout):
                    pids.add(int(match))
            except:
                pass
        except Exception as e:
            log_warning("Process Manager", f"Error scanning port {port}: {e}")

    return list(pids)


def is_port_available(port: int) -> bool:
    """
    Check if a port is available for binding.
    Uses actual bind test which is more reliable than connect_ex
    for detecting TIME_WAIT and other lingering states.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True
    except OSError:
        sock.close()
        return False


def kill_existing_service(port: int = 8001):
    """
    Kill any existing process running on the specified port.

    This function:
    1. Checks for PID files (both parent and worker) and kills those processes
    2. Scans for processes LISTENING on the local port (catches orphans)
    3. VALIDATES each process before killing (must be Python/uvicorn)
    4. Uses tree killing ONLY for trusted PIDs from PID files
    5. Retries with increasing waits if processes persist
    6. Cleans up stale PID files

    SAFETY FEATURES:
    - Never kills PIDs below MIN_SAFE_PID (system processes)
    - Validates process name before killing (must be Python/uvicorn)
    - Only uses tree killing for PIDs from our own PID files
    - Logs all kill decisions for debugging

    Works on both Windows (active) and Linux (ready for migration).
    """
    current_pid = os.getpid()
    killed_any = False
    trusted_pids = set()  # PIDs from our PID files - safe to tree-kill
    platform_name = "Windows" if is_windows_environment() else "Linux"

    log_process(f"Checking port {port} ({platform_name} mode)...")

    # Step 0: Handle LEGACY PID file (migration from old format)
    legacy_pid_file = Path(__file__).parent / f".python_service_{port}.pid"
    if legacy_pid_file.exists():
        try:
            old_pid = int(legacy_pid_file.read_text().strip())
            if old_pid != current_pid and old_pid > 0:
                trusted_pids.add(old_pid)
                if is_process_running(old_pid):
                    is_safe, reason = is_safe_to_kill(old_pid)
                    log_process(f"Legacy PID {old_pid}: {reason}")
                    if is_safe:
                        log_process(f"Terminating legacy process (trusted, tree mode)...")
                        if kill_process_tree(old_pid, force_tree=True):
                            killed_any = True
                    else:
                        log_process(f"SAFETY: Skipping legacy PID {old_pid} - validation failed")
                else:
                    log_process(f"Legacy PID file stale (PID {old_pid} not running)")
            legacy_pid_file.unlink()
            log_process(f"Removed legacy PID file")
        except (ValueError, FileNotFoundError):
            pass
        except Exception as e:
            log_process(f"Warning reading legacy PID file: {e}")

    # Step 1: Check and kill processes from PID files (both parent and worker)
    # These are TRUSTED PIDs from our own service - safe to use tree killing
    for is_parent in [True, False]:
        pid_file = get_pid_file_path(port, is_parent)
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                file_type = "parent" if is_parent else "worker"

                if old_pid != current_pid and old_pid > 0:
                    trusted_pids.add(old_pid)
                    if is_process_running(old_pid):
                        is_safe, reason = is_safe_to_kill(old_pid)
                        log_process(f"{file_type.title()} PID {old_pid}: {reason}")

                        if is_safe:
                            # Trusted PID from our file - use tree killing for parent only
                            use_tree = is_parent  # Only tree-kill parent process
                            mode = "tree" if use_tree else "single"
                            log_process(f"Terminating {file_type} process ({mode} mode)...")
                            if kill_process_tree(old_pid, force_tree=use_tree):
                                killed_any = True
                        else:
                            log_process(f"SAFETY: Skipping {file_type} PID {old_pid} - validation failed")
                    else:
                        log_process(f"Stale {file_type} PID file (PID {old_pid} not running)")

                # Always remove stale PID file
                pid_file.unlink()
                log_process(f"Removed stale {file_type} PID file")

            except (ValueError, FileNotFoundError):
                pass
            except Exception as e:
                log_process(f"Warning reading PID file: {e}")

    # Step 2: Kill processes on the port (catches orphans without PID files)
    # AGGRESSIVE MODE: Kill ANY process on the port - we need this port to start
    for attempt in range(PROCESS_KILL_MAX_RETRIES):
        pids = get_pids_on_port(port)
        # Filter out current process and already-trusted PIDs
        pids = [p for p in pids if p != current_pid and p > 0 and p not in trusted_pids]

        if not pids:
            if attempt == 0 and not killed_any:
                log_process(f"Port {port} is available.")
            elif killed_any:
                log_process(f"Port {port} is now available.")
            break

        log_process(f"Found {len(pids)} orphan process(es) on port {port}: {pids}")

        for pid in pids:
            is_safe, reason = is_safe_to_kill(pid)
            log_process(f"Orphan PID {pid}: {reason}")

            # On first attempt, respect safety checks
            # On subsequent attempts, force kill to ensure port is freed
            force_kill = attempt > 0

            if is_safe or force_kill:
                # Use tree killing to ensure all child processes are also terminated
                mode = "FORCE tree" if force_kill and not is_safe else "tree"
                log_process(f"Terminating orphan process ({mode} mode)...")
                if kill_process_tree(pid, force_tree=True):
                    killed_any = True
            else:
                log_process(f"Skipping PID {pid} on first attempt - will force kill if port still blocked")

        # Wait for processes to fully exit and port to be released
        log_process(f"Waiting {PROCESS_KILL_WAIT_SECONDS}s for port release...")
        time.sleep(PROCESS_KILL_WAIT_SECONDS)

        # Verify port is now free using bind test
        if is_port_available(port):
            log_process(f"Port {port} is now available.")
            break
        elif attempt < PROCESS_KILL_MAX_RETRIES - 1:
            log_process(f"Port {port} still in use, forcing aggressive cleanup... ({attempt + 2}/{PROCESS_KILL_MAX_RETRIES})")
    else:
        # FINAL ATTEMPT: Force kill everything on the port using taskkill directly
        pids = get_pids_on_port(port)
        pids = [p for p in pids if p != current_pid and p > 0]
        if pids:
            log_process(f"FINAL CLEANUP: Force killing remaining processes on port {port}")
            for pid in pids:
                process_name = get_process_name(pid)
                log_process(f"  Force killing PID {pid}: {process_name or 'unknown'}")
                try:
                    if is_windows_environment():
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                                      capture_output=True, timeout=10)
                    else:
                        os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    log_process(f"  Failed to kill PID {pid}: {e}")
            time.sleep(2)

            # Final check
            if not is_port_available(port):
                log_process(f"WARNING: Port {port} still not available after all attempts!")


def setup_worker_signal_handlers(port: int = 8001):
    """
    Setup signal handlers for the WORKER process.
    Must be called from within the lifespan context manager (worker process).
    """
    def cleanup_handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        print(f"\n[Worker] Received {sig_name}, cleaning up...")
        remove_pid_file(port, is_parent=False)
        # Don't call sys.exit() here - it disrupts the asyncio event loop
        # Instead, raise KeyboardInterrupt to let uvicorn handle shutdown gracefully
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)

    if not is_windows_environment():
        try:
            signal.signal(signal.SIGHUP, cleanup_handler)
        except (AttributeError, ValueError):
            pass


def setup_parent_signal_handlers(port: int = 8001):
    """
    Setup signal handlers for the PARENT (reloader) process.
    Must be called from __main__ before uvicorn.run().
    """
    def cleanup_handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        print(f"\n[Parent] Received {sig_name}, cleaning up...")
        remove_pid_file(port, is_parent=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)

    if not is_windows_environment():
        try:
            signal.signal(signal.SIGHUP, cleanup_handler)
        except (AttributeError, ValueError):
            pass
