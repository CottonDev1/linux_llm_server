"""
Port management utilities.

Provides utilities for checking port availability and finding processes
listening on specific ports. Works on both Windows and Linux/WSL.
"""
import re
import socket
import subprocess
from typing import List

from .platform_utils import is_windows_environment
from .log_utils import log_process, log_warning


def get_pids_on_port(port: int) -> List[int]:
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
