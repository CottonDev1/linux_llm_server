"""
Environment Agent - Infrastructure and Service Management
=========================================================

Manages the runtime environment for the LLM Website application:
- Service lifecycle (start/stop/restart)
- Health monitoring
- MongoDB connection management
- LLM availability checks

This agent supports both the Test Agent (for E2E testing) and
overnight training pipelines that need reliable infrastructure.
"""

import asyncio
import platform
import logging
import time
import subprocess
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse
import aiohttp

# Import config for MongoDB settings
from config import MONGODB_URI

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Status of a managed service."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


class ShellType(str, Enum):
    """Shell types for command execution."""
    BASH = "bash"
    POWERSHELL = "powershell"
    CMD = "cmd"


@dataclass
class ServiceConfig:
    """Configuration for a managed service."""
    name: str
    display_name: str
    port: int
    health_endpoint: str
    command: Optional[str] = None  # None for Windows services
    working_dir: Optional[str] = None
    is_windows_service: bool = False
    service_name: Optional[str] = None  # Windows service name
    startup_timeout: int = 30  # seconds
    required: bool = True  # Is this service required for operation?


@dataclass
class ServiceHealth:
    """Health status of a service."""
    name: str
    status: ServiceStatus
    port: int
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentStatus:
    """Overall environment status."""
    healthy: bool
    services: List[ServiceHealth]
    mongodb_connected: bool
    llm_available: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "services": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "port": s.port,
                    "latency_ms": s.latency_ms,
                    "error": s.error,
                }
                for s in self.services
            ],
            "mongodb_connected": self.mongodb_connected,
            "llm_available": self.llm_available,
            "timestamp": self.timestamp.isoformat(),
            "warnings": self.warnings,
        }


class EnvironmentAgent:
    """
    Manages the runtime environment for the LLM Website application.

    Capabilities:
    - SERVICE_START: Start services (Node.js, Python, LLMs)
    - SERVICE_STOP: Graceful service shutdown
    - SERVICE_HEALTH: Health check all services
    - MONGO_CONNECT: Verify/maintain MongoDB connection
    - LLM_AVAILABILITY: Ensure LLM models are responding
    - ENV_VALIDATE: Full environment validation
    """

    # Default service configurations
    DEFAULT_SERVICES: Dict[str, ServiceConfig] = {
        "nodejs": ServiceConfig(
            name="nodejs",
            display_name="Node.js Server",
            port=3000,
            health_endpoint="/health",
            command="node rag-server.js",
            working_dir="/mnt/c/Projects/llm_website",
            required=True,
        ),
        "python_api": ServiceConfig(
            name="python_api",
            display_name="Python FastAPI",
            port=8001,
            health_endpoint="/health",
            command="python -m uvicorn main:app --host 0.0.0.0 --port 8001",
            working_dir="/mnt/c/Projects/llm_website/python_services",
            required=True,
        ),
        "llamacpp_sql": ServiceConfig(
            name="llamacpp_sql",
            display_name="llama.cpp SQL Model",
            port=8080,
            health_endpoint="/health",
            is_windows_service=True,
            service_name="LlamaCppSqlService",
            required=True,
        ),
        "llamacpp_general": ServiceConfig(
            name="llamacpp_general",
            display_name="llama.cpp General Model",
            port=8081,
            health_endpoint="/health",
            is_windows_service=True,
            service_name="LlamaCppGeneralService",
            required=False,
        ),
        "llamacpp_code": ServiceConfig(
            name="llamacpp_code",
            display_name="llama.cpp Code Model",
            port=8082,
            health_endpoint="/health",
            is_windows_service=True,
            service_name="LlamaCppCodeService",
            required=False,
        ),
    }

    # MongoDB configuration (parsed from config.MONGODB_URI)
    _parsed_uri = urlparse(MONGODB_URI)
    MONGODB_HOST = _parsed_uri.hostname or "EWRSPT-AI"
    MONGODB_PORT = _parsed_uri.port or 27017

    def __init__(
        self,
        services: Optional[Dict[str, ServiceConfig]] = None,
        mongodb_host: str = None,
        mongodb_port: int = None,
    ):
        """
        Initialize the Environment Agent.

        Args:
            services: Custom service configurations (defaults used if None)
            mongodb_host: MongoDB hostname (from config.MONGODB_URI)
            mongodb_port: MongoDB port (from config.MONGODB_URI)
        """
        self.services = services or self.DEFAULT_SERVICES.copy()
        self.mongodb_host = mongodb_host or self.MONGODB_HOST
        self.mongodb_port = mongodb_port or self.MONGODB_PORT
        self._running_processes: Dict[str, subprocess.Popen] = {}
        self._is_windows = platform.system() == "Windows"
        self._default_shell = ShellType.POWERSHELL if self._is_windows else ShellType.BASH

    # =========================================================================
    # Service Lifecycle Management
    # =========================================================================

    async def start_service(self, service_name: str) -> ServiceHealth:
        """
        Start a specific service.

        Args:
            service_name: Name of the service to start

        Returns:
            ServiceHealth with current status
        """
        if service_name not in self.services:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.ERROR,
                port=0,
                error=f"Unknown service: {service_name}",
            )

        config = self.services[service_name]
        logger.info(f"Starting service: {config.display_name}")

        # Check if already running
        health = await self.check_service_health(service_name)
        if health.status == ServiceStatus.RUNNING:
            logger.info(f"Service {config.display_name} already running")
            return health

        try:
            if config.is_windows_service:
                await self._start_windows_service(config)
            else:
                await self._start_process(config)

            # Wait for service to become healthy
            return await self._wait_for_healthy(config)

        except Exception as e:
            logger.error(f"Failed to start {config.display_name}: {e}")
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.ERROR,
                port=config.port,
                error=str(e),
            )

    async def stop_service(self, service_name: str) -> ServiceHealth:
        """
        Stop a specific service.

        Args:
            service_name: Name of the service to stop

        Returns:
            ServiceHealth with current status
        """
        if service_name not in self.services:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.ERROR,
                port=0,
                error=f"Unknown service: {service_name}",
            )

        config = self.services[service_name]
        logger.info(f"Stopping service: {config.display_name}")

        try:
            if config.is_windows_service:
                await self._stop_windows_service(config)
            else:
                await self._stop_process(service_name)

            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.STOPPED,
                port=config.port,
            )

        except Exception as e:
            logger.error(f"Failed to stop {config.display_name}: {e}")
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.ERROR,
                port=config.port,
                error=str(e),
            )

    async def restart_service(self, service_name: str) -> ServiceHealth:
        """
        Restart a specific service.

        Args:
            service_name: Name of the service to restart

        Returns:
            ServiceHealth with current status
        """
        await self.stop_service(service_name)
        await asyncio.sleep(2)  # Brief pause between stop and start
        return await self.start_service(service_name)

    async def start_all_required(self) -> EnvironmentStatus:
        """
        Start all required services.

        Returns:
            EnvironmentStatus with all service states
        """
        results = []

        for name, config in self.services.items():
            if config.required:
                health = await self.start_service(name)
                results.append(health)

        # Check MongoDB
        mongo_ok = await self.check_mongodb_connection()

        # Check LLM
        llm_ok = await self.check_llm_availability()

        all_healthy = all(
            r.status == ServiceStatus.RUNNING
            for r in results
            if self.services[r.name].required
        )

        return EnvironmentStatus(
            healthy=all_healthy and mongo_ok and llm_ok,
            services=results,
            mongodb_connected=mongo_ok,
            llm_available=llm_ok,
        )

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """
        Check health of a specific service.

        Args:
            service_name: Name of the service to check

        Returns:
            ServiceHealth with current status
        """
        if service_name not in self.services:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.ERROR,
                port=0,
                error=f"Unknown service: {service_name}",
            )

        config = self.services[service_name]
        url = f"http://localhost:{config.port}{config.health_endpoint}"

        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    latency = (time.time() - start_time) * 1000

                    if resp.status == 200:
                        return ServiceHealth(
                            name=service_name,
                            status=ServiceStatus.RUNNING,
                            port=config.port,
                            latency_ms=latency,
                        )
                    else:
                        return ServiceHealth(
                            name=service_name,
                            status=ServiceStatus.ERROR,
                            port=config.port,
                            latency_ms=latency,
                            error=f"Health check returned status {resp.status}",
                        )

        except aiohttp.ClientConnectorError:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.STOPPED,
                port=config.port,
                error="Connection refused - service not running",
            )
        except asyncio.TimeoutError:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.ERROR,
                port=config.port,
                error="Health check timed out",
            )
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNKNOWN,
                port=config.port,
                error=str(e),
            )

    async def check_all_services(self) -> List[ServiceHealth]:
        """
        Check health of all configured services.

        Returns:
            List of ServiceHealth for all services
        """
        tasks = [
            self.check_service_health(name)
            for name in self.services.keys()
        ]
        return await asyncio.gather(*tasks)

    async def check_mongodb_connection(self) -> bool:
        """
        Check MongoDB connectivity.

        Returns:
            True if MongoDB is accessible
        """
        try:
            # Use pymongo if available, otherwise try socket connection
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.mongodb_host, self.mongodb_port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.warning(f"MongoDB connection check failed: {e}")
            return False

    async def check_llm_availability(self) -> bool:
        """
        Check if at least one LLM is available.

        Returns:
            True if any LLM service is responding
        """
        llm_services = ["llamacpp_sql", "llamacpp_general", "llamacpp_code"]

        for service_name in llm_services:
            if service_name in self.services:
                health = await self.check_service_health(service_name)
                if health.status == ServiceStatus.RUNNING:
                    return True

        return False

    async def validate_environment(self) -> EnvironmentStatus:
        """
        Perform full environment validation.

        Returns:
            EnvironmentStatus with all checks
        """
        services = await self.check_all_services()
        mongo_ok = await self.check_mongodb_connection()
        llm_ok = await self.check_llm_availability()

        warnings = []

        # Check required services
        for health in services:
            config = self.services.get(health.name)
            if config and config.required and health.status != ServiceStatus.RUNNING:
                warnings.append(f"Required service {health.name} is not running")

        if not mongo_ok:
            warnings.append("MongoDB is not accessible")

        if not llm_ok:
            warnings.append("No LLM services are available")

        all_required_running = all(
            h.status == ServiceStatus.RUNNING
            for h in services
            if self.services.get(h.name, ServiceConfig("", "", 0, "")).required
        )

        return EnvironmentStatus(
            healthy=all_required_running and mongo_ok and llm_ok,
            services=services,
            mongodb_connected=mongo_ok,
            llm_available=llm_ok,
            warnings=warnings,
        )

    # =========================================================================
    # Environment Setup for Testing
    # =========================================================================

    async def ensure_test_environment(self) -> EnvironmentStatus:
        """
        Ensure the environment is ready for E2E testing.

        This method:
        1. Checks all services
        2. Starts any stopped required services
        3. Validates MongoDB connection
        4. Returns final environment status

        Returns:
            EnvironmentStatus indicating readiness
        """
        logger.info("Ensuring test environment is ready...")

        # First, check current status
        status = await self.validate_environment()

        if status.healthy:
            logger.info("Environment is already healthy")
            return status

        # Try to start any stopped required services
        for health in status.services:
            config = self.services.get(health.name)
            if config and config.required and health.status != ServiceStatus.RUNNING:
                logger.info(f"Starting required service: {config.display_name}")
                await self.start_service(health.name)

        # Re-validate
        return await self.validate_environment()

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _start_windows_service(self, config: ServiceConfig) -> None:
        """Start a Windows service."""
        if not config.service_name:
            raise ValueError(f"No service_name configured for {config.name}")

        cmd = f'powershell.exe -Command "Start-Service -Name \'{config.service_name}\'"'
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Failed to start Windows service: {stderr.decode()}")

    async def _stop_windows_service(self, config: ServiceConfig) -> None:
        """Stop a Windows service."""
        if not config.service_name:
            raise ValueError(f"No service_name configured for {config.name}")

        cmd = f'powershell.exe -Command "Stop-Service -Name \'{config.service_name}\' -Force"'
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

    async def _start_process(self, config: ServiceConfig) -> None:
        """Start a process-based service."""
        if not config.command:
            raise ValueError(f"No command configured for {config.name}")

        working_dir = config.working_dir
        if working_dir and working_dir.startswith("/mnt/c"):
            # Convert WSL path to Windows path for subprocess
            working_dir = working_dir.replace("/mnt/c", "C:")

        # Start in background
        if self._is_windows:
            cmd = f'powershell.exe -Command "Start-Process -FilePath cmd -ArgumentList \'/c {config.command}\' -WorkingDirectory \'{working_dir}\' -WindowStyle Hidden"'
        else:
            cmd = f"cd {config.working_dir} && {config.command} &"

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._running_processes[config.name] = process

    async def _stop_process(self, service_name: str) -> None:
        """Stop a process-based service."""
        config = self.services[service_name]

        # Try to kill by port
        if self._is_windows:
            cmd = f'powershell.exe -Command "Get-Process | Where-Object {{$_.Id -in (Get-NetTCPConnection -LocalPort {config.port} -ErrorAction SilentlyContinue).OwningProcess}} | Stop-Process -Force"'
        else:
            cmd = f"fuser -k {config.port}/tcp 2>/dev/null || true"

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        # Remove from tracked processes
        self._running_processes.pop(service_name, None)

    async def _wait_for_healthy(
        self,
        config: ServiceConfig,
        check_interval: float = 1.0,
    ) -> ServiceHealth:
        """Wait for a service to become healthy."""
        deadline = time.time() + config.startup_timeout

        while time.time() < deadline:
            health = await self.check_service_health(config.name)
            if health.status == ServiceStatus.RUNNING:
                return health
            await asyncio.sleep(check_interval)

        return ServiceHealth(
            name=config.name,
            status=ServiceStatus.ERROR,
            port=config.port,
            error=f"Service did not become healthy within {config.startup_timeout}s",
        )

    # =========================================================================
    # Shell Execution (for advanced operations)
    # =========================================================================

    async def execute_shell(
        self,
        command: str,
        shell_type: ShellType = None,
        working_dir: str = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            shell_type: Type of shell (defaults to platform default)
            working_dir: Working directory for the command
            timeout: Command timeout in seconds

        Returns:
            Dict with stdout, stderr, return_code
        """
        shell_type = shell_type or self._default_shell

        # Security check - block dangerous commands
        blocked_patterns = [
            "rm -rf /",
            ":(){ :|:& };:",  # Fork bomb
            "dd if=/dev/zero",
            "mkfs.",
            "> /dev/sda",
        ]

        for pattern in blocked_patterns:
            if pattern in command:
                return {
                    "stdout": "",
                    "stderr": f"Blocked dangerous command pattern: {pattern}",
                    "return_code": 1,
                    "success": False,
                }

        # Build shell command
        if shell_type == ShellType.POWERSHELL:
            full_cmd = f'powershell.exe -Command "{command}"'
        elif shell_type == ShellType.CMD:
            full_cmd = f'cmd.exe /c "{command}"'
        else:
            full_cmd = f'bash -c "{command}"'

        try:
            process = await asyncio.create_subprocess_shell(
                full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode,
                "success": process.returncode == 0,
            }

        except asyncio.TimeoutError:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "return_code": -1,
                "success": False,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False,
            }


# Convenience functions for quick access
async def ensure_environment_ready() -> EnvironmentStatus:
    """Ensure the environment is ready for operation."""
    agent = EnvironmentAgent()
    return await agent.ensure_test_environment()


async def check_environment_health() -> EnvironmentStatus:
    """Quick health check of the environment."""
    agent = EnvironmentAgent()
    return await agent.validate_environment()
