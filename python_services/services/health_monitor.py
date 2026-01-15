"""
Health Monitor Service

Centralized health monitoring that runs checks on a schedule and caches results.
Browser clients fetch cached status rather than triggering expensive checks.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from core.log_utils import log_health, log_error


@dataclass
class HealthCheckResult:
    """Cached health check result"""
    status: Dict[str, Any]
    checked_at: datetime
    check_duration_ms: float


@dataclass
class HealthMonitor:
    """
    Singleton health monitor that runs background health checks.

    Intervals are controlled server-side, not by browser clients.
    """
    # Check intervals (in seconds) - both every 2 minutes
    python_service_interval: int = 120  # MongoDB, embeddings every 2 minutes
    llm_interval: int = 120  # LLM backends every 2 minutes

    # Cached results
    _python_health: Optional[HealthCheckResult] = field(default=None, init=False)
    _llm_health: Optional[HealthCheckResult] = field(default=None, init=False)

    # Background tasks
    _python_task: Optional[asyncio.Task] = field(default=None, init=False)
    _llm_task: Optional[asyncio.Task] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)

    _instance: Optional['HealthMonitor'] = field(default=None, init=False, repr=False)

    @classmethod
    def get_instance(cls) -> 'HealthMonitor':
        """Get singleton instance"""
        if not hasattr(cls, '_singleton') or cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    async def start(self):
        """Start background health check tasks"""
        if self._running:
            return

        self._running = True
        log_health("Starting background health checks")
        log_health(f"Python/MongoDB checks: every {self.python_service_interval}s")
        log_health(f"LLM backend checks: every {self.llm_interval}s")

        # Run initial checks immediately
        await self._run_python_health_check()
        await self._run_llm_health_check()

        # Start background tasks
        self._python_task = asyncio.create_task(self._python_health_loop())
        self._llm_task = asyncio.create_task(self._llm_health_loop())

    async def stop(self):
        """Stop background health check tasks"""
        self._running = False

        if self._python_task:
            self._python_task.cancel()
            try:
                await self._python_task
            except asyncio.CancelledError:
                pass

        if self._llm_task:
            self._llm_task.cancel()
            try:
                await self._llm_task
            except asyncio.CancelledError:
                pass

        log_health("Stopped")

    async def _python_health_loop(self):
        """Background loop for Python service health checks"""
        while self._running:
            try:
                await asyncio.sleep(self.python_service_interval)
                if self._running:
                    await self._run_python_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Health Monitor", f"Python health check error: {e}")

    async def _llm_health_loop(self):
        """Background loop for LLM health checks"""
        while self._running:
            try:
                await asyncio.sleep(self.llm_interval)
                if self._running:
                    await self._run_llm_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Health Monitor", f"LLM health check error: {e}")

    async def _run_python_health_check(self):
        """Run Python service health check and cache result"""
        start_time = datetime.now()

        try:
            from mongodb import get_mongodb_service
            mongodb = get_mongodb_service()

            # Log what we're checking
            log_health("Checking: MongoDB connection, Vector search, Embedding model")

            result = await mongodb.health_check()

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            self._python_health = HealthCheckResult(
                status=result,
                checked_at=datetime.now(),
                check_duration_ms=duration_ms
            )

            # Log summary
            mongo_connected = result.get("mongodb_connected", False)
            vector_search = result.get("vector_search", {})
            vector_status = vector_search.get("status", "unknown") if vector_search else "unknown"
            embedding_loaded = result.get("embedding_model_loaded", False)

            log_health(f"Results: MongoDB={mongo_connected}, "
                       f"VectorSearch={vector_status}, Embeddings={embedding_loaded} "
                       f"({duration_ms:.0f}ms)")

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_error("Health Monitor", f"Python service check failed: {e}")

            self._python_health = HealthCheckResult(
                status={
                    "status": "unhealthy",
                    "mongodb_connected": False,
                    "embedding_model_loaded": False,
                    "vector_search": {
                        "enabled": True,
                        "native_available": False,
                        "status": "unavailable",
                        "error": str(e)
                    },
                    "collections": {},
                    "error": str(e)
                },
                checked_at=datetime.now(),
                check_duration_ms=duration_ms
            )

    async def _run_llm_health_check(self):
        """Run LLM health check and cache result"""
        start_time = datetime.now()

        try:
            from services.llm_service import get_llm_service

            # Log what we're checking
            log_health("Checking: LLM backends (SQL model, General model, Code model)")

            service = await get_llm_service()
            result = await service.health_check()

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            self._llm_health = HealthCheckResult(
                status=result,
                checked_at=datetime.now(),
                check_duration_ms=duration_ms
            )

            # Log summary
            healthy = result.get("healthy", False)
            endpoints = result.get("endpoints", {})

            endpoint_status = []
            for name, info in endpoints.items():
                ep_status = "OK" if info.get("healthy", False) else "DOWN"
                endpoint_status.append(f"{name}={ep_status}")

            status_str = ", ".join(endpoint_status) if endpoint_status else "No endpoints"
            log_health(f"Results: LLM healthy={healthy}, {status_str} ({duration_ms:.0f}ms)")

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_error("Health Monitor", f"LLM check failed: {e}")

            self._llm_health = HealthCheckResult(
                status={
                    "healthy": False,
                    "host": "",
                    "models_available": 0,
                    "models": [],
                    "configured_model": "",
                    "sql_model": "",
                    "error": str(e),
                    "use_dedicated_endpoints": True,
                    "endpoints": {}
                },
                checked_at=datetime.now(),
                check_duration_ms=duration_ms
            )

    def get_python_health(self) -> Dict[str, Any]:
        """Get cached Python service health status"""
        if self._python_health is None:
            return {
                "status": "unknown",
                "mongodb_connected": False,
                "embedding_model_loaded": False,
                "vector_search": None,
                "collections": {},
                "cached": False,
                "message": "Health check not yet completed"
            }

        result = dict(self._python_health.status)
        result["cached"] = True
        result["checked_at"] = self._python_health.checked_at.isoformat()
        result["check_duration_ms"] = self._python_health.check_duration_ms
        result["next_check_in"] = self._get_next_check_seconds(
            self._python_health.checked_at,
            self.python_service_interval
        )
        return result

    def get_llm_health(self) -> Dict[str, Any]:
        """Get cached LLM health status"""
        if self._llm_health is None:
            return {
                "healthy": False,
                "host": "",
                "models_available": 0,
                "models": [],
                "configured_model": "",
                "sql_model": "",
                "error": "Health check not yet completed",
                "use_dedicated_endpoints": True,
                "endpoints": {},
                "cached": False,
                "message": "Health check not yet completed"
            }

        result = dict(self._llm_health.status)
        result["cached"] = True
        result["checked_at"] = self._llm_health.checked_at.isoformat()
        result["check_duration_ms"] = self._llm_health.check_duration_ms
        result["next_check_in"] = self._get_next_check_seconds(
            self._llm_health.checked_at,
            self.llm_interval
        )
        return result

    def _get_next_check_seconds(self, last_check: datetime, interval: int) -> int:
        """Calculate seconds until next check"""
        elapsed = (datetime.now() - last_check).total_seconds()
        remaining = max(0, interval - elapsed)
        return int(remaining)

    async def force_refresh(self, check_type: str = "all"):
        """Force an immediate health check refresh"""
        if check_type in ("all", "python"):
            log_health("Force refresh: Python service")
            await self._run_python_health_check()
        if check_type in ("all", "llm"):
            log_health("Force refresh: LLM backends")
            await self._run_llm_health_check()


def get_health_monitor() -> HealthMonitor:
    """Get the singleton health monitor instance"""
    return HealthMonitor.get_instance()
