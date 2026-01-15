"""
EWR Agent Registry
==================

Central registry for agent discovery and coordination.
Supports:
- Local in-memory registry (default)
- Remote registry via HTTP API
"""

import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
import aiohttp

from .models import AgentInfo, AgentState, AgentCapability, AgentType


# Global singleton registry
_global_registry: Optional["AgentRegistry"] = None


class AgentRegistry:
    """
    Central registry for agent discovery.

    Agents register themselves on startup and send periodic heartbeats.
    Other agents can discover available agents by capability or type.

    Example:
        registry = AgentRegistry()

        # Register an agent
        await registry.register(agent_info)

        # Find agents with code analysis capability
        code_agents = await registry.find_by_capability("code_analyze")

        # Find all idle code agents
        idle_agents = await registry.find_by_capability(
            "code_analyze",
            state=AgentState.IDLE
        )
    """

    def __init__(
        self,
        remote_url: str = None,
        stale_timeout: int = 60,
        cleanup_interval: int = 30,
    ):
        """
        Initialize the registry.

        Args:
            remote_url: Optional URL of remote registry service
            stale_timeout: Seconds before an agent is considered stale (no heartbeat)
            cleanup_interval: Seconds between automatic stale agent cleanup
        """
        self._agents: Dict[str, AgentInfo] = {}
        self._remote_url = remote_url
        self._stale_timeout = stale_timeout
        self._cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self.logger = logging.getLogger("ewr.agent.registry")

    async def start(self) -> None:
        """Start the registry (enables automatic cleanup)."""
        self._running = True
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="registry-cleanup"
        )
        self.logger.info("Registry started")

    async def stop(self) -> None:
        """Stop the registry."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Registry stopped")

    async def register(self, agent: AgentInfo) -> bool:
        """
        Register an agent.

        Args:
            agent: AgentInfo to register

        Returns:
            True if registration successful
        """
        if self._remote_url:
            return await self._remote_register(agent)

        async with self._lock:
            agent.last_heartbeat = datetime.utcnow()
            agent.registered_at = datetime.utcnow()
            self._agents[agent.agent_id] = agent
            self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
            return True

    async def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: ID of agent to unregister

        Returns:
            True if agent was found and removed
        """
        if self._remote_url:
            return await self._remote_unregister(agent_id)

        async with self._lock:
            if agent_id in self._agents:
                agent = self._agents.pop(agent_id)
                self.logger.info(f"Unregistered agent: {agent.name} ({agent_id})")
                return True
            return False

    async def heartbeat(self, agent_id: str) -> bool:
        """
        Update agent heartbeat.

        Args:
            agent_id: ID of agent sending heartbeat

        Returns:
            True if heartbeat recorded
        """
        if self._remote_url:
            return await self._remote_heartbeat(agent_id)

        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = datetime.utcnow()
                return True
            return False

    async def update_state(
        self,
        agent_id: str,
        state: AgentState,
        current_task_id: str = None
    ) -> bool:
        """
        Update agent state.

        Args:
            agent_id: ID of agent
            state: New state
            current_task_id: Optional current task ID

        Returns:
            True if state updated
        """
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].state = state
                self._agents[agent_id].current_task_id = current_task_id
                return True
            return False

    async def find_by_capability(
        self,
        capability: Union[str, AgentCapability],
        state: AgentState = None,
        exclude_ids: List[str] = None,
    ) -> List[AgentInfo]:
        """
        Find agents with a specific capability.

        Args:
            capability: Capability to search for (string or enum)
            state: Optional filter by state
            exclude_ids: Optional list of agent IDs to exclude

        Returns:
            List of matching AgentInfo
        """
        if isinstance(capability, AgentCapability):
            capability = capability.value

        exclude_ids = exclude_ids or []
        results = []

        async with self._lock:
            for agent in self._agents.values():
                # Check capability
                if not agent.has_capability(capability):
                    continue

                # Check state if specified
                if state and agent.state != state:
                    continue

                # Check exclusions
                if agent.agent_id in exclude_ids:
                    continue

                # Check not stale
                if self._is_stale(agent):
                    continue

                results.append(agent)

        # Sort by tasks completed (prefer more experienced agents)
        results.sort(key=lambda a: a.tasks_completed, reverse=True)
        return results

    async def find_by_type(
        self,
        agent_type: Union[str, AgentType],
        state: AgentState = None,
    ) -> List[AgentInfo]:
        """
        Find agents by type.

        Args:
            agent_type: Type to search for
            state: Optional filter by state

        Returns:
            List of matching AgentInfo
        """
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)

        results = []
        async with self._lock:
            for agent in self._agents.values():
                if agent.agent_type != agent_type:
                    continue
                if state and agent.state != state:
                    continue
                if not self._is_stale(agent):
                    results.append(agent)

        return results

    async def get_all(self, include_stale: bool = False) -> List[AgentInfo]:
        """
        Get all registered agents.

        Args:
            include_stale: Include agents that haven't sent heartbeat

        Returns:
            List of all AgentInfo
        """
        async with self._lock:
            if include_stale:
                return list(self._agents.values())
            return [a for a in self._agents.values() if not self._is_stale(a)]

    async def get(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get a specific agent by ID.

        Args:
            agent_id: Agent ID to look up

        Returns:
            AgentInfo or None if not found
        """
        async with self._lock:
            return self._agents.get(agent_id)

    async def cleanup_stale(self, max_age_seconds: int = None) -> int:
        """
        Remove agents that haven't sent heartbeat.

        Args:
            max_age_seconds: Override default stale timeout

        Returns:
            Number of agents removed
        """
        max_age = max_age_seconds or self._stale_timeout
        cutoff = datetime.utcnow() - timedelta(seconds=max_age)

        async with self._lock:
            stale = [
                aid for aid, agent in self._agents.items()
                if agent.last_heartbeat < cutoff
            ]
            for aid in stale:
                agent = self._agents.pop(aid)
                self.logger.info(f"Removed stale agent: {agent.name} ({aid})")

            return len(stale)

    async def get_stats(self) -> Dict:
        """
        Get registry statistics.

        Returns:
            Dict with counts by type and state
        """
        async with self._lock:
            by_type = {}
            by_state = {}
            total = len(self._agents)
            stale = 0

            for agent in self._agents.values():
                # Count by type
                type_key = agent.agent_type.value
                by_type[type_key] = by_type.get(type_key, 0) + 1

                # Count by state
                state_key = agent.state.value
                by_state[state_key] = by_state.get(state_key, 0) + 1

                # Count stale
                if self._is_stale(agent):
                    stale += 1

            return {
                "total": total,
                "active": total - stale,
                "stale": stale,
                "by_type": by_type,
                "by_state": by_state,
            }

    def _is_stale(self, agent: AgentInfo) -> bool:
        """Check if an agent is stale (no recent heartbeat)."""
        age = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
        return age > self._stale_timeout

    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale agents."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                removed = await self.cleanup_stale()
                if removed:
                    self.logger.debug(f"Cleaned up {removed} stale agents")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Cleanup error: {e}")

    # Remote registry methods (for distributed setup)

    async def _remote_register(self, agent: AgentInfo) -> bool:
        """Register with remote registry service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._remote_url}/agents",
                    json=agent.model_dump(mode="json"),
                ) as resp:
                    return resp.status == 200 or resp.status == 201
        except Exception as e:
            self.logger.error(f"Remote register failed: {e}")
            return False

    async def _remote_unregister(self, agent_id: str) -> bool:
        """Unregister from remote registry service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self._remote_url}/agents/{agent_id}"
                ) as resp:
                    return resp.status == 200 or resp.status == 204
        except Exception as e:
            self.logger.error(f"Remote unregister failed: {e}")
            return False

    async def _remote_heartbeat(self, agent_id: str) -> bool:
        """Send heartbeat to remote registry service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._remote_url}/agents/{agent_id}/heartbeat"
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            self.logger.warning(f"Remote heartbeat failed: {e}")
            return False


def get_registry(
    remote_url: str = None,
    create_if_missing: bool = True,
) -> AgentRegistry:
    """
    Get the global registry singleton.

    Args:
        remote_url: Optional remote registry URL
        create_if_missing: Create registry if it doesn't exist

    Returns:
        AgentRegistry instance
    """
    global _global_registry

    if _global_registry is None and create_if_missing:
        _global_registry = AgentRegistry(remote_url=remote_url)

    return _global_registry


def set_registry(registry: AgentRegistry) -> None:
    """
    Set the global registry singleton.

    Args:
        registry: Registry instance to use globally
    """
    global _global_registry
    _global_registry = registry
