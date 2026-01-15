"""
EWR Agent Core
==============

Core library for the EWR Agent framework providing:
- Base agent class for building specialized agents
- Pydantic models for agent communication
- Agent registry for discovery and coordination
- Message broker for inter-agent communication
- Pluggable LLM backends (OpenAI, Anthropic)

Usage:
    from ewr_agent_core import BaseAgent, AgentRegistry, LLMBackend
    from ewr_agent_core.llm_backends import OpenAIBackend

    class MyAgent(BaseAgent):
        async def process(self, task):
            response = await self.llm.generate(task.prompt)
            return response
"""

from .models import (
    AgentType,
    AgentState,
    AgentCapability,
    AgentInfo,
    AgentMessage,
    MessageType,
    MessagePriority,
    DelegationRequest,
    DelegationResponse,
    TaskStatus,
    TaskResult,
)
from .base_agent import BaseAgent
from .registry import AgentRegistry, get_registry
from .message_broker import MessageBroker, get_broker
from .config import AgentConfig, load_config

__version__ = "1.0.0"

__all__ = [
    # Models
    "AgentType",
    "AgentState",
    "AgentCapability",
    "AgentInfo",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "DelegationRequest",
    "DelegationResponse",
    "TaskStatus",
    "TaskResult",
    # Core classes
    "BaseAgent",
    "AgentRegistry",
    "get_registry",
    "MessageBroker",
    "get_broker",
    "AgentConfig",
    "load_config",
]
