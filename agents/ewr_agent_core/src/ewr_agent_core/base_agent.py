"""
EWR Agent Base Class
====================

Abstract base class that all agents inherit from.
Provides:
- Auto-registration with registry
- Message queue integration
- Heartbeat management
- Inter-agent delegation via request_help()
"""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING

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
    TaskResult,
    TaskStatus,
)
from .config import AgentConfig

if TYPE_CHECKING:
    from .registry import AgentRegistry
    from .message_broker import MessageBroker
    from .llm_backends.base import LLMBackend


class BaseAgent(ABC):
    """
    Abstract base class for all EWR agents.

    Subclasses must implement:
    - agent_type: Property returning the AgentType enum
    - capabilities: Property returning list of AgentCapability
    - _initialize(): Async method for agent-specific setup
    - handle_task(): Async method to process incoming tasks

    Example:
        class MyCodeAgent(BaseAgent):
            @property
            def agent_type(self) -> AgentType:
                return AgentType.CODE

            @property
            def capabilities(self) -> List[AgentCapability]:
                return [AgentCapability.CODE_ANALYZE, AgentCapability.CODE_GENERATE]

            async def _initialize(self):
                # Load any resources
                pass

            async def handle_task(self, task: Dict[str, Any]) -> TaskResult:
                # Process the task
                return TaskResult(
                    task_id=task.get("id", "unknown"),
                    status=TaskStatus.COMPLETED,
                    result={"output": "done"}
                )
    """

    def __init__(
        self,
        config: AgentConfig = None,
        registry: "AgentRegistry" = None,
        broker: "MessageBroker" = None,
        **kwargs
    ):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration (uses defaults if None)
            registry: Optional agent registry for discovery
            broker: Optional message broker for communication
            **kwargs: Additional config overrides
        """
        self.config = config or AgentConfig(**kwargs)
        self._state = AgentState.STARTING
        self._llm: Optional["LLMBackend"] = None
        self._registry = registry
        self._broker = broker
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_handler_task: Optional[asyncio.Task] = None
        self._running = False

        # Set up logging
        self.logger = logging.getLogger(f"ewr.agent.{self.config.name}")

        # Create agent info
        self._info = AgentInfo(
            agent_type=self.agent_type,
            name=self.config.name,
            version=self.config.version,
            description=self.config.description,
            capabilities=self.capabilities,
            state=AgentState.STARTING,
        )

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Agent type identifier."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent provides."""
        pass

    @property
    def agent_id(self) -> str:
        """Get the unique agent ID."""
        return self._info.agent_id

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @property
    def info(self) -> AgentInfo:
        """Get agent info."""
        return self._info

    @property
    def llm(self) -> "LLMBackend":
        """Get or initialize LLM backend."""
        if self._llm is None:
            from .llm_backends import get_backend
            self._llm = get_backend(
                self.config.llm_backend,
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
            )
        return self._llm

    def set_llm(self, llm: "LLMBackend") -> None:
        """Set a custom LLM backend."""
        self._llm = llm

    def set_registry(self, registry: "AgentRegistry") -> None:
        """Set the agent registry."""
        self._registry = registry

    def set_broker(self, broker: "MessageBroker") -> None:
        """Set the message broker."""
        self._broker = broker

    async def start(self) -> None:
        """
        Start the agent.

        This will:
        1. Run agent-specific initialization
        2. Register with the registry (if enabled)
        3. Create message queue (if broker enabled)
        4. Start heartbeat loop (if broker enabled)
        5. Start message handler loop (if broker enabled)
        """
        self.logger.info(f"Starting agent: {self.config.name}")
        self._state = AgentState.STARTING
        self._info.state = AgentState.STARTING
        self._running = True

        try:
            # Run subclass initialization
            await self._initialize()

            # Register with registry
            if self.config.registry_enabled and self._registry:
                await self._registry.register(self._info)
                self.logger.info(f"Registered with registry: {self.agent_id}")

            # Set up message broker
            if self.config.broker_enabled and self._broker:
                await self._broker.create_queue(self.agent_id)

                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(),
                    name=f"{self.config.name}-heartbeat"
                )

                # Start message handler
                self._message_handler_task = asyncio.create_task(
                    self._message_loop(),
                    name=f"{self.config.name}-messages"
                )

            self._state = AgentState.IDLE
            self._info.state = AgentState.IDLE
            self.logger.info(f"Agent started: {self.config.name}")

        except Exception as e:
            self._state = AgentState.ERROR
            self._info.state = AgentState.ERROR
            self.logger.error(f"Failed to start agent: {e}")
            raise

    async def stop(self) -> None:
        """
        Stop the agent.

        This will:
        1. Stop heartbeat loop
        2. Stop message handler
        3. Unregister from registry
        4. Delete message queue
        """
        self.logger.info(f"Stopping agent: {self.config.name}")
        self._state = AgentState.STOPPING
        self._info.state = AgentState.STOPPING
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel message handler
        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass

        # Unregister
        if self._registry:
            await self._registry.unregister(self.agent_id)

        # Delete queue
        if self._broker:
            await self._broker.delete_queue(self.agent_id)

        self._state = AgentState.OFFLINE
        self._info.state = AgentState.OFFLINE
        self.logger.info(f"Agent stopped: {self.config.name}")

    async def request_help(
        self,
        capability: str,
        task_description: str,
        task_params: Dict[str, Any] = None,
        target_agent_type: AgentType = None,
        target_agent_id: str = None,
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> DelegationResponse:
        """
        Request help from another agent.

        Args:
            capability: The capability needed
            task_description: Description of what needs to be done
            task_params: Parameters for the task
            target_agent_type: Optionally target a specific agent type
            target_agent_id: Optionally target a specific agent
            timeout: How long to wait for response
            priority: Message priority

        Returns:
            DelegationResponse from the helping agent

        Raises:
            RuntimeError: If no registry/broker available or no capable agent found
        """
        if not self._registry:
            raise RuntimeError("No registry available for agent discovery")
        if not self._broker:
            raise RuntimeError("No message broker available for communication")

        # Find a capable agent
        if target_agent_id:
            target = await self._registry.get(target_agent_id)
            if not target:
                raise RuntimeError(f"Target agent not found: {target_agent_id}")
        else:
            # Find by capability
            candidates = await self._registry.find_by_capability(
                capability,
                state=AgentState.IDLE
            )
            if target_agent_type:
                candidates = [
                    c for c in candidates
                    if c.agent_type == target_agent_type
                ]
            if not candidates:
                raise RuntimeError(f"No agent found with capability: {capability}")
            target = candidates[0]

        # Create delegation request
        request = DelegationRequest(
            sender_agent_id=self.agent_id,
            target_agent_type=target.agent_type,
            target_agent_id=target.agent_id,
            required_capability=capability,
            task_description=task_description,
            task_params=task_params or {},
            priority=priority,
            timeout_seconds=int(timeout),
        )

        # Create message
        message = AgentMessage(
            message_type=MessageType.REQUEST,
            sender_id=self.agent_id,
            recipient_id=target.agent_id,
            priority=priority,
            payload=request.model_dump(),
        )

        # Send and wait for response
        response_msg = await self._broker.request(message, timeout=timeout)

        if response_msg is None:
            return DelegationResponse(
                request_id=request.request_id,
                responder_agent_id=target.agent_id,
                accepted=False,
                error="Request timed out"
            )

        return DelegationResponse(**response_msg.payload)

    async def send_notification(
        self,
        recipient_id: str,
        data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> bool:
        """
        Send a one-way notification to another agent.

        Args:
            recipient_id: Target agent ID
            data: Notification payload
            priority: Message priority

        Returns:
            True if sent successfully
        """
        if not self._broker:
            return False

        message = AgentMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            priority=priority,
            payload=data,
        )

        return await self._broker.send(message)

    async def broadcast(
        self,
        data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> bool:
        """
        Broadcast a message to all agents.

        Args:
            data: Message payload
            priority: Message priority

        Returns:
            True if sent successfully
        """
        if not self._broker:
            return False

        message = AgentMessage(
            message_type=MessageType.BROADCAST,
            sender_id=self.agent_id,
            priority=priority,
            payload=data,
        )

        return await self._broker.broadcast(message)

    @abstractmethod
    async def _initialize(self) -> None:
        """
        Initialize agent-specific resources.

        Override this to set up databases, load models, etc.
        """
        pass

    @abstractmethod
    async def handle_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Handle an incoming task.

        Args:
            task: Task parameters including:
                - task_id: Unique task identifier
                - task_type: Type of task
                - params: Task-specific parameters

        Returns:
            TaskResult with execution outcome
        """
        pass

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Handle an incoming message.

        Override to customize message handling.
        Default implementation handles REQUEST messages by calling handle_task.

        Args:
            message: Incoming message

        Returns:
            Response message (for REQUEST type) or None
        """
        if message.message_type == MessageType.REQUEST:
            # Extract delegation request
            request = DelegationRequest(**message.payload)

            self._state = AgentState.BUSY
            self._info.state = AgentState.BUSY
            self._info.current_task_id = request.request_id

            try:
                # Handle the task
                result = await self.handle_task({
                    "task_id": request.request_id,
                    "task_type": request.required_capability,
                    "description": request.task_description,
                    "params": request.task_params,
                    "context": request.context,
                })

                # Create response
                response = DelegationResponse(
                    request_id=request.request_id,
                    responder_agent_id=self.agent_id,
                    accepted=True,
                    result=result.model_dump() if result else None,
                )

                self._info.tasks_completed += 1

            except Exception as e:
                response = DelegationResponse(
                    request_id=request.request_id,
                    responder_agent_id=self.agent_id,
                    accepted=False,
                    error=str(e),
                )
                self._info.tasks_failed += 1
                self.logger.error(f"Task failed: {e}")

            finally:
                self._state = AgentState.IDLE
                self._info.state = AgentState.IDLE
                self._info.current_task_id = None

            # Return response message
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                payload=response.model_dump(),
            )

        elif message.message_type == MessageType.NOTIFICATION:
            # Handle notification (no response needed)
            await self._handle_notification(message.payload)
            return None

        return None

    async def _handle_notification(self, payload: Dict[str, Any]) -> None:
        """
        Handle a notification message.

        Override to handle specific notification types.

        Args:
            payload: Notification data
        """
        self.logger.debug(f"Received notification: {payload}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to registry."""
        while self._running:
            try:
                if self._registry:
                    await self._registry.heartbeat(self.agent_id)
                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Heartbeat failed: {e}")

    async def _message_loop(self) -> None:
        """Process incoming messages."""
        while self._running:
            try:
                if self._broker:
                    message = await self._broker.receive(
                        self.agent_id,
                        timeout=1.0  # Short timeout for responsiveness
                    )
                    if message:
                        response = await self.handle_message(message)
                        if response:
                            await self._broker.send(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Message handling error: {e}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name!r}, "
            f"type={self.agent_type.value!r}, "
            f"state={self._state.value!r})"
        )
