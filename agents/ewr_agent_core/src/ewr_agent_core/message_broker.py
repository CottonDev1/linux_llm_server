"""
EWR Agent Message Broker
========================

In-memory message queue for inter-agent communication.
Supports:
- Priority-based message queuing
- Request-response correlation
- Broadcast messaging
- Message expiration (TTL)
"""

import asyncio
from typing import Dict, Optional, List
from datetime import datetime
import logging

from .models import AgentMessage, MessageType, MessagePriority


# Global singleton broker
_global_broker: Optional["MessageBroker"] = None


class PriorityMessage:
    """Wrapper for priority queue ordering."""

    def __init__(self, priority: int, timestamp: float, message: AgentMessage):
        self.priority = priority
        self.timestamp = timestamp
        self.message = message

    def __lt__(self, other: "PriorityMessage") -> bool:
        # Higher priority number = higher priority (processed first)
        # Use negative priority so heapq gives us highest first
        if self.priority != other.priority:
            return self.priority > other.priority
        # Same priority: earlier timestamp first
        return self.timestamp < other.timestamp


class MessageBroker:
    """
    In-memory message broker for agent communication.

    Provides:
    - Per-agent message queues
    - Priority-based message ordering
    - Request-response correlation
    - Broadcast to all agents
    - Message TTL expiration

    Example:
        broker = MessageBroker()

        # Create queue for an agent
        await broker.create_queue("agent-123")

        # Send a message
        message = AgentMessage(
            message_type=MessageType.REQUEST,
            sender_id="agent-456",
            recipient_id="agent-123",
            payload={"task": "analyze code"}
        )
        await broker.send(message)

        # Receive messages
        msg = await broker.receive("agent-123", timeout=5.0)
    """

    def __init__(self, default_ttl: int = 300):
        """
        Initialize the message broker.

        Args:
            default_ttl: Default time-to-live for messages in seconds
        """
        self._queues: Dict[str, asyncio.PriorityQueue] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl
        self.logger = logging.getLogger("ewr.agent.broker")

    async def create_queue(self, agent_id: str) -> bool:
        """
        Create a message queue for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            True if queue created (False if already exists)
        """
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.PriorityQueue()
                self.logger.debug(f"Created queue for agent: {agent_id}")
                return True
            return False

    async def delete_queue(self, agent_id: str) -> bool:
        """
        Delete an agent's message queue.

        Args:
            agent_id: ID of the agent

        Returns:
            True if queue was deleted
        """
        async with self._lock:
            if agent_id in self._queues:
                # Cancel any pending futures for this agent
                to_cancel = [
                    (mid, fut) for mid, fut in self._pending_responses.items()
                    if not fut.done()
                ]
                for mid, fut in to_cancel:
                    fut.cancel()

                del self._queues[agent_id]
                self.logger.debug(f"Deleted queue for agent: {agent_id}")
                return True
            return False

    async def send(self, message: AgentMessage) -> bool:
        """
        Send a message to an agent.

        If recipient_id is None, message is treated as broadcast.

        Args:
            message: Message to send

        Returns:
            True if message was queued
        """
        if message.recipient_id is None:
            return await self.broadcast(message)

        if message.recipient_id not in self._queues:
            self.logger.warning(f"No queue for recipient: {message.recipient_id}")
            return False

        # Create priority message (higher number = higher priority)
        priority_msg = PriorityMessage(
            priority=message.priority.value,
            timestamp=message.timestamp.timestamp(),
            message=message
        )

        await self._queues[message.recipient_id].put(priority_msg)
        self.logger.debug(
            f"Queued message {message.message_id[:8]} for {message.recipient_id[:8]}"
        )
        return True

    async def broadcast(self, message: AgentMessage) -> bool:
        """
        Broadcast message to all agents.

        Args:
            message: Message to broadcast

        Returns:
            True if broadcast to at least one agent
        """
        message.message_type = MessageType.BROADCAST
        count = 0

        async with self._lock:
            for agent_id, queue in self._queues.items():
                if agent_id != message.sender_id:  # Don't send to self
                    priority_msg = PriorityMessage(
                        priority=message.priority.value,
                        timestamp=message.timestamp.timestamp(),
                        message=message
                    )
                    await queue.put(priority_msg)
                    count += 1

        self.logger.debug(f"Broadcast message to {count} agents")
        return count > 0

    async def receive(
        self,
        agent_id: str,
        timeout: float = None,
        filter_expired: bool = True,
    ) -> Optional[AgentMessage]:
        """
        Receive next message for an agent.

        Args:
            agent_id: ID of the agent
            timeout: How long to wait (None = wait forever, 0 = don't wait)
            filter_expired: Skip expired messages

        Returns:
            AgentMessage or None if timeout/no messages
        """
        if agent_id not in self._queues:
            return None

        try:
            if timeout == 0:
                # Non-blocking
                try:
                    priority_msg = self._queues[agent_id].get_nowait()
                except asyncio.QueueEmpty:
                    return None
            elif timeout:
                priority_msg = await asyncio.wait_for(
                    self._queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                priority_msg = await self._queues[agent_id].get()

            message = priority_msg.message

            # Check expiration
            if filter_expired and message.is_expired():
                self.logger.debug(f"Skipped expired message: {message.message_id[:8]}")
                # Try to get next message (recursive, but with limit)
                return await self.receive(agent_id, timeout=0, filter_expired=True)

            # Handle response correlation
            if message.message_type == MessageType.RESPONSE:
                correlation_id = message.correlation_id
                if correlation_id and correlation_id in self._pending_responses:
                    future = self._pending_responses.pop(correlation_id)
                    if not future.done():
                        future.set_result(message)

            return message

        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None

    async def receive_all(
        self,
        agent_id: str,
        filter_expired: bool = True,
    ) -> List[AgentMessage]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_id: ID of the agent
            filter_expired: Skip expired messages

        Returns:
            List of messages (may be empty)
        """
        messages = []
        while True:
            msg = await self.receive(agent_id, timeout=0, filter_expired=filter_expired)
            if msg is None:
                break
            messages.append(msg)
        return messages

    async def request(
        self,
        message: AgentMessage,
        timeout: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response.

        Sets up correlation tracking and waits for a response
        message with matching correlation_id.

        Args:
            message: Request message to send
            timeout: How long to wait for response

        Returns:
            Response message or None if timeout
        """
        message.message_type = MessageType.REQUEST

        # Create future for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_responses[message.message_id] = future

        try:
            # Send the request
            sent = await self.send(message)
            if not sent:
                return None

            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self.logger.warning(f"Request timed out: {message.message_id[:8]}")
            return None

        finally:
            # Clean up pending response
            self._pending_responses.pop(message.message_id, None)

    async def respond(
        self,
        original_message: AgentMessage,
        response_payload: dict,
        sender_id: str,
    ) -> bool:
        """
        Send a response to a request message.

        Args:
            original_message: The request message being responded to
            response_payload: Response data
            sender_id: ID of the responding agent

        Returns:
            True if response was queued
        """
        response = AgentMessage(
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            correlation_id=original_message.message_id,
            payload=response_payload,
        )
        return await self.send(response)

    async def get_queue_size(self, agent_id: str) -> int:
        """
        Get number of messages in an agent's queue.

        Args:
            agent_id: ID of the agent

        Returns:
            Number of queued messages
        """
        if agent_id not in self._queues:
            return 0
        return self._queues[agent_id].qsize()

    async def get_stats(self) -> Dict:
        """
        Get broker statistics.

        Returns:
            Dict with queue counts and pending requests
        """
        async with self._lock:
            queue_sizes = {
                agent_id: queue.qsize()
                for agent_id, queue in self._queues.items()
            }
            return {
                "queues": len(self._queues),
                "pending_responses": len(self._pending_responses),
                "queue_sizes": queue_sizes,
                "total_messages": sum(queue_sizes.values()),
            }

    async def clear_queue(self, agent_id: str) -> int:
        """
        Clear all messages from an agent's queue.

        Args:
            agent_id: ID of the agent

        Returns:
            Number of messages cleared
        """
        if agent_id not in self._queues:
            return 0

        count = 0
        while True:
            try:
                self._queues[agent_id].get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break

        return count


def get_broker(create_if_missing: bool = True) -> MessageBroker:
    """
    Get the global broker singleton.

    Args:
        create_if_missing: Create broker if it doesn't exist

    Returns:
        MessageBroker instance
    """
    global _global_broker

    if _global_broker is None and create_if_missing:
        _global_broker = MessageBroker()

    return _global_broker


def set_broker(broker: MessageBroker) -> None:
    """
    Set the global broker singleton.

    Args:
        broker: Broker instance to use globally
    """
    global _global_broker
    _global_broker = broker
