"""
Call Chain Builder Service
==========================

Builds call chains and call trees from method call relationships stored
in MongoDB.

Design Rationale:
-----------------
Call chains trace the execution path from an entry point (typically a UI
event handler) through business logic to data layer operations. This is
essential for understanding:

1. How user actions trigger code execution
2. Which methods are involved in a business process
3. Where database operations occur in the flow
4. Dependencies between components

Graph Traversal Strategy:
-------------------------
The call chain builder uses a depth-first search (DFS) with cycle detection.
This approach:
- Handles recursive method calls without infinite loops
- Respects a configurable maximum depth
- Identifies terminal nodes (database operations)
- Builds both linear chains and tree structures

Best Practices for Call Graph Analysis:
---------------------------------------
1. Limit depth to prevent explosion in large codebases
2. Track visited nodes to detect cycles
3. Prioritize paths that reach database operations
4. Consider edge weights (call frequency) for ranking

Implementation Notes:
--------------------
The call graph is stored in MongoDB with edges represented as documents
containing caller/callee relationships. The builder reconstructs the
graph structure by querying these edges and following them recursively.
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

from code_flow_pipeline.models.query_models import (
    CallChain,
    CallChainNode,
    CallTree,
    FormattedResult,
)

logger = logging.getLogger(__name__)


@dataclass
class TraversalState:
    """State maintained during graph traversal."""
    visited: Set[str] = field(default_factory=set)
    current_chain: List[CallChainNode] = field(default_factory=list)
    all_chains: List[CallChain] = field(default_factory=list)
    depth: int = 0
    max_depth: int = 10


class CallChainBuilder:
    """
    Builds call chains from method call relationships.

    This service traverses the call graph stored in MongoDB to build
    execution paths from entry points to terminal operations.

    Usage:
        builder = CallChainBuilder(mongodb_service)
        chains = await builder.build_chains(
            start_method="btnCommit_Click",
            project="gin",
            max_depth=10
        )

    Architecture:
        - Async DFS traversal with cycle detection
        - Caches method lookups to avoid redundant queries
        - Supports both chain (linear) and tree (hierarchical) output
        - Identifies terminal nodes (database accessors)
    """

    # Maximum fan-out per method to prevent explosion
    MAX_CHILDREN_PER_NODE = 5

    def __init__(self, mongodb_service=None):
        """
        Initialize the call chain builder.

        Args:
            mongodb_service: MongoDBService instance (lazy-loaded if None)
        """
        self._mongodb_service = mongodb_service
        self._method_cache: Dict[str, Dict[str, Any]] = {}

    async def _get_mongodb_service(self):
        """Lazy-load MongoDB service."""
        if self._mongodb_service is None:
            from mongodb import MongoDBService
            self._mongodb_service = MongoDBService.get_instance()
            if not self._mongodb_service.is_initialized:
                await self._mongodb_service.initialize()
        return self._mongodb_service

    async def build_chains(
        self,
        start_method: str,
        project: Optional[str] = None,
        target_method: Optional[str] = None,
        max_depth: int = 10,
    ) -> List[CallChain]:
        """
        Build call chains starting from a method.

        Args:
            start_method: Starting method name (may include class prefix)
            project: Project scope
            target_method: Optional target to find paths to
            max_depth: Maximum chain depth

        Returns:
            List of CallChain objects representing execution paths

        Algorithm:
        1. Initialize traversal state
        2. Perform DFS from start_method
        3. At each node:
           a. Check for cycles (skip if visited)
           b. Look up method metadata (calls, database tables)
           c. If terminal (has database tables or no outgoing calls), record chain
           d. Otherwise, recurse to called methods
        4. Return all discovered chains
        """
        mongodb = await self._get_mongodb_service()

        state = TraversalState(max_depth=max_depth)

        await self._traverse_dfs(
            mongodb=mongodb,
            method_name=start_method,
            project=project,
            target_method=target_method,
            state=state,
        )

        logger.info(f"Built {len(state.all_chains)} call chains from '{start_method}'")
        return state.all_chains

    async def _traverse_dfs(
        self,
        mongodb,
        method_name: str,
        project: Optional[str],
        target_method: Optional[str],
        state: TraversalState,
    ):
        """
        Perform DFS traversal of the call graph.

        Args:
            mongodb: MongoDB service instance
            method_name: Current method to process
            project: Project scope
            target_method: Optional target method
            state: Traversal state
        """
        # Check depth limit
        if state.depth > state.max_depth:
            return

        # Check for cycles
        if method_name in state.visited:
            return

        # Mark as visited
        state.visited.add(method_name)

        # Create node for current method
        node = CallChainNode(
            method_name=method_name,
            depth=state.depth,
        )

        # Add to current chain
        state.current_chain.append(node)

        # Check if we reached the target
        if target_method and method_name.lower().find(target_method.lower()) >= 0:
            self._record_chain(state)
            state.current_chain.pop()
            state.visited.discard(method_name)
            return

        # Look up method metadata
        method_info = await self._get_method_info(mongodb, method_name, project)

        if method_info:
            # Check for database tables (terminal indicator)
            database_tables = self._try_parse_json(method_info.get("databaseTables"))
            if database_tables:
                node.database_tables = database_tables
                self._record_chain(state)
                state.current_chain.pop()
                state.visited.discard(method_name)
                return

            # Get called methods
            called_methods = self._try_parse_json(method_info.get("callsMethod"))

            if called_methods:
                # Limit fan-out
                for called_method in called_methods[:self.MAX_CHILDREN_PER_NODE]:
                    state.depth += 1
                    await self._traverse_dfs(
                        mongodb=mongodb,
                        method_name=called_method,
                        project=project,
                        target_method=target_method,
                        state=state,
                    )
                    state.depth -= 1
            else:
                # Leaf node - record chain
                self._record_chain(state)
        else:
            # Method not found - record chain as-is
            self._record_chain(state)

        # Backtrack
        state.current_chain.pop()
        state.visited.discard(method_name)

    def _record_chain(self, state: TraversalState):
        """Record the current chain as a complete execution path."""
        if not state.current_chain:
            return

        chain = CallChain(
            start_method=state.current_chain[0].method,
            end_method=state.current_chain[-1].method,
            steps=[CallChainNode(**node.model_dump()) for node in state.current_chain],
            depth=len(state.current_chain),
        )
        chain.compute_touches_database()
        state.all_chains.append(chain)

    async def _get_method_info(
        self,
        mongodb,
        method_name: str,
        project: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Get method metadata from cache or MongoDB.

        Args:
            mongodb: MongoDB service instance
            method_name: Method name to look up
            project: Project scope

        Returns:
            Method metadata dict or None if not found
        """
        cache_key = f"{project or 'all'}:{method_name}"

        if cache_key in self._method_cache:
            return self._method_cache[cache_key]

        try:
            results = await mongodb.search_vectors(
                query=f"method {method_name}",
                project=project,
                category="code",
                doc_type="method",
                limit=1,
            )

            if results:
                metadata = results[0].get("metadata", {})
                self._method_cache[cache_key] = metadata
                return metadata

        except Exception as e:
            logger.warning(f"Failed to lookup method '{method_name}': {e}")

        return None

    async def build_call_tree(
        self,
        entry_point: str,
        call_graph_results: List[FormattedResult],
        max_depth: int = 10,
    ) -> CallTree:
        """
        Build a hierarchical call tree from call graph results.

        This method builds a tree structure from pre-fetched call graph
        documents, which is more efficient than individual queries.

        Args:
            entry_point: Root method name
            call_graph_results: Pre-fetched call relationship documents
            max_depth: Maximum tree depth

        Returns:
            CallTree with hierarchical method structure
        """
        # Index call graph by caller method
        calls_by_caller: Dict[str, List[Dict[str, Any]]] = {}
        for result in call_graph_results:
            metadata = result.metadata
            caller = metadata.get("callerMethod", "")
            if caller:
                if caller not in calls_by_caller:
                    calls_by_caller[caller] = []
                calls_by_caller[caller].append(metadata)

        # Build tree recursively
        visited: Set[str] = set()
        root = self._build_tree_node(
            method=entry_point,
            calls_by_caller=calls_by_caller,
            visited=visited,
            depth=0,
            max_depth=max_depth,
        )

        return root

    def _build_tree_node(
        self,
        method: str,
        calls_by_caller: Dict[str, List[Dict[str, Any]]],
        visited: Set[str],
        depth: int,
        max_depth: int,
        caller: Optional[str] = None,
    ) -> CallTree:
        """
        Recursively build a call tree node.

        Args:
            method: Current method name
            calls_by_caller: Index of calls by caller method
            visited: Set of visited methods (cycle detection)
            depth: Current depth
            max_depth: Maximum depth
            caller: Parent method name

        Returns:
            CallTree node with children
        """
        node = CallTree(
            method=method,
            depth=depth,
            caller=caller,
            children=[],
        )

        if depth >= max_depth or method in visited:
            return node

        visited.add(method)

        # Find outgoing calls
        outgoing = calls_by_caller.get(method, [])

        for call in outgoing[:self.MAX_CHILDREN_PER_NODE]:
            callee = call.get("calleeMethod")
            if callee:
                child = self._build_tree_node(
                    method=callee,
                    calls_by_caller=calls_by_caller,
                    visited=set(visited),  # Copy for each branch
                    depth=depth + 1,
                    max_depth=max_depth,
                    caller=method,
                )
                child.call_type = call.get("callType", "direct")
                node.children.append(child)

        node.compute_total_methods()
        return node

    async def build_chains_from_methods(
        self,
        methods: List[FormattedResult],
        ui_events: List[FormattedResult],
        project: Optional[str],
        max_hops: int = 5,
    ) -> List[CallChain]:
        """
        Build call chains from retrieved methods and UI events.

        This is a convenience method that:
        1. Identifies starting points from UI events or top methods
        2. Builds chains from each starting point
        3. Returns aggregated chains

        Args:
            methods: Retrieved method results
            ui_events: Retrieved UI event handler results
            project: Project scope
            max_hops: Maximum chain depth

        Returns:
            List of all discovered call chains
        """
        # Determine starting points
        starting_points: List[str] = []

        # Prefer UI events as entry points
        if ui_events:
            for event in ui_events:
                handler = event.metadata.get("handlerMethod")
                if handler:
                    starting_points.append(handler)

        # Fall back to top methods if no UI events
        if not starting_points and methods:
            for method in methods[:3]:  # Top 3 methods
                method_name = method.metadata.get("methodName")
                if method_name:
                    starting_points.append(method_name)

        # Build chains from each starting point
        all_chains: List[CallChain] = []

        for start_method in starting_points:
            try:
                chains = await self.build_chains(
                    start_method=start_method,
                    project=project,
                    max_depth=max_hops,
                )
                all_chains.extend(chains)
            except Exception as e:
                logger.warning(f"Failed to build chain from '{start_method}': {e}")

        logger.info(
            f"Built {len(all_chains)} total chains from "
            f"{len(starting_points)} starting points"
        )

        return all_chains

    def clear_cache(self):
        """Clear the method metadata cache."""
        self._method_cache.clear()
        logger.debug("Method cache cleared")

    def _try_parse_json(self, value: Any) -> List[str]:
        """Safely parse a JSON string to a list."""
        if not value:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return []


# Singleton instance
_builder_instance: CallChainBuilder | None = None


async def get_call_chain_builder() -> CallChainBuilder:
    """Get or create the singleton CallChainBuilder instance."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = CallChainBuilder()
    return _builder_instance
