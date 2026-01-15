"""
Code Retriever Service
======================

Retrieves code entities (methods, classes, event handlers) from MongoDB
using semantic search via the RoslynMongoDBService.

Design Rationale:
-----------------
This service acts as a facade over the RoslynMongoDBService, providing
a specialized interface for the code assistance pipeline. It adds:

1. Query Classification: Determines which entity types to search based
   on query keywords (e.g., "click" suggests event handlers)

2. Multi-Source Retrieval: Orchestrates parallel searches across multiple
   collections for comprehensive context

3. Result Normalization: Converts MongoDB documents to typed SourceInfo
   models for consistent downstream processing

Architecture:
- Lazy initialization pattern for RoslynMongoDBService
- Async-first design for non-blocking I/O
- Parallel search execution using asyncio.gather
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from code_assistance_pipeline.models.query_models import SourceInfo, SourceType

logger = logging.getLogger(__name__)


class CodeRetriever:
    """
    Retrieves code context from MongoDB for code assistance queries.

    This service orchestrates multi-source retrieval:
    - Methods: Semantic search for relevant method implementations
    - Classes: Class definitions with inheritance information
    - Event Handlers: UI-to-code mappings for interactive elements
    - Call Chains: Caller/callee relationships for code flow analysis

    Usage:
        retriever = CodeRetriever()
        await retriever.initialize()

        methods = await retriever.search_methods("save bale data", project="Gin")
        classes = await retriever.search_classes("view model for orders")
    """

    # Keywords that suggest event handler search
    EVENT_KEYWORDS = frozenset([
        "click", "button", "event", "handler", "pressed",
        "selected", "changed", "focus", "mouse", "key"
    ])

    # Keywords that suggest class-level search
    CLASS_KEYWORDS = frozenset([
        "class", "viewmodel", "view model", "service", "repository",
        "controller", "interface", "base class", "inheritance"
    ])

    # Keywords that suggest database/SQL operations
    SQL_KEYWORDS = frozenset([
        "sql", "database", "stored procedure", "query", "select",
        "insert", "update", "delete", "execute", "command"
    ])

    def __init__(self):
        """Initialize the code retriever with lazy service loading."""
        self._roslyn_service = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the underlying Roslyn MongoDB service.

        Uses lazy loading to avoid connection overhead until first use.
        """
        if self._initialized:
            return

        from roslyn_mongodb_service import get_roslyn_mongodb_service

        self._roslyn_service = get_roslyn_mongodb_service()
        await self._roslyn_service.initialize()
        self._initialized = True
        logger.info("CodeRetriever initialized")

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized before use."""
        if not self._initialized:
            await self.initialize()

    def classify_query(self, query: str) -> Dict[str, bool]:
        """
        Classify a query to determine which entity types to search.

        Analyzes the query text for keywords that suggest specific
        search strategies. This enables targeted retrieval that
        prioritizes relevant entity types.

        Args:
            query: Natural language query

        Returns:
            Dict with boolean flags for each search type:
            - search_methods: Always True (methods are always searched)
            - search_classes: True if class-related keywords found
            - search_events: True if event-related keywords found
            - search_sql_only: True if SQL-specific keywords found

        Example:
            >>> classifier = CodeRetriever()
            >>> classifier.classify_query("what happens when save button is clicked")
            {'search_methods': True, 'search_classes': False, 'search_events': True, 'search_sql_only': False}
        """
        query_lower = query.lower()
        words = set(re.findall(r'\w+', query_lower))

        return {
            "search_methods": True,  # Always search methods
            "search_classes": bool(words & self.CLASS_KEYWORDS),
            "search_events": bool(words & self.EVENT_KEYWORDS),
            "search_sql_only": bool(words & self.SQL_KEYWORDS),
        }

    async def search_methods(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 10,
        sql_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for methods using semantic similarity.

        Args:
            query: Natural language search query
            project: Optional project name filter
            limit: Maximum number of results
            sql_only: Only return methods with SQL operations

        Returns:
            List of method dictionaries with metadata

        Design Note:
        Methods are the primary search target because they contain
        the implementation logic that typically answers "how" questions.
        """
        await self._ensure_initialized()

        results = await self._roslyn_service.search_methods(
            query=query,
            project=project,
            limit=limit,
            include_sql_only=sql_only
        )

        logger.debug(f"Found {len(results)} methods for query: {query[:50]}...")
        return results

    async def search_classes(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for classes using semantic similarity.

        Args:
            query: Natural language search query
            project: Optional project name filter
            limit: Maximum number of results

        Returns:
            List of class dictionaries with metadata

        Design Note:
        Classes provide structural context - inheritance relationships,
        interface implementations, and member overviews that help
        understand the codebase organization.
        """
        await self._ensure_initialized()

        results = await self._roslyn_service.search_classes(
            query=query,
            project=project,
            limit=limit
        )

        logger.debug(f"Found {len(results)} classes for query: {query[:50]}...")
        return results

    async def search_event_handlers(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for event handlers (UI-to-code mappings).

        Args:
            query: Natural language search query
            project: Optional project name filter
            limit: Maximum number of results

        Returns:
            List of event handler dictionaries

        Design Note:
        Event handlers are critical for understanding UI behavior.
        They bridge the gap between visual elements (buttons, forms)
        and the code that responds to user actions.
        """
        await self._ensure_initialized()

        results = await self._roslyn_service.search_event_handlers(
            query=query,
            project=project,
            limit=limit
        )

        logger.debug(f"Found {len(results)} event handlers for query: {query[:50]}...")
        return results

    async def get_call_chain(
        self,
        method_name: str,
        class_name: str,
        project: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get the call chain for a specific method.

        Args:
            method_name: Name of the method to trace
            class_name: Name of the containing class
            project: Optional project name filter
            direction: "callers", "callees", or "both"
            max_depth: Maximum depth for traversal

        Returns:
            Dict with 'callers' and/or 'callees' lists

        Design Note:
        Call chains are essential for understanding code flow.
        They answer questions like "what calls this method?" and
        "what does this method do?" by tracing the execution path.
        """
        await self._ensure_initialized()

        result = await self._roslyn_service.get_call_chain(
            method_name=method_name,
            class_name=class_name,
            project=project,
            direction=direction,
            max_depth=max_depth
        )

        callers = len(result.get("callers", []))
        callees = len(result.get("callees", []))
        logger.debug(f"Call chain for {class_name}.{method_name}: {callers} callers, {callees} callees")

        return result

    async def retrieve_comprehensive(
        self,
        query: str,
        project: Optional[str] = None,
        method_limit: int = 10,
        class_limit: int = 5,
        event_limit: int = 5,
        include_call_chains: bool = True,
        max_depth: int = 2
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[str]]:
        """
        Perform comprehensive retrieval across all entity types.

        This method orchestrates parallel searches and call chain
        analysis to gather maximum relevant context for the query.

        Args:
            query: Natural language query
            project: Optional project filter
            method_limit: Max methods to retrieve
            class_limit: Max classes to retrieve
            event_limit: Max event handlers to retrieve
            include_call_chains: Whether to trace call chains
            max_depth: Call chain traversal depth

        Returns:
            Tuple of (methods, classes, event_handlers, call_chain)

        Design Note:
        Parallel execution significantly reduces latency when querying
        multiple collections. The call chain is only fetched for the
        top method result to avoid excessive API calls.
        """
        await self._ensure_initialized()

        # Classify query to determine search strategy
        classification = self.classify_query(query)

        # Build list of search tasks
        tasks = []

        # Always search methods
        tasks.append(("methods", self.search_methods(
            query=query,
            project=project,
            limit=method_limit,
            sql_only=classification["search_sql_only"]
        )))

        # Search classes if relevant
        if classification["search_classes"] or True:  # Always include classes for context
            tasks.append(("classes", self.search_classes(
                query=query,
                project=project,
                limit=class_limit
            )))

        # Search event handlers if relevant
        if classification["search_events"]:
            tasks.append(("events", self.search_event_handlers(
                query=query,
                project=project,
                limit=event_limit
            )))

        # Execute searches in parallel
        results = {}
        task_coros = [task[1] for task in tasks]
        task_names = [task[0] for task in tasks]

        gathered = await asyncio.gather(*task_coros, return_exceptions=True)

        for name, result in zip(task_names, gathered):
            if isinstance(result, Exception):
                logger.error(f"Search failed for {name}: {result}")
                results[name] = []
            else:
                results[name] = result

        methods = results.get("methods", [])
        classes = results.get("classes", [])
        event_handlers = results.get("events", [])

        # Get call chain for top method if requested
        call_chain = []
        if include_call_chains and methods:
            top_method = methods[0]
            method_name = top_method.get("method_name", "")
            class_name = top_method.get("class_name", "")

            if method_name and class_name:
                try:
                    chain_result = await self.get_call_chain(
                        method_name=method_name,
                        class_name=class_name,
                        project=project,
                        direction="both",
                        max_depth=max_depth
                    )

                    # Build call chain string
                    callers = [f"{c['class']}.{c['method']}" for c in chain_result.get("callers", [])[:3]]
                    callees = [f"{c['class']}.{c['method']}" for c in chain_result.get("callees", [])[:3]]
                    center = f"{class_name}.{method_name}"

                    call_chain = callers + [center] + callees
                except Exception as e:
                    logger.warning(f"Failed to get call chain: {e}")

        logger.info(
            f"Comprehensive retrieval: {len(methods)} methods, "
            f"{len(classes)} classes, {len(event_handlers)} events, "
            f"{len(call_chain)} call chain items"
        )

        return methods, classes, event_handlers, call_chain

    def to_source_info(self, entity: Dict[str, Any], entity_type: SourceType) -> SourceInfo:
        """
        Convert a raw entity dictionary to a SourceInfo model.

        Args:
            entity: Raw entity dictionary from MongoDB
            entity_type: Type of the entity

        Returns:
            SourceInfo model with normalized data

        Design Note:
        This normalization ensures consistent data structure for
        downstream processing regardless of the source collection.
        """
        if entity_type == SourceType.METHOD:
            return SourceInfo(
                type=SourceType.METHOD,
                name=f"{entity.get('class_name', '')}.{entity.get('method_name', '')}",
                file=entity.get("file_path", ""),
                line=entity.get("line_number", 0),
                similarity=entity.get("similarity", 0.0),
                snippet=entity.get("signature", "")
            )
        elif entity_type == SourceType.CLASS:
            base = entity.get("base_class", "")
            snippet = f"class {entity.get('class_name', '')}"
            if base:
                snippet += f" : {base}"
            return SourceInfo(
                type=SourceType.CLASS,
                name=f"{entity.get('namespace', '')}.{entity.get('class_name', '')}",
                file=entity.get("file_path", ""),
                line=0,
                similarity=entity.get("similarity", 0.0),
                snippet=snippet
            )
        elif entity_type == SourceType.EVENT_HANDLER:
            return SourceInfo(
                type=SourceType.EVENT_HANDLER,
                name=f"{entity.get('event_name', '')} -> {entity.get('handler_method', '')}",
                file=entity.get("file_path", ""),
                line=entity.get("line_number", 0),
                similarity=entity.get("similarity", 0.0),
                snippet=f"{entity.get('element_name', '')}.{entity.get('event_name', '')} += {entity.get('handler_method', '')}"
            )
        else:
            return SourceInfo(
                type=entity_type,
                name=entity.get("id", "Unknown"),
                file=entity.get("file_path", ""),
                line=entity.get("line_number", 0),
                similarity=entity.get("similarity", 0.0),
                snippet=""
            )

    async def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about indexed code entities.

        Returns:
            Dict with counts for each entity type
        """
        await self._ensure_initialized()
        return await self._roslyn_service.get_stats()
