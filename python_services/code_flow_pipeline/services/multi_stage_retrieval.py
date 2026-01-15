"""
Multi-Stage Retrieval Service
=============================

Executes multi-hop retrieval across different MongoDB collections to build
comprehensive context for code flow analysis.

Design Rationale:
-----------------
Multi-stage retrieval is a RAG pattern that addresses the limitations of
single-query vector search. For complex code flow questions, relevant
information is spread across multiple collections:

1. Business Processes: High-level workflow documentation
2. Methods: Implementation details and code
3. Classes: Object structure and responsibilities
4. UI Events: User interface entry points
5. Call Relationships: Method call graph edges

By querying each collection in parallel and combining results, we build
a richer context that enables more accurate LLM synthesis.

Key Design Decisions:
1. Parallel Execution: All stage queries run concurrently for performance
2. Configurable Stages: Each query type activates different stages
3. Result Normalization: All results are converted to a common format
4. Metadata Preservation: Original metadata is retained for call chain building

Best Practices for Multi-Hop RAG:
---------------------------------
1. Start broad, then narrow: First stage retrieves many candidates,
   subsequent stages filter and expand
2. Use metadata for expansion: Retrieved documents often contain references
   to related documents (e.g., method calls)
3. Balance breadth vs depth: Too many hops add latency without improving quality
4. Cache intermediate results: Avoid redundant queries for the same documents
"""

import asyncio
import logging
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from code_flow_pipeline.models.query_models import (
    RetrievalStage,
    RetrievalStageType,
    RetrievalResults,
    FormattedResult,
    QueryClassification,
)
from code_flow_pipeline.services.query_classifier import get_query_classifier

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Context accumulated during multi-stage retrieval."""
    query: str
    project: Optional[str]
    classification: QueryClassification
    stages: List[RetrievalStage]
    results: RetrievalResults = field(default_factory=RetrievalResults)
    referenced_methods: set = field(default_factory=set)
    referenced_classes: set = field(default_factory=set)
    elapsed_ms: int = 0


class MultiStageRetrieval:
    """
    Executes multi-hop retrieval for code flow analysis.

    This service coordinates parallel vector searches across multiple
    MongoDB collections, aggregating results into a unified context.

    Usage:
        retrieval = MultiStageRetrieval(mongodb_service)
        results = await retrieval.execute(
            query="How are bales committed?",
            project="gin",
            classification=QueryClassification(type=QueryType.BUSINESS_PROCESS)
        )

    Architecture:
        - Uses MongoDB vector search via MongoDBService
        - Parallel execution via asyncio.gather
        - Result normalization via FormattedResult model
        - Metadata parsing for JSON-encoded arrays
    """

    def __init__(self, mongodb_service=None):
        """
        Initialize the multi-stage retrieval service.

        Args:
            mongodb_service: MongoDBService instance (lazy-loaded if None)
        """
        self._mongodb_service = mongodb_service
        self._classifier = get_query_classifier()

    async def _get_mongodb_service(self):
        """Lazy-load MongoDB service."""
        if self._mongodb_service is None:
            from mongodb import MongoDBService
            self._mongodb_service = MongoDBService.get_instance()
            if not self._mongodb_service.is_initialized:
                await self._mongodb_service.initialize()
        return self._mongodb_service

    async def execute(
        self,
        query: str,
        project: Optional[str] = None,
        classification: Optional[QueryClassification] = None,
        include_call_graph: bool = True,
        stages: Optional[List[RetrievalStage]] = None,
    ) -> RetrievalResults:
        """
        Execute multi-stage retrieval for a code flow query.

        Args:
            query: The natural language query
            project: Project scope (e.g., 'gin', 'warehouse')
            classification: Query classification (auto-classified if None)
            include_call_graph: Whether to include call graph retrieval
            stages: Custom stage configuration (auto-determined if None)

        Returns:
            RetrievalResults with documents from all stages

        Execution Strategy:
        1. Classify query (if not provided)
        2. Determine retrieval stages
        3. Execute all stages in parallel
        4. Aggregate and normalize results
        """
        mongodb = await self._get_mongodb_service()

        # Auto-classify if needed
        if classification is None:
            classification = self._classifier.classify(query)

        # Determine stages if not provided
        if stages is None:
            stages = self._classifier.get_retrieval_stages(
                classification,
                include_call_graph=include_call_graph
            )

        logger.info(
            f"Executing multi-stage retrieval for query type '{classification.type.value}' "
            f"with {len(stages)} stages"
        )

        # Build context for tracking
        context = RetrievalContext(
            query=query,
            project=project,
            classification=classification,
            stages=stages,
        )

        # Execute all stages in parallel
        stage_tasks = []
        for stage in stages:
            if stage.enabled:
                task = self._execute_stage(mongodb, context, stage)
                stage_tasks.append(task)

        # Wait for all stages to complete
        stage_results = await asyncio.gather(*stage_tasks, return_exceptions=True)

        # Process results
        for stage, result in zip(stages, stage_results):
            if isinstance(result, Exception):
                logger.error(f"Stage {stage.stage_type.value} failed: {result}")
                continue

            # Assign results to appropriate category
            self._assign_stage_results(context.results, stage.stage_type, result)

        # Compute total
        context.results.compute_total()

        logger.info(
            f"Multi-stage retrieval complete: {context.results.total_results} results "
            f"across {len([s for s in stages if s.enabled])} stages"
        )

        return context.results

    async def _execute_stage(
        self,
        mongodb,
        context: RetrievalContext,
        stage: RetrievalStage,
    ) -> List[FormattedResult]:
        """
        Execute a single retrieval stage.

        Args:
            mongodb: MongoDB service instance
            context: Retrieval context
            stage: Stage configuration

        Returns:
            List of formatted results
        """
        logger.debug(f"Executing stage: {stage.stage_type.value}")

        # Build filter
        filter_dict = {}
        if stage.filter_category:
            filter_dict["category"] = stage.filter_category
        if stage.filter_type:
            filter_dict["type"] = stage.filter_type

        try:
            # Execute vector search
            # Note: The actual collection is determined by the category filter
            results = await mongodb.search_vectors(
                query=context.query,
                project=context.project,
                category=stage.filter_category,
                doc_type=stage.filter_type,
                limit=stage.limit,
            )

            # Format results
            formatted = []
            for r in results:
                formatted_result = self._format_result(r)
                formatted.append(formatted_result)

                # Track referenced items for potential expansion
                metadata = r.get("metadata", {})
                if calls := self._try_parse_json(metadata.get("callsMethod")):
                    context.referenced_methods.update(calls)
                if called_by := self._try_parse_json(metadata.get("calledByMethod")):
                    context.referenced_methods.update(called_by)

            logger.debug(f"Stage {stage.stage_type.value} returned {len(formatted)} results")
            return formatted

        except Exception as e:
            logger.error(f"Stage {stage.stage_type.value} error: {e}")
            return []

    def _assign_stage_results(
        self,
        results: RetrievalResults,
        stage_type: RetrievalStageType,
        stage_results: List[FormattedResult],
    ):
        """Assign stage results to the appropriate results category."""
        if stage_type == RetrievalStageType.BUSINESS_PROCESS:
            results.business_processes.extend(stage_results)
        elif stage_type == RetrievalStageType.METHODS:
            results.methods.extend(stage_results)
        elif stage_type == RetrievalStageType.CLASSES:
            results.classes.extend(stage_results)
        elif stage_type == RetrievalStageType.UI_EVENTS:
            results.ui_events.extend(stage_results)
        elif stage_type == RetrievalStageType.CALL_GRAPH:
            results.call_relationships.extend(stage_results)

    def _format_result(self, raw_result: Dict[str, Any]) -> FormattedResult:
        """
        Format a raw MongoDB result into a FormattedResult.

        Args:
            raw_result: Raw document from MongoDB search

        Returns:
            Normalized FormattedResult
        """
        metadata = raw_result.get("metadata", {})
        content = raw_result.get("content", "")

        return FormattedResult(
            id=raw_result.get("id") or raw_result.get("_id"),
            similarity=raw_result.get("similarity", raw_result.get("score", 0.0)),
            content=content[:300] if content else None,
            metadata=metadata,
            project=metadata.get("project"),
            category=metadata.get("category"),
            type=metadata.get("type"),
        )

    def _try_parse_json(self, value: Any) -> List[str]:
        """
        Safely parse a JSON string to a list.

        MongoDB stores arrays as JSON strings in some cases.
        This method handles both cases gracefully.
        """
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

    async def expand_references(
        self,
        context: RetrievalContext,
        max_expansion: int = 5,
    ) -> RetrievalResults:
        """
        Expand results by following references to related documents.

        This is the "hop" in multi-hop retrieval. After the initial
        retrieval, we identify referenced methods/classes and fetch
        their full details.

        Args:
            context: Current retrieval context
            max_expansion: Maximum number of additional documents to fetch

        Returns:
            Updated RetrievalResults with expanded context

        Design Note:
        This method implements a breadth-first expansion strategy.
        For deep call chains, consider limiting depth or using
        iterative expansion with relevance filtering.
        """
        mongodb = await self._get_mongodb_service()

        # Identify methods we haven't retrieved yet
        retrieved_methods = {
            r.metadata.get("methodName", "")
            for r in context.results.methods
        }
        missing_methods = context.referenced_methods - retrieved_methods

        if not missing_methods:
            logger.debug("No additional methods to expand")
            return context.results

        # Limit expansion
        methods_to_fetch = list(missing_methods)[:max_expansion]
        logger.info(f"Expanding {len(methods_to_fetch)} referenced methods")

        # Fetch missing methods
        expansion_tasks = []
        for method_name in methods_to_fetch:
            task = mongodb.search_vectors(
                query=f"method {method_name}",
                project=context.project,
                category="code",
                doc_type="method",
                limit=1,
            )
            expansion_tasks.append(task)

        expansion_results = await asyncio.gather(*expansion_tasks, return_exceptions=True)

        # Add expansion results
        for result in expansion_results:
            if isinstance(result, Exception):
                continue
            for doc in result:
                formatted = self._format_result(doc)
                context.results.methods.append(formatted)

        context.results.compute_total()
        return context.results


# Singleton instance
_retrieval_instance: MultiStageRetrieval | None = None


async def get_multi_stage_retrieval() -> MultiStageRetrieval:
    """Get or create the singleton MultiStageRetrieval instance."""
    global _retrieval_instance
    if _retrieval_instance is None:
        _retrieval_instance = MultiStageRetrieval()
    return _retrieval_instance
