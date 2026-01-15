"""
Document Retrieval Pipeline Orchestrator - Coordinates all pipeline steps

This module provides the main entry point for the document retrieval pipeline,
orchestrating the flow through:
1. Query Understanding
2. Hybrid Retrieval
3. Document Grading
4. (Optional) Corrective Retrieval
5. Learning Feedback

The orchestrator implements the Corrective RAG (CRAG) pattern where
low-quality retrievals trigger corrective actions like query expansion
or alternative retrieval strategies.

Architecture:
------------
```
User Query
    |
    v
[Query Understanding] -> classify, rewrite, expand
    |
    v
[Hybrid Retrieval] -> vector + BM25 + RRF
    |
    v
[Document Grading] -> relevance filtering
    |
    +---> [Low Quality?] --yes--> [Corrective Retrieval]
    |                                    |
    v                                    v
[Graded Documents] <---------------------|
    |
    v
[Context for Generation]
    |
    v
[User Feedback] ---> [Learning Feedback Step]
```

Usage Example:
-------------
```python
from pipelines.document import DocumentRetrievalPipeline

# Initialize services
pipeline = DocumentRetrievalPipeline(
    mongodb_service=mongodb,
    embedding_service=embeddings,
    llm_service=llm,  # Optional for enhanced processing
)

# Execute retrieval
result = await pipeline.retrieve("What is the vacation policy?")

# Access results
if result.success:
    documents = result.graded_documents
    for doc in documents:
        print(f"{doc.title}: {doc.grading_score:.2f}")

# Record feedback
await pipeline.record_feedback(
    result.context,
    feedback_type="positive"
)
```

Configuration:
-------------
The pipeline can be configured for different use cases:
- Fast mode: Skip grading, simpler queries
- Accurate mode: Full pipeline with LLM grading
- Learning mode: Enhanced feedback recording
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import uuid

from .base import (
    PipelineStep,
    PipelineContext,
    StepResult,
    RetrievedDocument,
    QueryType,
)
from .query_understanding import QueryUnderstandingStep
from .hybrid_retrieval import HybridRetrievalStep, VectorOnlyRetrievalStep
from .document_grading import DocumentGradingStep
from .learning_feedback import LearningFeedbackStep, FeedbackType


@dataclass
class PipelineResult:
    """
    Result of a complete pipeline execution.

    Provides a high-level interface to pipeline outputs with
    convenience methods for common access patterns.
    """
    success: bool
    context: PipelineContext
    step_results: Dict[str, StepResult]
    error: Optional[str] = None
    total_duration_ms: float = 0.0

    @property
    def graded_documents(self) -> List[RetrievedDocument]:
        """Get the final graded documents."""
        return self.context.graded_documents

    @property
    def query_type(self) -> QueryType:
        """Get the classified query type."""
        return self.context.query_type

    @property
    def average_relevance(self) -> float:
        """Get the average relevance score."""
        return self.context.average_relevance

    @property
    def needs_fallback(self) -> bool:
        """Check if corrective retrieval is recommended."""
        return self.context.require_web_fallback

    def get_step_timing(self) -> Dict[str, float]:
        """Get timing for each step in milliseconds."""
        return {
            name: result.metadata.get("duration_ms", 0)
            for name, result in self.step_results.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error": self.error,
            "query_id": self.context.query_id,
            "original_query": self.context.original_query,
            "query_type": self.context.query_type.value,
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content[:500],  # Truncate for response
                    "score": doc.score,
                    "grading_score": doc.grading_score,
                    "source": doc.source,
                }
                for doc in self.graded_documents
            ],
            "average_relevance": self.average_relevance,
            "needs_fallback": self.needs_fallback,
            "total_duration_ms": self.total_duration_ms,
            "step_timings": self.get_step_timing(),
        }


class DocumentRetrievalPipeline:
    """
    Main orchestrator for the document retrieval pipeline.

    Coordinates the execution of all pipeline steps and handles
    error recovery, corrective retrieval, and feedback recording.
    """

    def __init__(
        self,
        mongodb_service: Any,
        embedding_service: Any,
        llm_service: Optional[Any] = None,
        collection_name: str = "documents",
        # Query understanding config
        enable_query_expansion: bool = True,
        expansion_count: int = 4,
        # Retrieval config
        use_hybrid_search: bool = True,
        top_k: int = 10,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        # Grading config
        enable_grading: bool = True,
        relevance_threshold: float = 0.5,
        min_documents: int = 1,
        max_documents: int = 5,
        # Corrective retrieval config
        enable_corrective_retrieval: bool = True,
        corrective_threshold: float = 0.4,
        max_correction_attempts: int = 1,
        # Feedback config
        enable_learning: bool = True,
        # Logging
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        """
        Initialize the document retrieval pipeline.

        Args:
            mongodb_service: MongoDBService for database operations
            embedding_service: EmbeddingService for embeddings
            llm_service: Optional LLM service for enhanced processing
            collection_name: MongoDB collection to search

            Query understanding:
            enable_query_expansion: Whether to expand queries
            expansion_count: Number of query variants (3-5)

            Retrieval:
            use_hybrid_search: Use hybrid vs vector-only
            top_k: Number of documents to retrieve
            rrf_k: RRF fusion constant
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)

            Grading:
            enable_grading: Whether to grade documents
            relevance_threshold: Minimum relevance score
            min_documents: Minimum docs to return
            max_documents: Maximum docs to return

            Corrective:
            enable_corrective_retrieval: Retry on low quality
            corrective_threshold: Threshold for triggering correction
            max_correction_attempts: Max retry attempts

            Learning:
            enable_learning: Whether to record feedback

            Logging:
            logger: Optional logger
            verbose: Enable verbose logging
        """
        self.mongodb_service = mongodb_service
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.collection_name = collection_name
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose

        # Store config
        self.enable_corrective = enable_corrective_retrieval
        self.corrective_threshold = corrective_threshold
        self.max_corrections = max_correction_attempts

        # Initialize pipeline steps
        self.query_step = QueryUnderstandingStep(
            llm_service=llm_service,
            expansion_count=expansion_count if enable_query_expansion else 1,
            enable_entity_extraction=True,
            logger=self.logger,
        )

        if use_hybrid_search:
            self.retrieval_step = HybridRetrievalStep(
                mongodb_service=mongodb_service,
                embedding_service=embedding_service,
                collection_name=collection_name,
                rrf_k=rrf_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                top_k=top_k,
                logger=self.logger,
            )
        else:
            self.retrieval_step = VectorOnlyRetrievalStep(
                mongodb_service=mongodb_service,
                embedding_service=embedding_service,
                collection_name=collection_name,
                top_k=top_k,
                logger=self.logger,
            )

        self.grading_step = DocumentGradingStep(
            llm_service=llm_service if enable_grading else None,
            relevance_threshold=relevance_threshold,
            min_documents=min_documents,
            max_documents=max_documents,
            logger=self.logger,
        ) if enable_grading else None

        self.feedback_step = LearningFeedbackStep(
            mongodb_service=mongodb_service,
            embedding_service=embedding_service,
            logger=self.logger,
        ) if enable_learning else None

    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        session_history: Optional[List[str]] = None,
        skip_grading: bool = False,
    ) -> PipelineResult:
        """
        Execute the full retrieval pipeline for a query.

        This is the main entry point for document retrieval. It
        orchestrates all pipeline steps and handles errors gracefully.

        Args:
            query: The user's natural language query
            user_id: Optional user identifier for personalization
            filters: Optional filters (department, type, etc.)
            session_history: Previous queries in session
            skip_grading: Force skip document grading

        Returns:
            PipelineResult with retrieved documents and metadata
        """
        start_time = datetime.utcnow()
        step_results: Dict[str, StepResult] = {}

        # Initialize context
        context = PipelineContext(
            query_id=str(uuid.uuid4()),
            original_query=query,
            timestamp=start_time,
            user_id=user_id,
            filters=filters or {},
            session_history=session_history or [],
            skip_grading=skip_grading,
        )

        try:
            # Step 1: Query Understanding
            self._log(f"Step 1: Query Understanding for '{query[:50]}...'")
            query_result = await self.query_step.run(context)
            step_results["query_understanding"] = query_result

            if not query_result.success:
                return self._error_result(
                    context, step_results, start_time,
                    f"Query understanding failed: {query_result.errors}"
                )

            context = context.merge_result(query_result)
            self._log(f"  - Query type: {context.query_type.value}")
            self._log(f"  - Expanded to {len(context.expanded_queries)} variants")

            # Step 2: Hybrid Retrieval
            self._log("Step 2: Hybrid Retrieval")
            retrieval_result = await self.retrieval_step.run(context)
            step_results["retrieval"] = retrieval_result

            if not retrieval_result.success:
                return self._error_result(
                    context, step_results, start_time,
                    f"Retrieval failed: {retrieval_result.errors}"
                )

            context = context.merge_result(retrieval_result)
            self._log(f"  - Retrieved {len(context.retrieved_documents)} documents")
            self._log(f"  - Method: {context.retrieval_method}")

            # Step 3: Document Grading (optional)
            if self.grading_step and not context.skip_grading:
                self._log("Step 3: Document Grading")
                grading_result = await self.grading_step.run(context)
                step_results["grading"] = grading_result

                if grading_result.success:
                    context = context.merge_result(grading_result)
                    self._log(f"  - Graded documents: {len(context.graded_documents)}")
                    self._log(f"  - Average relevance: {context.average_relevance:.2f}")
                    self._log(f"  - Filtered out: {context.documents_filtered_count}")
                else:
                    # Grading failed, use retrieved docs
                    context.graded_documents = context.retrieved_documents[:5]
                    self._log(f"  - Grading failed, using retrieval results")

                # Step 4: Corrective Retrieval (if needed)
                if (self.enable_corrective and context.require_web_fallback and
                    context.average_relevance < self.corrective_threshold):
                    self._log("Step 4: Corrective Retrieval (low relevance)")
                    corrective_result = await self._corrective_retrieval(context)
                    step_results["corrective_retrieval"] = corrective_result

                    if corrective_result.success:
                        context = context.merge_result(corrective_result)
                        self._log(f"  - Corrected to {len(context.graded_documents)} docs")
            else:
                # No grading, use retrieved documents directly
                context.graded_documents = context.retrieved_documents[:5]
                self._log("Step 3: Grading skipped, using retrieval results")

            # Calculate total duration
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self._log(f"Pipeline complete in {duration_ms:.2f}ms")

            return PipelineResult(
                success=True,
                context=context,
                step_results=step_results,
                total_duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("Pipeline execution failed")
            return self._error_result(
                context, step_results, start_time,
                f"Pipeline exception: {str(e)}"
            )

    async def _corrective_retrieval(
        self,
        context: PipelineContext
    ) -> StepResult:
        """
        Attempt corrective retrieval when initial results are poor.

        Strategies:
        1. Expand query further with different approaches
        2. Relax filters
        3. Use alternative retrieval method
        """
        self._log("  Attempting corrective retrieval...")

        # Strategy 1: Add more query variants using different phrasing
        additional_queries = []

        # Try a hypothetical document approach (HyDE-lite)
        if self.llm_service:
            try:
                hyde_prompt = f"""Write a brief passage that would answer this question:
{context.original_query}

Write as if from a knowledge base document, 2-3 sentences."""

                hyde_response = await self.llm_service.generate(
                    prompt=hyde_prompt,
                    max_tokens=100,
                    temperature=0.7,
                )
                additional_queries.append(hyde_response.strip())
            except Exception:
                pass

        # Add question word variants
        original = context.original_query.lower()
        if not original.startswith("how"):
            additional_queries.append(f"How to {original}")
        if not original.startswith("what"):
            additional_queries.append(f"What is {original}")

        # Create new context with expanded queries
        corrective_context = PipelineContext(
            query_id=context.query_id,
            original_query=context.original_query,
            rewritten_query=context.rewritten_query,
            expanded_queries=context.expanded_queries + additional_queries,
            extracted_entities=context.extracted_entities,
            query_type=context.query_type,
            filters={},  # Relax filters
            skip_grading=True,  # Skip grading on retry
        )

        # Re-run retrieval with expanded queries
        retrieval_result = await self.retrieval_step.run(corrective_context)

        if retrieval_result.success:
            # Merge with original results, keeping best from both
            new_docs = retrieval_result.data.get("retrieved_documents", [])
            existing_ids = {d.id for d in context.graded_documents}

            combined = list(context.graded_documents)
            for doc in new_docs:
                if doc.id not in existing_ids:
                    combined.append(doc)

            # Re-sort by score and limit
            combined.sort(key=lambda d: d.score, reverse=True)
            combined = combined[:5]

            return StepResult(
                success=True,
                data={
                    "graded_documents": combined,
                    "require_web_fallback": len(combined) < 2,
                },
                metadata={"corrective_docs_added": len(new_docs)},
            )

        return StepResult(
            success=False,
            errors=["Corrective retrieval failed"],
        )

    async def record_feedback(
        self,
        context: PipelineContext,
        feedback_type: str,
        feedback_text: Optional[str] = None,
        corrected_answer: Optional[str] = None,
        user_id: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> StepResult:
        """
        Record user feedback for a pipeline execution.

        This should be called after the user provides feedback
        (thumbs up/down, correction, etc.).

        Args:
            context: The PipelineContext from retrieve()
            feedback_type: "positive", "negative", or "correction"
            feedback_text: Optional detailed feedback
            corrected_answer: User's correction if type is "correction"
            user_id: Optional user identifier
            answer: The answer that was generated

        Returns:
            StepResult indicating success/failure
        """
        if self.feedback_step is None:
            return StepResult(
                success=False,
                errors=["Learning is not enabled"],
            )

        return await self.feedback_step.record_feedback(
            context=context,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            corrected_answer=corrected_answer,
            user_id=user_id or context.user_id,
            answer=answer,
        )

    async def get_feedback_stats(
        self,
        days: int = 30,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get feedback statistics for monitoring.

        Args:
            days: Number of days to analyze
            user_id: Optional user filter

        Returns:
            Dictionary with feedback statistics
        """
        if self.feedback_step is None:
            return {"error": "Learning is not enabled"}

        return await self.feedback_step.get_feedback_statistics(
            days=days,
            user_id=user_id
        )

    def _error_result(
        self,
        context: PipelineContext,
        step_results: Dict[str, StepResult],
        start_time: datetime,
        error: str
    ) -> PipelineResult:
        """Create an error result."""
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.logger.error(f"Pipeline error: {error}")

        return PipelineResult(
            success=False,
            context=context,
            step_results=step_results,
            error=error,
            total_duration_ms=duration_ms,
        )

    def _log(self, message: str):
        """Log if verbose mode is enabled."""
        if self.verbose:
            self.logger.info(message)
        else:
            self.logger.debug(message)


class SimplifiedRetrievalPipeline(DocumentRetrievalPipeline):
    """
    Simplified pipeline for fast retrieval without LLM overhead.

    This variant skips LLM-based query understanding and grading,
    relying on pattern-based processing and retrieval scores.
    Suitable for:
    - Low-latency requirements
    - High-volume queries
    - When LLM is unavailable
    """

    def __init__(
        self,
        mongodb_service: Any,
        embedding_service: Any,
        collection_name: str = "documents",
        top_k: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            mongodb_service=mongodb_service,
            embedding_service=embedding_service,
            llm_service=None,  # No LLM
            collection_name=collection_name,
            enable_query_expansion=False,
            use_hybrid_search=True,
            top_k=top_k,
            enable_grading=False,  # Skip grading
            enable_corrective_retrieval=False,
            enable_learning=False,
            logger=logger,
        )
