"""
Prefect Document Q&A Pipeline

Provides Prefect-based workflow orchestration for the Document Q&A (RAG) pipeline.
This flow wraps the existing document retrieval pipeline with Prefect tasks for:
- Query Understanding - Query expansion and filter extraction
- Hybrid Retrieval - Vector + BM25 search
- Document Grading - Relevance scoring
- Answer Generation - LLM synthesis with citations
- Validation - Hallucination detection
- Learning Feedback - Record interaction

Features:
- Built-in retries for resilience
- Prefect artifact generation for observability
- Per-step timing metrics
- User tracking across all tasks
- Fallback to direct pipeline if Prefect fails
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class QueryUnderstandingResult:
    """Result from query understanding task."""
    original_query: str
    expanded_query: str = ""
    query_type: str = "factual"
    extracted_filters: Dict[str, Any] = field(default_factory=dict)
    clarification_needed: bool = False
    suggested_refinements: List[str] = field(default_factory=list)
    elapsed_ms: int = 0
    success: bool = True
    error: str = ""


@dataclass
class HybridRetrievalResult:
    """Result from hybrid retrieval task."""
    query: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunks_count: int = 0
    retrieval_methods: Dict[str, str] = field(default_factory=dict)
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: int = 0
    success: bool = True
    error: str = ""


@dataclass
class DocumentGradingResult:
    """Result from document grading task."""
    graded_chunks: List[Dict[str, Any]] = field(default_factory=list)
    rejected_chunks: List[Dict[str, Any]] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    graded_count: int = 0
    rejected_count: int = 0
    elapsed_ms: int = 0
    success: bool = True
    error: str = ""


@dataclass
class AnswerGenerationResult:
    """Result from answer generation task."""
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    chunks_used: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    success: bool = True
    error: str = ""


@dataclass
class ValidationResult:
    """Result from validation task."""
    is_valid: bool = False
    hallucination_score: float = 0.0
    completeness_score: float = 0.0
    validation_issues: List[str] = field(default_factory=list)
    elapsed_ms: int = 0
    success: bool = True
    error: str = ""


@dataclass
class LearningFeedbackResult:
    """Result from learning feedback task."""
    learning_record_id: str = ""
    stored: bool = False
    elapsed_ms: int = 0
    success: bool = True
    error: str = ""


# ============================================================================
# Prefect Tasks
# ============================================================================

@task(
    name="query_understanding",
    description="Expand query and extract filters for document retrieval",
    retries=2,
    retry_delay_seconds=5,
    tags=["document", "query", "understanding"]
)
async def query_understanding_task(
    query: str,
    user_id: str,
    filters: Optional[Dict[str, Any]] = None,
    context_history: Optional[List[str]] = None
) -> QueryUnderstandingResult:
    """
    Query understanding task - expands query and extracts filters.

    This step enhances the user query by:
    - Expanding with synonyms and related terms
    - Extracting structured filter criteria
    - Detecting query type (factual, procedural, exploratory)
    - Identifying ambiguities

    Args:
        query: Original user query
        user_id: User identifier for tracking
        filters: Optional pre-specified filters
        context_history: Previous conversation context

    Returns:
        QueryUnderstandingResult with expanded query and filters
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Query understanding for user {user_id}: '{query}'")

    result = QueryUnderstandingResult(original_query=query)

    try:
        # Import the pipeline step
        from pipelines.document.steps import QueryUnderstandingStep
        from pipelines.document.context import DocumentPipelineContext

        # Create context
        context = DocumentPipelineContext(
            query=query,
            filters=filters or {},
            context_history=context_history or []
        )

        # Execute step
        step = QueryUnderstandingStep(enabled=True, llm_model="general")
        step_result = await step.execute(context)

        if step_result.success:
            result.expanded_query = context.expanded_query or query
            result.query_type = context.query_type or "factual"
            result.extracted_filters = context.extracted_filters
            result.clarification_needed = context.clarification_needed
            result.suggested_refinements = context.suggested_refinements
            result.success = True
        else:
            result.expanded_query = query  # Fallback to original
            result.error = step_result.error or "Unknown error"
            result.success = True  # Non-fatal

        logger.info(
            f"Query understood: type={result.query_type}, "
            f"expanded={result.expanded_query[:100]}..."
        )

    except Exception as e:
        logger.error(f"Query understanding failed: {e}")
        result.expanded_query = query  # Fallback
        result.error = str(e)
        result.success = True  # Non-fatal, continue with original query

    result.elapsed_ms = int((time.time() - start_time) * 1000)

    # Create Prefect artifact
    await create_markdown_artifact(
        key="query-understanding",
        markdown=f"""
## Query Understanding Results

- **User**: {user_id}
- **Original Query**: {query}
- **Expanded Query**: {result.expanded_query}
- **Query Type**: {result.query_type}
- **Clarification Needed**: {result.clarification_needed}
- **Extracted Filters**: {result.extracted_filters or 'None'}
- **Processing Time**: {result.elapsed_ms}ms
- **Status**: {"Success" if result.success else "Failed"}
{f"- **Error**: {result.error}" if result.error else ""}
        """,
        description=f"Query understanding for '{query[:50]}...'"
    )

    return result


@task(
    name="hybrid_retrieval",
    description="Perform vector + keyword search for document retrieval",
    retries=2,
    retry_delay_seconds=10,
    tags=["document", "retrieval", "search"]
)
async def hybrid_retrieval_task(
    query_result: QueryUnderstandingResult,
    user_id: str,
    top_k: int = 5,
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2,
    filters: Optional[Dict[str, Any]] = None
) -> HybridRetrievalResult:
    """
    Hybrid retrieval task - combines vector and keyword search.

    This step performs:
    - Vector similarity search (semantic matching)
    - Keyword/BM25 search (exact term matching)
    - Merges and deduplicates results
    - Applies filters

    Args:
        query_result: Result from query understanding
        user_id: User identifier for tracking
        top_k: Number of documents to retrieve
        vector_weight: Weight for vector search (0-1)
        keyword_weight: Weight for keyword search (0-1)
        filters: Additional filter criteria

    Returns:
        HybridRetrievalResult with retrieved chunks
    """
    logger = get_run_logger()
    start_time = time.time()

    query = query_result.expanded_query or query_result.original_query
    logger.info(f"Hybrid retrieval for user {user_id}: '{query[:50]}...', top_k={top_k}")

    result = HybridRetrievalResult(query=query)

    try:
        # Import the pipeline step
        from pipelines.document.steps import HybridRetrievalStep
        from pipelines.document.context import DocumentPipelineContext

        # Merge filters
        combined_filters = {**(filters or {}), **query_result.extracted_filters}

        # Create context
        context = DocumentPipelineContext(
            query=query_result.original_query,
            expanded_query=query,
            filters=combined_filters,
            top_k=top_k
        )

        # Execute step
        step = HybridRetrievalStep(
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            use_reranking=True,
            candidate_multiplier=4
        )
        step_result = await step.execute(context)

        if step_result.success:
            # Convert chunks to serializable dicts
            result.retrieved_chunks = [
                {
                    "id": getattr(c, 'id', str(c)),
                    "content": getattr(c, 'content', str(c))[:500],
                    "source": getattr(c, 'source', 'unknown'),
                    "score": context.retrieval_scores.get(getattr(c, 'id', str(c)), 0)
                }
                for c in context.retrieved_chunks
            ]
            result.chunks_count = context.chunks_retrieved
            result.retrieval_methods = context.retrieval_methods
            result.retrieval_scores = context.retrieval_scores
            result.success = True
        else:
            result.error = step_result.error or "Retrieval failed"
            result.success = False

        logger.info(f"Retrieved {result.chunks_count} chunks")

    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        result.error = str(e)
        result.success = False

    result.elapsed_ms = int((time.time() - start_time) * 1000)

    # Create Prefect artifact
    await create_markdown_artifact(
        key="hybrid-retrieval",
        markdown=f"""
## Hybrid Retrieval Results

- **User**: {user_id}
- **Query**: {query[:100]}...
- **Top K**: {top_k}
- **Vector Weight**: {vector_weight}
- **Keyword Weight**: {keyword_weight}
- **Chunks Retrieved**: {result.chunks_count}
- **Processing Time**: {result.elapsed_ms}ms
- **Status**: {"Success" if result.success else "Failed"}
{f"- **Error**: {result.error}" if result.error else ""}

### Retrieved Chunks Preview
{chr(10).join(f"- [{c.get('source', 'unknown')}] Score: {c.get('score', 0):.3f}" for c in result.retrieved_chunks[:5]) or "No chunks retrieved"}
        """,
        description=f"Hybrid retrieval: {result.chunks_count} chunks"
    )

    return result


@task(
    name="document_grading",
    description="Score and filter documents for relevance",
    retries=2,
    retry_delay_seconds=5,
    tags=["document", "grading", "relevance"]
)
async def document_grading_task(
    retrieval_result: HybridRetrievalResult,
    query_result: QueryUnderstandingResult,
    user_id: str,
    min_score: float = 5.0,
    use_llm_grading: bool = True
) -> DocumentGradingResult:
    """
    Document grading task - scores and filters documents.

    This step:
    - Applies LLM-based relevance scoring (0-10 scale)
    - Applies recency weighting
    - Applies source trust scoring
    - Filters out low-scoring chunks

    Args:
        retrieval_result: Result from hybrid retrieval
        query_result: Result from query understanding
        user_id: User identifier for tracking
        min_score: Minimum relevance score to keep
        use_llm_grading: Whether to use LLM for grading

    Returns:
        DocumentGradingResult with graded and filtered chunks
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Document grading for user {user_id}: {len(retrieval_result.retrieved_chunks)} chunks")

    result = DocumentGradingResult()

    try:
        if not retrieval_result.success or not retrieval_result.retrieved_chunks:
            logger.warning("No chunks to grade")
            result.success = True
            result.elapsed_ms = int((time.time() - start_time) * 1000)
            return result

        # Import the pipeline step
        from pipelines.document.steps import DocumentGradingStep
        from pipelines.document.context import DocumentPipelineContext

        # Create context with retrieved chunks
        context = DocumentPipelineContext(
            query=query_result.original_query,
            expanded_query=query_result.expanded_query
        )
        context.retrieved_chunks = retrieval_result.retrieved_chunks

        # Execute step
        step = DocumentGradingStep(
            min_score=min_score,
            use_llm_grading=use_llm_grading,
            recency_weight=0.2,
            trust_weight=0.1
        )
        step_result = await step.execute(context)

        if step_result.success:
            result.graded_chunks = [
                {
                    "id": getattr(c, 'id', c.get('id', str(c))),
                    "content": getattr(c, 'content', c.get('content', str(c)))[:500],
                    "source": getattr(c, 'source', c.get('source', 'unknown')),
                    "score": context.relevance_scores.get(
                        getattr(c, 'id', c.get('id', str(c))), 0
                    )
                }
                for c in context.graded_chunks
            ]
            result.rejected_chunks = [
                {"id": getattr(c, 'id', c.get('id', str(c)))}
                for c in context.rejected_chunks
            ]
            result.relevance_scores = context.relevance_scores
            result.graded_count = len(result.graded_chunks)
            result.rejected_count = len(result.rejected_chunks)
            result.success = True
        else:
            # Non-fatal: use ungraded chunks
            result.graded_chunks = retrieval_result.retrieved_chunks
            result.graded_count = len(result.graded_chunks)
            result.error = step_result.error or "Grading failed"
            result.success = True

        logger.info(
            f"Graded {result.graded_count} chunks, rejected {result.rejected_count}"
        )

    except Exception as e:
        logger.error(f"Document grading failed: {e}")
        # Non-fatal: use ungraded chunks
        result.graded_chunks = retrieval_result.retrieved_chunks
        result.graded_count = len(result.graded_chunks)
        result.error = str(e)
        result.success = True

    result.elapsed_ms = int((time.time() - start_time) * 1000)

    # Create Prefect artifact
    await create_markdown_artifact(
        key="document-grading",
        markdown=f"""
## Document Grading Results

- **User**: {user_id}
- **Input Chunks**: {len(retrieval_result.retrieved_chunks)}
- **Graded Chunks**: {result.graded_count}
- **Rejected Chunks**: {result.rejected_count}
- **Min Score Threshold**: {min_score}
- **LLM Grading**: {use_llm_grading}
- **Processing Time**: {result.elapsed_ms}ms
- **Status**: {"Success" if result.success else "Failed"}
{f"- **Error**: {result.error}" if result.error else ""}

### Top Graded Chunks
{chr(10).join(f"- Score {c.get('score', 0):.2f}: {c.get('source', 'unknown')}" for c in sorted(result.graded_chunks, key=lambda x: x.get('score', 0), reverse=True)[:5]) or "No graded chunks"}
        """,
        description=f"Document grading: {result.graded_count} passed"
    )

    return result


@task(
    name="answer_generation",
    description="Generate answer from document context using LLM",
    retries=2,
    retry_delay_seconds=15,
    tags=["document", "generation", "llm"]
)
async def answer_generation_task(
    grading_result: DocumentGradingResult,
    query_result: QueryUnderstandingResult,
    user_id: str,
    max_context_tokens: int = 4000,
    temperature: float = 0.3,
    include_citations: bool = True
) -> AnswerGenerationResult:
    """
    Answer generation task - synthesizes answer from context.

    This step:
    - Builds context window from top-ranked chunks
    - Generates comprehensive answer using LLM
    - Extracts and formats citations
    - Calculates confidence score

    Args:
        grading_result: Result from document grading
        query_result: Result from query understanding
        user_id: User identifier for tracking
        max_context_tokens: Maximum tokens for context window
        temperature: LLM temperature
        include_citations: Whether to include citations

    Returns:
        AnswerGenerationResult with answer and citations
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Answer generation for user {user_id}: {len(grading_result.graded_chunks)} context chunks")

    result = AnswerGenerationResult()

    try:
        if not grading_result.graded_chunks:
            result.answer = "I couldn't find any relevant information to answer your question."
            result.confidence = 0.0
            result.success = True
            result.elapsed_ms = int((time.time() - start_time) * 1000)
            return result

        # Import the pipeline step
        from pipelines.document.steps import AnswerGenerationStep
        from pipelines.document.context import DocumentPipelineContext

        # Create context
        context = DocumentPipelineContext(
            query=query_result.original_query,
            expanded_query=query_result.expanded_query,
            include_citations=include_citations
        )
        context.graded_chunks = grading_result.graded_chunks

        # Execute step
        step = AnswerGenerationStep(
            llm_model="general",
            max_context_tokens=max_context_tokens,
            temperature=temperature
        )
        step_result = await step.execute(context)

        if step_result.success:
            result.answer = context.answer or ""
            result.citations = [
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "relevance_score": c.relevance_score,
                    "excerpt": c.excerpt
                }
                for c in context.citations
            ]
            result.confidence = context.confidence
            result.chunks_used = context.chunks_used
            result.success = True
        else:
            result.answer = f"Error generating answer: {step_result.error}"
            result.error = step_result.error or "Generation failed"
            result.success = False

        logger.info(
            f"Answer generated: {len(result.answer)} chars, "
            f"confidence={result.confidence:.2f}, citations={len(result.citations)}"
        )

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        result.answer = f"Error generating answer: {str(e)}"
        result.error = str(e)
        result.success = False

    result.elapsed_ms = int((time.time() - start_time) * 1000)

    # Create Prefect artifact
    await create_markdown_artifact(
        key="answer-generation",
        markdown=f"""
## Answer Generation Results

- **User**: {user_id}
- **Query**: {query_result.original_query[:100]}...
- **Context Chunks Used**: {result.chunks_used}
- **Answer Length**: {len(result.answer)} chars
- **Confidence**: {result.confidence:.2f}
- **Citations**: {len(result.citations)}
- **Processing Time**: {result.elapsed_ms}ms
- **Status**: {"Success" if result.success else "Failed"}
{f"- **Error**: {result.error}" if result.error else ""}

### Answer Preview
{result.answer[:500]}{'...' if len(result.answer) > 500 else ''}

### Citations
{chr(10).join(f"- [{c.get('source', 'unknown')}] Relevance: {c.get('relevance_score', 0):.2f}" for c in result.citations[:5]) or "No citations"}
        """,
        description=f"Answer generation: confidence {result.confidence:.2f}"
    )

    return result


@task(
    name="validation",
    description="Validate answer quality and detect hallucinations",
    retries=1,
    retry_delay_seconds=5,
    tags=["document", "validation", "quality"]
)
async def validation_task(
    generation_result: AnswerGenerationResult,
    grading_result: DocumentGradingResult,
    query_result: QueryUnderstandingResult,
    user_id: str,
    check_hallucinations: bool = True,
    check_completeness: bool = True
) -> ValidationResult:
    """
    Validation task - verifies answer quality.

    This step:
    - Checks citations are valid and accurate
    - Detects hallucinated information (not in context)
    - Verifies answer completeness
    - Flags low-confidence answers

    Args:
        generation_result: Result from answer generation
        grading_result: Result from document grading
        query_result: Result from query understanding
        user_id: User identifier for tracking
        check_hallucinations: Enable hallucination detection
        check_completeness: Enable completeness checking

    Returns:
        ValidationResult with validation status
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Validation for user {user_id}")

    result = ValidationResult()

    try:
        if not generation_result.success or not generation_result.answer:
            result.is_valid = False
            result.validation_issues.append("No answer to validate")
            result.success = True
            result.elapsed_ms = int((time.time() - start_time) * 1000)
            return result

        # Import the pipeline step
        from pipelines.document.steps import ValidationStep
        from pipelines.document.context import DocumentPipelineContext

        # Create context
        context = DocumentPipelineContext(
            query=query_result.original_query
        )
        context.answer = generation_result.answer
        context.graded_chunks = grading_result.graded_chunks
        context.citations = []  # Will be populated from generation_result

        # Execute step
        step = ValidationStep(
            enabled=True,
            check_hallucinations=check_hallucinations,
            check_completeness=check_completeness,
            max_hallucination_score=0.3
        )
        step_result = await step.execute(context)

        result.is_valid = context.is_valid
        result.hallucination_score = context.hallucination_score
        result.completeness_score = context.completeness_score
        result.validation_issues = context.validation_issues
        result.success = True

        logger.info(
            f"Validation: valid={result.is_valid}, "
            f"hallucination={result.hallucination_score:.2f}, "
            f"completeness={result.completeness_score:.2f}"
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        result.is_valid = False
        result.validation_issues.append(f"Validation error: {str(e)}")
        result.error = str(e)
        result.success = True  # Non-fatal

    result.elapsed_ms = int((time.time() - start_time) * 1000)

    # Create Prefect artifact
    await create_markdown_artifact(
        key="validation",
        markdown=f"""
## Validation Results

- **User**: {user_id}
- **Is Valid**: {result.is_valid}
- **Hallucination Score**: {result.hallucination_score:.2f}
- **Completeness Score**: {result.completeness_score:.2f}
- **Processing Time**: {result.elapsed_ms}ms
- **Status**: {"Success" if result.success else "Failed"}
{f"- **Error**: {result.error}" if result.error else ""}

### Validation Issues
{chr(10).join(f"- {issue}" for issue in result.validation_issues) or "No issues detected"}
        """,
        description=f"Validation: {'passed' if result.is_valid else 'failed'}"
    )

    return result


@task(
    name="learning_feedback",
    description="Record interaction for continuous improvement",
    retries=2,
    retry_delay_seconds=5,
    tags=["document", "learning", "feedback"]
)
async def learning_feedback_task(
    query_result: QueryUnderstandingResult,
    retrieval_result: HybridRetrievalResult,
    grading_result: DocumentGradingResult,
    generation_result: AnswerGenerationResult,
    validation_result: ValidationResult,
    user_id: str,
    learning_enabled: bool = True
) -> LearningFeedbackResult:
    """
    Learning feedback task - records interaction for improvement.

    This step:
    - Logs query, results, and performance metrics
    - Stores data for future analysis
    - Enables reinforcement learning improvements

    Args:
        query_result: Result from query understanding
        retrieval_result: Result from hybrid retrieval
        grading_result: Result from document grading
        generation_result: Result from answer generation
        validation_result: Result from validation
        user_id: User identifier for tracking
        learning_enabled: Whether learning is enabled

    Returns:
        LearningFeedbackResult with record ID
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Learning feedback for user {user_id}")

    result = LearningFeedbackResult()

    if not learning_enabled:
        logger.info("Learning feedback disabled, skipping")
        result.success = True
        result.elapsed_ms = int((time.time() - start_time) * 1000)
        return result

    try:
        # Import the pipeline step
        from pipelines.document.steps import LearningFeedbackStep
        from pipelines.document.context import DocumentPipelineContext

        # Create context with all results
        context = DocumentPipelineContext(
            query=query_result.original_query,
            expanded_query=query_result.expanded_query,
            query_type=query_result.query_type
        )
        context.answer = generation_result.answer
        context.confidence = generation_result.confidence
        context.is_valid = validation_result.is_valid
        context.hallucination_score = validation_result.hallucination_score
        context.completeness_score = validation_result.completeness_score
        context.validation_issues = validation_result.validation_issues
        context.chunks_retrieved = retrieval_result.chunks_count
        context.chunks_used = generation_result.chunks_used

        # Calculate total processing time
        total_time = (
            query_result.elapsed_ms +
            retrieval_result.elapsed_ms +
            grading_result.elapsed_ms +
            generation_result.elapsed_ms +
            validation_result.elapsed_ms
        )
        context.processing_time_ms = total_time

        # Execute step
        step = LearningFeedbackStep(
            enabled=True,
            auto_update_embeddings=False,
            auto_adjust_weights=False
        )
        step_result = await step.execute(context)

        result.learning_record_id = context.learning_record_id or ""
        result.stored = bool(result.learning_record_id)
        result.success = True

        logger.info(f"Learning record stored: {result.learning_record_id}")

    except Exception as e:
        logger.error(f"Learning feedback failed: {e}")
        result.error = str(e)
        result.success = True  # Non-fatal

    result.elapsed_ms = int((time.time() - start_time) * 1000)

    # Create Prefect artifact
    await create_markdown_artifact(
        key="learning-feedback",
        markdown=f"""
## Learning Feedback Results

- **User**: {user_id}
- **Record ID**: {result.learning_record_id or 'Not stored'}
- **Stored**: {result.stored}
- **Processing Time**: {result.elapsed_ms}ms
- **Status**: {"Success" if result.success else "Failed"}
{f"- **Error**: {result.error}" if result.error else ""}
        """,
        description=f"Learning feedback: {'stored' if result.stored else 'not stored'}"
    )

    return result


# ============================================================================
# Main Prefect Flow
# ============================================================================

@flow(
    name="document-qa-pipeline",
    description="Document Q&A Pipeline with CRAG pattern - Query Understanding, Hybrid Retrieval, Grading, Generation, Validation, and Learning",
    retries=1,
    retry_delay_seconds=30
)
async def document_qa_flow(
    query: str,
    user_id: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    include_citations: bool = True,
    skip_validation: bool = False,
    learning_enabled: bool = True,
    temperature: float = 0.3,
    max_context_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Document Q&A Pipeline Flow.

    This flow orchestrates the complete document retrieval and answer generation
    pipeline with Prefect tracking:

    1. Query Understanding - Expand query and extract filters
    2. Hybrid Retrieval - Vector + keyword search
    3. Document Grading - Relevance scoring and filtering
    4. Answer Generation - LLM synthesis with citations
    5. Validation - Hallucination detection and quality checks
    6. Learning Feedback - Record for continuous improvement

    Args:
        query: User's natural language query
        user_id: User identifier for tracking and personalization
        filters: Optional filter criteria (project, department, type, etc.)
        top_k: Number of documents to retrieve
        include_citations: Whether to include source citations
        skip_validation: Skip answer validation step
        learning_enabled: Enable learning feedback recording
        temperature: LLM temperature for generation
        max_context_tokens: Maximum tokens for context window

    Returns:
        Dict with comprehensive pipeline results including:
        - answer: Generated answer
        - citations: Source citations
        - confidence: Answer confidence score
        - validation: Validation results
        - timing: Per-step timing breakdown
        - metadata: Query and processing metadata
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting Document Q&A Pipeline for user {user_id}: '{query[:50]}...'")

    # Step 1: Query Understanding
    query_result = await query_understanding_task(
        query=query,
        user_id=user_id,
        filters=filters,
        context_history=[]
    )

    # Step 2: Hybrid Retrieval
    retrieval_result = await hybrid_retrieval_task(
        query_result=query_result,
        user_id=user_id,
        top_k=top_k,
        filters=filters
    )

    # Step 3: Document Grading
    grading_result = await document_grading_task(
        retrieval_result=retrieval_result,
        query_result=query_result,
        user_id=user_id,
        min_score=5.0,
        use_llm_grading=True
    )

    # Step 4: Answer Generation
    generation_result = await answer_generation_task(
        grading_result=grading_result,
        query_result=query_result,
        user_id=user_id,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        include_citations=include_citations
    )

    # Step 5: Validation (optional)
    if skip_validation:
        validation_result = ValidationResult(
            is_valid=True,
            success=True,
            elapsed_ms=0
        )
    else:
        validation_result = await validation_task(
            generation_result=generation_result,
            grading_result=grading_result,
            query_result=query_result,
            user_id=user_id,
            check_hallucinations=True,
            check_completeness=True
        )

    # Step 6: Learning Feedback
    learning_result = await learning_feedback_task(
        query_result=query_result,
        retrieval_result=retrieval_result,
        grading_result=grading_result,
        generation_result=generation_result,
        validation_result=validation_result,
        user_id=user_id,
        learning_enabled=learning_enabled
    )

    # Calculate total time and build result
    total_duration = time.time() - flow_start
    total_duration_ms = int(total_duration * 1000)

    # Collect all errors
    all_errors = []
    if query_result.error:
        all_errors.append(f"Query Understanding: {query_result.error}")
    if retrieval_result.error:
        all_errors.append(f"Retrieval: {retrieval_result.error}")
    if grading_result.error:
        all_errors.append(f"Grading: {grading_result.error}")
    if generation_result.error:
        all_errors.append(f"Generation: {generation_result.error}")
    if validation_result.error:
        all_errors.append(f"Validation: {validation_result.error}")
    if learning_result.error:
        all_errors.append(f"Learning: {learning_result.error}")

    # Determine overall success
    overall_success = (
        generation_result.success and
        (skip_validation or validation_result.is_valid)
    )

    # Build timing breakdown
    timing = {
        "query_understanding_ms": query_result.elapsed_ms,
        "hybrid_retrieval_ms": retrieval_result.elapsed_ms,
        "document_grading_ms": grading_result.elapsed_ms,
        "answer_generation_ms": generation_result.elapsed_ms,
        "validation_ms": validation_result.elapsed_ms,
        "learning_feedback_ms": learning_result.elapsed_ms,
        "total_ms": total_duration_ms
    }

    # Create final flow summary artifact
    await create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"""
# Document Q&A Pipeline Complete

## Overview
- **User**: {user_id}
- **Query**: {query}
- **Total Processing Time**: {total_duration:.2f}s
- **Status**: {"Success" if overall_success else "Failed"}

## Results
| Metric | Value |
|--------|-------|
| Expanded Query | {query_result.expanded_query[:50]}... |
| Query Type | {query_result.query_type} |
| Chunks Retrieved | {retrieval_result.chunks_count} |
| Chunks After Grading | {grading_result.graded_count} |
| Chunks Used | {generation_result.chunks_used} |
| Answer Length | {len(generation_result.answer)} chars |
| Confidence | {generation_result.confidence:.2f} |
| Validation Passed | {validation_result.is_valid} |
| Citations | {len(generation_result.citations)} |

## Timing Breakdown
| Step | Time (ms) |
|------|-----------|
| Query Understanding | {query_result.elapsed_ms} |
| Hybrid Retrieval | {retrieval_result.elapsed_ms} |
| Document Grading | {grading_result.elapsed_ms} |
| Answer Generation | {generation_result.elapsed_ms} |
| Validation | {validation_result.elapsed_ms} |
| Learning Feedback | {learning_result.elapsed_ms} |
| **Total** | **{total_duration_ms}** |

## Answer Preview
{generation_result.answer[:500]}{'...' if len(generation_result.answer) > 500 else ''}

{"## Errors" + chr(10) + chr(10).join(f"- {e}" for e in all_errors) if all_errors else ""}
        """,
        description=f"Document Q&A Pipeline summary for user {user_id}"
    )

    return {
        "success": overall_success,
        "query": query,
        "user_id": user_id,
        "answer": generation_result.answer,
        "citations": generation_result.citations,
        "confidence": generation_result.confidence,
        "validation": {
            "is_valid": validation_result.is_valid,
            "hallucination_score": validation_result.hallucination_score,
            "completeness_score": validation_result.completeness_score,
            "issues": validation_result.validation_issues
        },
        "query_understanding": {
            "expanded_query": query_result.expanded_query,
            "query_type": query_result.query_type,
            "extracted_filters": query_result.extracted_filters
        },
        "retrieval": {
            "chunks_retrieved": retrieval_result.chunks_count,
            "chunks_after_grading": grading_result.graded_count,
            "chunks_used": generation_result.chunks_used
        },
        "timing": timing,
        "token_usage": generation_result.token_usage,
        "learning_record_id": learning_result.learning_record_id,
        "errors": all_errors
    }


# ============================================================================
# Public Wrapper Function
# ============================================================================

def run_document_qa_flow(
    query: str,
    user_id: str = "anonymous",
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    include_citations: bool = True,
    skip_validation: bool = False,
    learning_enabled: bool = True,
    temperature: float = 0.3,
    max_context_tokens: int = 4000,
    use_prefect: bool = True
) -> Dict[str, Any]:
    """
    Run the Document Q&A flow and return structured results.

    This is the main entry point for the API to use Prefect-tracked document Q&A.
    Provides fallback to direct pipeline execution if Prefect fails.

    Args:
        query: User's natural language query
        user_id: User identifier for tracking (default: "anonymous")
        filters: Optional filter criteria
        top_k: Number of documents to retrieve
        include_citations: Whether to include source citations
        skip_validation: Skip answer validation step
        learning_enabled: Enable learning feedback recording
        temperature: LLM temperature for generation
        max_context_tokens: Maximum tokens for context window
        use_prefect: If True, run through Prefect flow for tracking

    Returns:
        Dict with answer, citations, confidence, timing, and metadata
    """
    if not query or not query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty"
        }

    if use_prefect:
        try:
            result = asyncio.run(document_qa_flow(
                query=query,
                user_id=user_id,
                filters=filters,
                top_k=top_k,
                include_citations=include_citations,
                skip_validation=skip_validation,
                learning_enabled=learning_enabled,
                temperature=temperature,
                max_context_tokens=max_context_tokens
            ))
            return result
        except Exception as e:
            # Log the Prefect failure but fall through to direct execution
            import logging
            logging.getLogger("document_qa_flow").warning(
                f"Prefect flow failed, falling back to direct pipeline: {e}"
            )

    # Direct pipeline execution (fallback)
    return _run_direct_pipeline(
        query=query,
        user_id=user_id,
        filters=filters,
        top_k=top_k,
        include_citations=include_citations,
        skip_validation=skip_validation,
        learning_enabled=learning_enabled,
        temperature=temperature,
        max_context_tokens=max_context_tokens
    )


async def run_document_qa_flow_async(
    query: str,
    user_id: str = "anonymous",
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    include_citations: bool = True,
    skip_validation: bool = False,
    learning_enabled: bool = True,
    temperature: float = 0.3,
    max_context_tokens: int = 4000,
    use_prefect: bool = True
) -> Dict[str, Any]:
    """
    Async version of run_document_qa_flow.

    Use this when calling from an async context (e.g., FastAPI endpoints).

    Args:
        Same as run_document_qa_flow

    Returns:
        Dict with answer, citations, confidence, timing, and metadata
    """
    if not query or not query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty"
        }

    if use_prefect:
        try:
            result = await document_qa_flow(
                query=query,
                user_id=user_id,
                filters=filters,
                top_k=top_k,
                include_citations=include_citations,
                skip_validation=skip_validation,
                learning_enabled=learning_enabled,
                temperature=temperature,
                max_context_tokens=max_context_tokens
            )
            return result
        except Exception as e:
            import logging
            logging.getLogger("document_qa_flow").warning(
                f"Prefect flow failed, falling back to direct pipeline: {e}"
            )

    # Direct pipeline execution (fallback)
    return await _run_direct_pipeline_async(
        query=query,
        user_id=user_id,
        filters=filters,
        top_k=top_k,
        include_citations=include_citations,
        skip_validation=skip_validation,
        learning_enabled=learning_enabled,
        temperature=temperature,
        max_context_tokens=max_context_tokens
    )


def _run_direct_pipeline(
    query: str,
    user_id: str,
    filters: Optional[Dict[str, Any]],
    top_k: int,
    include_citations: bool,
    skip_validation: bool,
    learning_enabled: bool,
    temperature: float,
    max_context_tokens: int
) -> Dict[str, Any]:
    """
    Run the document pipeline directly without Prefect.

    This is the fallback when Prefect is unavailable or fails.
    """
    try:
        return asyncio.run(_run_direct_pipeline_async(
            query=query,
            user_id=user_id,
            filters=filters,
            top_k=top_k,
            include_citations=include_citations,
            skip_validation=skip_validation,
            learning_enabled=learning_enabled,
            temperature=temperature,
            max_context_tokens=max_context_tokens
        ))
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "user_id": user_id
        }


async def _run_direct_pipeline_async(
    query: str,
    user_id: str,
    filters: Optional[Dict[str, Any]],
    top_k: int,
    include_citations: bool,
    skip_validation: bool,
    learning_enabled: bool,
    temperature: float,
    max_context_tokens: int
) -> Dict[str, Any]:
    """
    Async direct pipeline execution.
    """
    import time
    start_time = time.time()

    try:
        from pipelines.document.document_pipeline import DocumentRetrievalPipeline
        from pipelines.document.context import DocumentPipelineContext

        # Build config
        config = {
            "use_query_understanding": True,
            "use_reranking": True,
            "use_llm_grading": True,
            "use_validation": not skip_validation,
            "learning_enabled": learning_enabled,
            "temperature": temperature,
            "max_context_tokens": max_context_tokens,
        }

        # Create pipeline and context
        pipeline = DocumentRetrievalPipeline(config=config)
        context = DocumentPipelineContext(
            query=query,
            filters=filters or {},
            top_k=top_k,
            include_citations=include_citations
        )

        # Execute pipeline
        result_context = await pipeline.execute(context)

        total_time = time.time() - start_time

        return {
            "success": result_context.status.value == "completed",
            "query": query,
            "user_id": user_id,
            "answer": result_context.answer or "",
            "citations": [
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "relevance_score": c.relevance_score,
                    "excerpt": c.excerpt
                }
                for c in result_context.citations
            ],
            "confidence": result_context.confidence,
            "validation": {
                "is_valid": result_context.is_valid,
                "hallucination_score": result_context.hallucination_score,
                "completeness_score": result_context.completeness_score,
                "issues": result_context.validation_issues
            },
            "query_understanding": {
                "expanded_query": result_context.expanded_query,
                "query_type": result_context.query_type,
                "extracted_filters": result_context.extracted_filters
            },
            "retrieval": {
                "chunks_retrieved": result_context.chunks_retrieved,
                "chunks_after_grading": len(result_context.graded_chunks),
                "chunks_used": result_context.chunks_used
            },
            "timing": {
                **{f"{k}_ms": int(v * 1000) for k, v in result_context.step_timings.items()},
                "total_ms": int(total_time * 1000)
            },
            "learning_record_id": result_context.learning_record_id,
            "errors": [e.get("error", str(e)) for e in result_context.errors],
            "prefect_used": False
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "user_id": user_id,
            "prefect_used": False
        }


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "How do I configure MongoDB?"

    print(f"Running Document Q&A Pipeline for query: '{test_query}'")
    print("-" * 60)

    result = run_document_qa_flow(
        query=test_query,
        user_id="cli_test",
        use_prefect=True
    )

    print(f"\nSuccess: {result.get('success')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
    print(f"Citations: {len(result.get('citations', []))}")
    print(f"Timing: {result.get('timing', {})}")

    if result.get('errors'):
        print(f"Errors: {result.get('errors')}")
