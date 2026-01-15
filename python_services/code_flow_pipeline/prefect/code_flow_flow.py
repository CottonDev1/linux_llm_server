"""
Prefect Code Flow Analysis Pipeline
====================================

Orchestrates code flow analysis workflows with:
1. Query Classification - Categorize incoming questions
2. Multi-Stage Retrieval - Search across code collections
3. Call Chain Building - Trace execution paths
4. LLM Synthesis - Generate coherent answers

Features:
- Built-in retries with exponential backoff
- Detailed logging and artifact tracking
- Async-native execution
- Visual progress in Prefect dashboard
- Batch processing for multiple queries

Design Rationale:
-----------------
Prefect flows are useful for:
1. Batch processing of code flow queries (e.g., documentation generation)
2. Scheduled analysis of new code contexts
3. Monitoring and alerting on analysis quality
4. Reproducible, trackable analysis runs

The flow structure mirrors the CodeFlowPipeline but adds:
- Task-level retries for resilience
- Artifact generation for dashboards
- Parameter validation and logging
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class ClassificationResult:
    """Result from query classification task."""
    query: str
    query_type: str = "general"
    confidence: float = 0.0
    matched_patterns: List[str] = field(default_factory=list)
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from multi-stage retrieval task."""
    query: str
    project: Optional[str] = None
    total_results: int = 0
    methods_count: int = 0
    classes_count: int = 0
    ui_events_count: int = 0
    business_processes_count: int = 0
    call_relationships_count: int = 0
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class CallChainResult:
    """Result from call chain building task."""
    query: str
    chains_built: int = 0
    total_methods_traced: int = 0
    database_touching_chains: int = 0
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class SynthesisResult:
    """Result from LLM synthesis task."""
    query: str
    answer_length: int = 0
    tokens_used: int = 0
    model_used: str = ""
    is_fallback: bool = False
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class CodeFlowAnalysisResult:
    """Complete result from code flow analysis."""
    query: str
    project: Optional[str]
    classification: ClassificationResult
    retrieval: RetrievalResult
    call_chains: CallChainResult
    synthesis: SynthesisResult
    total_duration_ms: int = 0
    success: bool = True
    answer: str = ""
    sources_count: int = 0


@task(
    name="classify_query",
    description="Classify the query type to determine retrieval strategy",
    retries=2,
    retry_delay_seconds=5,
    tags=["classification"]
)
async def classify_query_task(query: str) -> ClassificationResult:
    """
    Classify a code flow query.

    Args:
        query: Natural language question

    Returns:
        ClassificationResult with type and confidence
    """
    logger = get_run_logger()
    start_time = time.time()

    try:
        from code_flow_pipeline.services.query_classifier import get_query_classifier

        classifier = get_query_classifier()
        classification = classifier.classify(query)

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Query classified as '{classification.type.value}' "
            f"(confidence={classification.confidence:.2f})"
        )

        return ClassificationResult(
            query=query,
            query_type=classification.type.value,
            confidence=classification.confidence,
            matched_patterns=classification.matched_patterns,
            duration_ms=duration_ms,
            success=True,
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return ClassificationResult(
            query=query,
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


@task(
    name="retrieve_context",
    description="Execute multi-stage retrieval across code collections",
    retries=2,
    retry_delay_seconds=10,
    tags=["retrieval", "mongodb"]
)
async def retrieve_context_task(
    query: str,
    project: Optional[str],
    classification: ClassificationResult,
    include_call_graph: bool = True,
) -> RetrievalResult:
    """
    Execute multi-stage retrieval for a query.

    Args:
        query: Natural language question
        project: Project scope
        classification: Query classification result
        include_call_graph: Whether to include call graph stage

    Returns:
        RetrievalResult with document counts
    """
    logger = get_run_logger()
    start_time = time.time()

    try:
        from code_flow_pipeline.services.multi_stage_retrieval import get_multi_stage_retrieval
        from code_flow_pipeline.models.query_models import QueryClassification, QueryType

        retrieval = await get_multi_stage_retrieval()

        # Reconstruct classification object
        query_classification = QueryClassification(
            type=QueryType(classification.query_type),
            confidence=classification.confidence,
            matched_patterns=classification.matched_patterns,
        )

        results = await retrieval.execute(
            query=query,
            project=project,
            classification=query_classification,
            include_call_graph=include_call_graph,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Retrieved {results.total_results} total results "
            f"(methods={len(results.methods)}, classes={len(results.classes)})"
        )

        return RetrievalResult(
            query=query,
            project=project,
            total_results=results.total_results,
            methods_count=len(results.methods),
            classes_count=len(results.classes),
            ui_events_count=len(results.ui_events),
            business_processes_count=len(results.business_processes),
            call_relationships_count=len(results.call_relationships),
            duration_ms=duration_ms,
            success=True,
        )

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return RetrievalResult(
            query=query,
            project=project,
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


@task(
    name="build_call_chains",
    description="Build execution paths from methods and UI events",
    retries=2,
    retry_delay_seconds=10,
    tags=["call-chain", "graph"]
)
async def build_call_chains_task(
    query: str,
    project: Optional[str],
    max_hops: int = 5,
) -> CallChainResult:
    """
    Build call chains from retrieved methods.

    Args:
        query: Original query (for method hints)
        project: Project scope
        max_hops: Maximum chain depth

    Returns:
        CallChainResult with chain counts
    """
    logger = get_run_logger()
    start_time = time.time()

    try:
        from code_flow_pipeline.services.call_chain_builder import get_call_chain_builder
        from code_flow_pipeline.services.multi_stage_retrieval import get_multi_stage_retrieval

        # Get retrieval service to access cached results
        retrieval = await get_multi_stage_retrieval()
        builder = await get_call_chain_builder()

        # Execute retrieval to get methods and UI events
        results = await retrieval.execute(
            query=query,
            project=project,
            include_call_graph=True,
        )

        # Build chains
        chains = await builder.build_chains_from_methods(
            methods=results.methods,
            ui_events=results.ui_events,
            project=project,
            max_hops=max_hops,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Count database-touching chains
        db_chains = sum(1 for c in chains if c.touches_database)
        total_methods = sum(len(c.steps) for c in chains)

        logger.info(
            f"Built {len(chains)} call chains "
            f"({db_chains} touch database, {total_methods} total methods)"
        )

        return CallChainResult(
            query=query,
            chains_built=len(chains),
            total_methods_traced=total_methods,
            database_touching_chains=db_chains,
            duration_ms=duration_ms,
            success=True,
        )

    except Exception as e:
        logger.error(f"Call chain building failed: {e}")
        return CallChainResult(
            query=query,
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


@task(
    name="synthesize_answer",
    description="Generate coherent answer using LLM",
    retries=3,
    retry_delay_seconds=15,
    tags=["llm", "synthesis"]
)
async def synthesize_answer_task(
    query: str,
    project: Optional[str],
) -> SynthesisResult:
    """
    Synthesize an answer using the full pipeline.

    Args:
        query: Natural language question
        project: Project scope

    Returns:
        SynthesisResult with answer metadata
    """
    logger = get_run_logger()
    start_time = time.time()

    try:
        from code_flow_pipeline import get_code_flow_pipeline, CodeFlowRequest

        pipeline = await get_code_flow_pipeline()

        request = CodeFlowRequest(
            query=query,
            project=project,
            include_call_graph=True,
            max_hops=5,
        )

        response = await pipeline.analyze(request)

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Synthesized answer ({len(response.answer)} chars) "
            f"in {duration_ms}ms"
        )

        return SynthesisResult(
            query=query,
            answer_length=len(response.answer),
            tokens_used=0,  # Would need to track from LLM service
            model_used="llama.cpp",
            is_fallback=response.metadata.get("fallback", False),
            duration_ms=duration_ms,
            success=True,
        )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return SynthesisResult(
            query=query,
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


@flow(
    name="code-flow-analysis",
    description="Analyze code flow for a natural language question",
    version="1.0.0",
)
async def code_flow_analysis_flow(
    query: str,
    project: Optional[str] = None,
    include_call_graph: bool = True,
    max_hops: int = 5,
) -> CodeFlowAnalysisResult:
    """
    Complete code flow analysis workflow.

    Orchestrates:
    1. Query classification
    2. Multi-stage retrieval
    3. Call chain building
    4. LLM synthesis

    Args:
        query: Natural language question
        project: Project scope (e.g., 'gin', 'warehouse')
        include_call_graph: Whether to build call chains
        max_hops: Maximum call chain depth

    Returns:
        CodeFlowAnalysisResult with complete analysis
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Starting code flow analysis: '{query[:50]}...'")

    # Step 1: Classify query
    classification = await classify_query_task(query)

    # Step 2: Retrieve context
    retrieval = await retrieve_context_task(
        query=query,
        project=project,
        classification=classification,
        include_call_graph=include_call_graph,
    )

    # Step 3: Build call chains
    call_chains = await build_call_chains_task(
        query=query,
        project=project,
        max_hops=max_hops,
    )

    # Step 4: Synthesize answer
    synthesis = await synthesize_answer_task(
        query=query,
        project=project,
    )

    total_duration_ms = int((time.time() - start_time) * 1000)

    # Determine overall success
    success = all([
        classification.success,
        retrieval.success,
        call_chains.success,
        synthesis.success,
    ])

    result = CodeFlowAnalysisResult(
        query=query,
        project=project,
        classification=classification,
        retrieval=retrieval,
        call_chains=call_chains,
        synthesis=synthesis,
        total_duration_ms=total_duration_ms,
        success=success,
        sources_count=retrieval.total_results,
    )

    # Create summary artifact
    await create_markdown_artifact(
        key="code-flow-analysis-summary",
        markdown=_create_summary_markdown(result),
        description="Code Flow Analysis Summary",
    )

    logger.info(
        f"Code flow analysis complete "
        f"(success={success}, duration={total_duration_ms}ms)"
    )

    return result


@flow(
    name="batch-code-flow-analysis",
    description="Batch analyze multiple code flow questions",
    version="1.0.0",
)
async def batch_code_flow_analysis_flow(
    queries: List[str],
    project: Optional[str] = None,
    max_concurrent: int = 3,
) -> List[CodeFlowAnalysisResult]:
    """
    Batch analyze multiple code flow questions.

    Useful for:
    - Generating documentation for a feature
    - Testing query coverage
    - Building FAQ answers

    Args:
        queries: List of questions to analyze
        project: Project scope
        max_concurrent: Maximum concurrent analyses

    Returns:
        List of CodeFlowAnalysisResult
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Starting batch analysis of {len(queries)} queries")

    results = []

    # Process in batches to avoid overwhelming services
    for i in range(0, len(queries), max_concurrent):
        batch = queries[i:i + max_concurrent]
        batch_tasks = [
            code_flow_analysis_flow(
                query=q,
                project=project,
            )
            for q in batch
        ]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for query, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Query failed: {query[:50]}... - {result}")
                # Create error result
                results.append(CodeFlowAnalysisResult(
                    query=query,
                    project=project,
                    classification=ClassificationResult(query=query, success=False),
                    retrieval=RetrievalResult(query=query, success=False),
                    call_chains=CallChainResult(query=query, success=False),
                    synthesis=SynthesisResult(query=query, success=False, error=str(result)),
                    success=False,
                ))
            else:
                results.append(result)

    total_duration_ms = int((time.time() - start_time) * 1000)
    success_count = sum(1 for r in results if r.success)

    # Create batch summary artifact
    await create_markdown_artifact(
        key="batch-code-flow-summary",
        markdown=_create_batch_summary_markdown(results, total_duration_ms),
        description="Batch Code Flow Analysis Summary",
    )

    logger.info(
        f"Batch analysis complete "
        f"({success_count}/{len(queries)} successful, {total_duration_ms}ms)"
    )

    return results


def _create_summary_markdown(result: CodeFlowAnalysisResult) -> str:
    """Create markdown summary for a single analysis."""
    status = "Success" if result.success else "Failed"

    md = f"""# Code Flow Analysis Summary

## Query
> {result.query}

## Status: {status}

## Metrics

| Stage | Duration | Status |
|-------|----------|--------|
| Classification | {result.classification.duration_ms}ms | {"OK" if result.classification.success else "FAILED"} |
| Retrieval | {result.retrieval.duration_ms}ms | {"OK" if result.retrieval.success else "FAILED"} |
| Call Chains | {result.call_chains.duration_ms}ms | {"OK" if result.call_chains.success else "FAILED"} |
| Synthesis | {result.synthesis.duration_ms}ms | {"OK" if result.synthesis.success else "FAILED"} |
| **Total** | **{result.total_duration_ms}ms** | **{status}** |

## Classification
- Type: `{result.classification.query_type}`
- Confidence: {result.classification.confidence:.2%}

## Retrieval Results
- Methods: {result.retrieval.methods_count}
- Classes: {result.retrieval.classes_count}
- UI Events: {result.retrieval.ui_events_count}
- Business Processes: {result.retrieval.business_processes_count}
- **Total: {result.retrieval.total_results}**

## Call Chains
- Chains Built: {result.call_chains.chains_built}
- Methods Traced: {result.call_chains.total_methods_traced}
- Database-Touching: {result.call_chains.database_touching_chains}
"""

    return md


def _create_batch_summary_markdown(
    results: List[CodeFlowAnalysisResult],
    total_duration_ms: int,
) -> str:
    """Create markdown summary for batch analysis."""
    success_count = sum(1 for r in results if r.success)
    avg_duration = sum(r.total_duration_ms for r in results) / len(results) if results else 0

    md = f"""# Batch Code Flow Analysis Summary

## Overview
- **Total Queries**: {len(results)}
- **Successful**: {success_count}
- **Failed**: {len(results) - success_count}
- **Total Duration**: {total_duration_ms}ms
- **Average per Query**: {avg_duration:.0f}ms

## Results by Query

| Query | Type | Sources | Chains | Duration | Status |
|-------|------|---------|--------|----------|--------|
"""

    for r in results:
        query_short = r.query[:40] + "..." if len(r.query) > 40 else r.query
        status = "OK" if r.success else "FAILED"
        md += f"| {query_short} | {r.classification.query_type} | {r.retrieval.total_results} | {r.call_chains.chains_built} | {r.total_duration_ms}ms | {status} |\n"

    return md


if __name__ == "__main__":
    # Example usage
    asyncio.run(
        code_flow_analysis_flow(
            query="How are bales committed to purchase subcontract?",
            project="gin",
        )
    )
