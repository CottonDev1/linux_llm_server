"""
Prefect Query Pipeline Flow
===========================

Orchestrates batch RAG query processing with:
1. Query processing with caching
2. Performance metrics collection
3. Error tracking and logging
4. Artifact generation for monitoring

Use Cases:
- Batch processing of multiple queries
- Performance benchmarking
- Cache warming with common queries
- Integration testing

Features:
- Built-in retries with exponential backoff
- Detailed logging and artifact tracking
- Async-native execution
- Visual progress in Prefect dashboard
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class QueryResult:
    """Result from a single query task."""
    query: str
    project: Optional[str] = None
    success: bool = True
    answer: str = ""
    sources_count: int = 0
    cached: bool = False
    search_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Result from batch query processing."""
    total_queries: int = 0
    successful: int = 0
    failed: int = 0
    cached: int = 0
    avg_response_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@task(
    name="process_query",
    description="Process a single RAG query",
    retries=2,
    retry_delay_seconds=5,
    tags=["query", "rag"]
)
async def process_query_task(
    query: str,
    project: Optional[str] = None,
    limit: int = 10,
    include_ewr_library: bool = False
) -> QueryResult:
    """
    Process a single RAG query.

    Args:
        query: Natural language query
        project: Project scope (optional)
        limit: Maximum search results
        include_ewr_library: Include EWRLibrary in search

    Returns:
        QueryResult with response and timing
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Processing query: '{query[:50]}...' project={project}")

    result = QueryResult(query=query, project=project)

    try:
        # Import here to avoid circular dependencies
        import sys
        sys.path.insert(0, '..')
        from query_pipeline.pipeline import get_query_pipeline
        from query_pipeline.models.query_models import QueryRequest

        pipeline = await get_query_pipeline()

        request = QueryRequest(
            query=query,
            project=project,
            limit=limit,
            include_ewr_library=include_ewr_library
        )

        response = await pipeline.query(request)

        result.success = True
        result.answer = response.answer
        result.sources_count = len(response.sources)
        result.cached = response.cached

        if response.timing:
            result.search_time = response.timing.search
            result.llm_time = response.timing.llm
            result.total_time = response.timing.total

        logger.info(
            f"Query completed: {result.sources_count} sources, "
            f"{len(result.answer)} chars, cached={result.cached}"
        )

    except Exception as e:
        error_msg = f"Query failed: {str(e)}"
        logger.error(error_msg)
        result.success = False
        result.error = error_msg

    result.total_time = time.time() - start_time
    return result


@task(
    name="warm_cache",
    description="Warm cache with common queries",
    retries=1,
    retry_delay_seconds=10,
    tags=["cache", "warmup"]
)
async def warm_cache_task(
    queries: List[Dict[str, Any]]
) -> BatchResult:
    """
    Warm the cache with a list of common queries.

    Args:
        queries: List of query dicts with 'query', 'project' keys

    Returns:
        BatchResult with success/failure counts
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Warming cache with {len(queries)} queries")

    result = BatchResult(total_queries=len(queries))
    query_times = []

    for q in queries:
        query_result = await process_query_task(
            query=q.get("query", ""),
            project=q.get("project"),
            limit=q.get("limit", 10),
            include_ewr_library=q.get("include_ewr_library", False)
        )

        if query_result.success:
            result.successful += 1
            if query_result.cached:
                result.cached += 1
        else:
            result.failed += 1
            if query_result.error:
                result.errors.append(query_result.error)

        query_times.append(query_result.total_time)

    result.duration_seconds = time.time() - start_time
    result.avg_response_time = sum(query_times) / len(query_times) if query_times else 0

    logger.info(
        f"Cache warmup complete: {result.successful}/{result.total_queries} "
        f"successful, {result.cached} cached, avg time: {result.avg_response_time:.2f}s"
    )

    return result


@task(
    name="benchmark_queries",
    description="Benchmark query performance",
    tags=["benchmark", "performance"]
)
async def benchmark_queries_task(
    queries: List[Dict[str, Any]],
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark query performance with multiple iterations.

    Args:
        queries: List of test queries
        iterations: Number of iterations per query

    Returns:
        Benchmark results with timing statistics
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Benchmarking {len(queries)} queries, {iterations} iterations each")

    results = {
        "queries": [],
        "summary": {},
        "duration_seconds": 0
    }

    for q in queries:
        query_text = q.get("query", "")
        project = q.get("project")

        times = []
        for i in range(iterations):
            query_result = await process_query_task(
                query=query_text,
                project=project,
                limit=q.get("limit", 10)
            )
            times.append(query_result.total_time)

        results["queries"].append({
            "query": query_text[:50],
            "project": project,
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "iterations": iterations
        })

    # Calculate summary
    all_avg_times = [q["avg_time"] for q in results["queries"]]
    results["summary"] = {
        "total_queries": len(queries),
        "iterations_per_query": iterations,
        "overall_avg_time": sum(all_avg_times) / len(all_avg_times) if all_avg_times else 0,
        "fastest_query": min(all_avg_times) if all_avg_times else 0,
        "slowest_query": max(all_avg_times) if all_avg_times else 0
    }

    results["duration_seconds"] = time.time() - start_time

    # Create artifact
    await create_markdown_artifact(
        key="benchmark-results",
        markdown=f"""
## Query Benchmark Results

**Run at**: {datetime.now().isoformat()}
**Duration**: {results['duration_seconds']:.2f}s

### Summary
- Total Queries: {results['summary']['total_queries']}
- Iterations per Query: {results['summary']['iterations_per_query']}
- Overall Average Time: {results['summary']['overall_avg_time']:.3f}s
- Fastest Query: {results['summary']['fastest_query']:.3f}s
- Slowest Query: {results['summary']['slowest_query']:.3f}s

### Individual Results
| Query | Project | Min | Avg | Max |
|-------|---------|-----|-----|-----|
""" + "\n".join([
            f"| {q['query']} | {q['project'] or 'all'} | {q['min_time']:.3f}s | {q['avg_time']:.3f}s | {q['max_time']:.3f}s |"
            for q in results["queries"]
        ]),
        description="Query performance benchmark results"
    )

    return results


@flow(
    name="query-pipeline-batch",
    description="Batch process multiple RAG queries",
    retries=1,
    retry_delay_seconds=60
)
async def query_batch_flow(
    queries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Batch process multiple RAG queries.

    This flow processes a list of queries and returns
    aggregated results with timing and success metrics.

    Args:
        queries: List of query dicts with keys:
            - query: The query text
            - project: Optional project scope
            - limit: Optional result limit

    Returns:
        Dict with batch results and individual query results
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting batch query flow with {len(queries)} queries")

    results = []
    for q in queries:
        result = await process_query_task(
            query=q.get("query", ""),
            project=q.get("project"),
            limit=q.get("limit", 10),
            include_ewr_library=q.get("include_ewr_library", False)
        )
        results.append(result)

    # Calculate summary
    successful = sum(1 for r in results if r.success)
    cached = sum(1 for r in results if r.cached)
    total_time = time.time() - flow_start

    summary = {
        "total_queries": len(queries),
        "successful": successful,
        "failed": len(queries) - successful,
        "cached": cached,
        "total_duration_seconds": total_time,
        "avg_query_time": sum(r.total_time for r in results) / len(results) if results else 0
    }

    # Create summary artifact
    await create_markdown_artifact(
        key="batch-query-summary",
        markdown=f"""
## Batch Query Results

**Completed**: {datetime.now().isoformat()}
**Duration**: {total_time:.2f}s

### Summary
- Total Queries: {summary['total_queries']}
- Successful: {summary['successful']}
- Failed: {summary['failed']}
- Cache Hits: {summary['cached']}
- Avg Query Time: {summary['avg_query_time']:.2f}s

### Individual Results
| Query | Project | Status | Time |
|-------|---------|--------|------|
""" + "\n".join([
            f"| {r.query[:40]}... | {r.project or 'all'} | {'OK' if r.success else 'FAIL'} | {r.total_time:.2f}s |"
            for r in results
        ]),
        description="Batch query processing summary"
    )

    return {
        "summary": summary,
        "results": [
            {
                "query": r.query,
                "project": r.project,
                "success": r.success,
                "cached": r.cached,
                "sources_count": r.sources_count,
                "total_time": r.total_time,
                "error": r.error
            }
            for r in results
        ]
    }


@flow(
    name="cache-warmup-flow",
    description="Warm query cache with common queries"
)
async def cache_warmup_flow(
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Warm the query cache with common queries.

    If no queries provided, uses a default set of common queries.

    Args:
        queries: Optional list of queries to cache

    Returns:
        Warmup results with success metrics
    """
    logger = get_run_logger()

    # Default common queries if none provided
    if queries is None:
        queries = [
            {"query": "How does the bale weight calculation work?", "project": "gin"},
            {"query": "What is RecapGet?", "project": "gin"},
            {"query": "How do I configure the warehouse scanner?", "project": "warehouse"},
            {"query": "What are the work instructions for module processing?", "project": "knowledge_base"},
        ]

    logger.info(f"Starting cache warmup with {len(queries)} queries")

    result = await warm_cache_task(queries)

    # Create artifact
    await create_markdown_artifact(
        key="cache-warmup-summary",
        markdown=f"""
## Cache Warmup Results

**Completed**: {datetime.now().isoformat()}
**Duration**: {result.duration_seconds:.2f}s

### Summary
- Total Queries: {result.total_queries}
- Successful: {result.successful}
- Failed: {result.failed}
- Already Cached: {result.cached}
- Avg Response Time: {result.avg_response_time:.2f}s

{"### Errors" + chr(10) + chr(10).join(f"- {e}" for e in result.errors) if result.errors else ""}
        """,
        description="Cache warmup summary"
    )

    return {
        "success": result.failed == 0,
        "total_queries": result.total_queries,
        "successful": result.successful,
        "failed": result.failed,
        "cached": result.cached,
        "avg_response_time": result.avg_response_time,
        "duration_seconds": result.duration_seconds,
        "errors": result.errors
    }


def run_batch_queries(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to run batch queries synchronously.

    Example:
        from query_pipeline.prefect.query_flow import run_batch_queries

        result = run_batch_queries([
            {"query": "How does RecapGet work?", "project": "gin"},
            {"query": "What is the warehouse layout?", "project": "warehouse"}
        ])
    """
    return asyncio.run(query_batch_flow(queries))


def run_cache_warmup(queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Convenience function to run cache warmup synchronously.

    Example:
        from query_pipeline.prefect.query_flow import run_cache_warmup

        result = run_cache_warmup()  # Uses default queries
    """
    return asyncio.run(cache_warmup_flow(queries))


if __name__ == "__main__":
    # Test run
    print("Testing batch query flow...")
    result = run_batch_queries([
        {"query": "What is the RecapGet procedure?", "project": "gin"},
        {"query": "How do I configure the scanner?", "project": "warehouse"}
    ])
    print(f"Result: {result}")
