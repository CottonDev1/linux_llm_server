"""
Master Test Orchestrator
========================

Master Prefect flow that runs all pipeline tests.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (ports 8080, 8081, 8082)
- All configuration via Prefect parameters - no hardcoded values

Runs tests for:
1. SQL Pipeline (port 8080)
2. Audio Pipeline (port 8081)
3. Query/RAG Pipeline (port 8081)
4. Git Analysis Pipeline (port 8082)
5. Code Flow Pipeline (port 8082)
6. Code Assistance Pipeline (port 8082)
7. Document Agent Pipeline (port 8081)
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prefect_pipelines.test_flows.base_test_flow import TestFlowConfig
from prefect_pipelines.test_flows.test_flow_utils import (
    TestStatus,
    TestMetrics,
    TestResult,
    ProgressTracker,
    create_test_report_markdown,
    create_test_results_table,
)


# Import individual pipeline test flows
from prefect_pipelines.test_flows.sql_pipeline_test_flow import sql_pipeline_test_flow
from prefect_pipelines.test_flows.audio_pipeline_test_flow import audio_pipeline_test_flow
from prefect_pipelines.test_flows.query_pipeline_test_flow import query_pipeline_test_flow
from prefect_pipelines.test_flows.git_pipeline_test_flow import git_pipeline_test_flow
from prefect_pipelines.test_flows.code_flow_pipeline_test_flow import code_flow_pipeline_test_flow
from prefect_pipelines.test_flows.code_assistance_pipeline_test_flow import code_assistance_pipeline_test_flow
from prefect_pipelines.test_flows.document_pipeline_test_flow import document_pipeline_test_flow


@task(name="run_pipeline_tests", log_prints=True)
def run_pipeline_tests(
    pipeline: str,
    mongodb_uri: str,
    llm_endpoint: str,
    timeout_seconds: int,
    run_storage: bool,
    run_retrieval: bool,
    run_generation: bool,
    run_e2e: bool,
    cleanup: bool,
) -> Dict[str, Any]:
    """
    Run tests for a single pipeline.

    Returns results dict from the pipeline flow.
    """
    logger = get_run_logger()
    logger.info(f"Running {pipeline} pipeline tests...")

    # Map pipeline to flow
    flow_map = {
        "sql": sql_pipeline_test_flow,
        "audio": audio_pipeline_test_flow,
        "query": query_pipeline_test_flow,
        "git": git_pipeline_test_flow,
        "code_flow": code_flow_pipeline_test_flow,
        "code_assistance": code_assistance_pipeline_test_flow,
        "document_agent": document_pipeline_test_flow,
    }

    if pipeline not in flow_map:
        logger.warning(f"No flow defined for pipeline: {pipeline}")
        return {
            "pipeline": pipeline,
            "success": False,
            "error": f"No flow defined for pipeline: {pipeline}",
            "metrics": {"total": 0, "passed": 0, "failed": 0},
            "results": [],
        }

    try:
        result = flow_map[pipeline](
            mongodb_uri=mongodb_uri,
            llm_endpoint=llm_endpoint,
            timeout_seconds=timeout_seconds,
            run_storage=run_storage,
            run_retrieval=run_retrieval,
            run_generation=run_generation,
            run_e2e=run_e2e,
            cleanup_after_test=cleanup,
        )
        return result
    except Exception as e:
        logger.error(f"Pipeline {pipeline} failed: {e}")
        return {
            "pipeline": pipeline,
            "success": False,
            "error": str(e),
            "metrics": {"total": 0, "passed": 0, "failed": 1},
            "results": [],
        }


@task(name="aggregate_results", log_prints=True)
def aggregate_results(pipeline_results: List[Dict[str, Any]]) -> TestMetrics:
    """
    Aggregate metrics from all pipeline results.
    """
    metrics = TestMetrics()
    metrics.start_time = datetime.utcnow().isoformat()

    for pr in pipeline_results:
        pm = pr.get("metrics", {})
        metrics.total += pm.get("total", 0)
        metrics.passed += pm.get("passed", 0)
        metrics.failed += pm.get("failed", 0)
        metrics.skipped += pm.get("skipped", 0)
        metrics.errors += pm.get("errors", 0)
        metrics.duration_ms += pm.get("duration_ms", 0)

        # Track by pipeline
        pipeline_name = pr.get("pipeline", "unknown")
        metrics.pipelines[pipeline_name] = {
            "passed": pm.get("passed", 0),
            "failed": pm.get("failed", 0),
            "total": pm.get("total", 0),
        }

    metrics.end_time = datetime.utcnow().isoformat()
    return metrics


@task(name="create_master_report", log_prints=True)
async def create_master_report(
    metrics: TestMetrics,
    pipeline_results: List[Dict[str, Any]],
):
    """Create comprehensive test report artifact."""
    # Build all results
    all_results = []
    for pr in pipeline_results:
        for r in pr.get("results", []):
            all_results.append(TestResult(
                name=r.get("name", "unknown"),
                status=TestStatus(r.get("status", "failed")),
                duration_ms=r.get("duration_ms", 0),
                error=r.get("error"),
                pipeline=r.get("pipeline", pr.get("pipeline", "unknown")),
                category=r.get("category", "unknown"),
            ))

    report = create_test_report_markdown(
        metrics=metrics,
        results=all_results,
        title="Master Pipeline Test Report",
    )

    await create_markdown_artifact(
        key="master-test-results",
        markdown=report,
        description="Comprehensive test results for all pipelines",
    )


@flow(
    name="all-pipelines-tests",
    description="Run all pipeline tests - manual trigger only",
    log_prints=True,
)
def master_test_orchestrator_flow(
    mongodb_uri: str,
    llm_sql_endpoint: str = "http://localhost:8080",
    llm_general_endpoint: str = "http://localhost:8081",
    llm_code_endpoint: str = "http://localhost:8082",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
    pipelines: List[str] = None,
) -> Dict[str, Any]:
    """
    Run all pipeline tests.

    Args:
        mongodb_uri: MongoDB connection URI (required)
        llm_sql_endpoint: Local SQL LLM endpoint (port 8080)
        llm_general_endpoint: Local General LLM endpoint (port 8081)
        llm_code_endpoint: Local Code LLM endpoint (port 8082)
        timeout_seconds: Timeout per test category
        run_storage: Run storage tests
        run_retrieval: Run retrieval tests
        run_generation: Run generation tests
        run_e2e: Run end-to-end tests
        cleanup_after_test: Clean up test data
        pipelines: Optional list of specific pipelines to run

    Returns:
        Dict with aggregated results and metrics
    """
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("MASTER PIPELINE TEST ORCHESTRATOR")
    logger.info("=" * 60)

    # Validate all endpoints are local
    for name, endpoint in [
        ("SQL", llm_sql_endpoint),
        ("General", llm_general_endpoint),
        ("Code", llm_code_endpoint),
    ]:
        if not endpoint.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                f"Only local LLM endpoints allowed for {name}. Got: {endpoint}"
            )

    # Define pipeline configs (pipeline_name, llm_endpoint)
    pipeline_configs = [
        ("sql", llm_sql_endpoint),
        ("audio", llm_general_endpoint),
        ("query", llm_general_endpoint),
        ("git", llm_code_endpoint),
        ("code_flow", llm_code_endpoint),
        ("code_assistance", llm_code_endpoint),
        ("document_agent", llm_general_endpoint),
    ]

    # Filter to requested pipelines if specified
    if pipelines:
        pipeline_configs = [
            (p, e) for p, e in pipeline_configs if p in pipelines
        ]

    logger.info(f"Running tests for pipelines: {[p for p, _ in pipeline_configs]}")

    # Run each pipeline
    pipeline_results = []
    for pipeline, endpoint in pipeline_configs:
        result = run_pipeline_tests(
            pipeline=pipeline,
            mongodb_uri=mongodb_uri,
            llm_endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            run_storage=run_storage,
            run_retrieval=run_retrieval,
            run_generation=run_generation,
            run_e2e=run_e2e,
            cleanup=cleanup_after_test,
        )
        pipeline_results.append(result)

        status = "PASSED" if result.get("success") else "FAILED"
        logger.info(f"  {pipeline}: {status}")

    # Aggregate results
    metrics = aggregate_results(pipeline_results)

    # Create report
    import asyncio
    asyncio.run(create_master_report(metrics, pipeline_results))

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {metrics.total}")
    logger.info(f"Passed: {metrics.passed}")
    logger.info(f"Failed: {metrics.failed}")
    logger.info(f"Success Rate: {metrics.success_rate:.1f}%")
    logger.info(f"Duration: {metrics.duration_ms}ms")

    success = metrics.failed == 0 and metrics.errors == 0

    return {
        "success": success,
        "metrics": metrics.to_dict(),
        "pipeline_results": pipeline_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run All Pipeline Tests")
    parser.add_argument("--mongodb-uri", required=True, help="MongoDB URI")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--pipelines", nargs="+", help="Specific pipelines to run")

    args = parser.parse_args()

    result = master_test_orchestrator_flow(
        mongodb_uri=args.mongodb_uri,
        timeout_seconds=args.timeout,
        pipelines=args.pipelines,
    )

    print(f"\nFinal Result: {'SUCCESS' if result['success'] else 'FAILED'}")
