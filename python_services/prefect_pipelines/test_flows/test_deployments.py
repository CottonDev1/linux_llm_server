"""
Test Deployments Registration
=============================

Register all pipeline test flows as Prefect deployments.

IMPORTANT:
- NO SCHEDULES - all deployments are manual trigger only
- All configuration via Prefect parameters at runtime
- Local LLM endpoints only (llama.cpp ports 8080, 8081, 8082)

Usage:
    # Deploy all test flows
    python -m prefect_pipelines.test_flows.test_deployments

    # Or run the flows via CLI
    prefect deployment run 'sql-pipeline-tests/sql-pipeline-tests' \\
        --param mongodb_uri="mongodb://EWRSPT-AI:27018"

    # Or via Prefect UI (localhost:4200)
"""

import os
import sys

from prefect import serve
from prefect.runner.storage import GitRepository

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import all test flows
from prefect_pipelines.test_flows.sql_pipeline_test_flow import sql_pipeline_test_flow
from prefect_pipelines.test_flows.audio_pipeline_test_flow import audio_pipeline_test_flow
from prefect_pipelines.test_flows.query_pipeline_test_flow import query_pipeline_test_flow
from prefect_pipelines.test_flows.git_pipeline_test_flow import git_pipeline_test_flow
from prefect_pipelines.test_flows.code_flow_pipeline_test_flow import code_flow_pipeline_test_flow
from prefect_pipelines.test_flows.code_assistance_pipeline_test_flow import code_assistance_pipeline_test_flow
from prefect_pipelines.test_flows.document_pipeline_test_flow import document_pipeline_test_flow
from prefect_pipelines.test_flows.master_test_orchestrator import master_test_orchestrator_flow


def deploy_test_flows():
    """
    Deploy all test flows to Prefect server.

    CRITICAL: No schedules - manual trigger only.
    This prevents SQLite concurrency issues and ensures controlled test execution.
    """
    print("=" * 60)
    print("DEPLOYING PIPELINE TEST FLOWS")
    print("=" * 60)
    print("IMPORTANT: All flows are MANUAL TRIGGER ONLY")
    print("No automatic scheduling configured")
    print("=" * 60)

    # Create deployments (no schedule = manual only)
    deployments = [
        sql_pipeline_test_flow.to_deployment(
            name="sql-pipeline-tests",
            description="SQL query generation pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "sql", "manual"],
            parameters={
                "mongodb_uri": "",  # Required - must provide at runtime
                "llm_endpoint": "http://localhost:8080",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
            # NO schedule parameter - manual only
        ),
        audio_pipeline_test_flow.to_deployment(
            name="audio-pipeline-tests",
            description="Audio transcription pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "audio", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_endpoint": "http://localhost:8081",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
        ),
        query_pipeline_test_flow.to_deployment(
            name="query-pipeline-tests",
            description="Query/RAG pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "query", "rag", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_endpoint": "http://localhost:8081",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
        ),
        git_pipeline_test_flow.to_deployment(
            name="git-pipeline-tests",
            description="Git analysis pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "git", "code", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_endpoint": "http://localhost:8082",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
        ),
        code_flow_pipeline_test_flow.to_deployment(
            name="code-flow-pipeline-tests",
            description="Code flow analysis pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "code-flow", "code", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_endpoint": "http://localhost:8082",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
        ),
        code_assistance_pipeline_test_flow.to_deployment(
            name="code-assistance-pipeline-tests",
            description="Code assistance pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "code-assistance", "code", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_endpoint": "http://localhost:8082",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
        ),
        document_pipeline_test_flow.to_deployment(
            name="document-pipeline-tests",
            description="Document agent pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "document", "rag", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_endpoint": "http://localhost:8081",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
            },
        ),
        master_test_orchestrator_flow.to_deployment(
            name="all-pipelines-tests",
            description="Run all pipeline tests - MANUAL TRIGGER ONLY",
            tags=["testing", "master", "all-pipelines", "manual"],
            parameters={
                "mongodb_uri": "",
                "llm_sql_endpoint": "http://localhost:8080",
                "llm_general_endpoint": "http://localhost:8081",
                "llm_code_endpoint": "http://localhost:8082",
                "timeout_seconds": 300,
                "run_storage": True,
                "run_retrieval": True,
                "run_generation": True,
                "run_e2e": True,
                "cleanup_after_test": True,
                "pipelines": None,  # Run all by default
            },
        ),
    ]

    print(f"\nDeploying {len(deployments)} test flows:")
    for d in deployments:
        print(f"  - {d.name}")

    # Serve all deployments
    serve(*deployments)


def list_deployments():
    """List all available test deployments."""
    print("\nAvailable Test Deployments:")
    print("-" * 40)

    flows = [
        ("sql-pipeline-tests", "SQL query generation (port 8080)"),
        ("audio-pipeline-tests", "Audio transcription (port 8081)"),
        ("query-pipeline-tests", "Query/RAG retrieval (port 8081)"),
        ("git-pipeline-tests", "Git analysis (port 8082)"),
        ("code-flow-pipeline-tests", "Code flow analysis (port 8082)"),
        ("code-assistance-pipeline-tests", "Code assistance (port 8082)"),
        ("document-pipeline-tests", "Document processing (port 8081)"),
        ("all-pipelines-tests", "All pipelines (master)"),
    ]

    for name, desc in flows:
        print(f"\n  {name}")
        print(f"    {desc}")
        print(f"    Trigger: prefect deployment run '{name}/{name}' --param mongodb_uri='...'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Pipeline Test Flows")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available deployments",
    )

    args = parser.parse_args()

    if args.list:
        list_deployments()
    else:
        deploy_test_flows()
