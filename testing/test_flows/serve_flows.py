"""
Serve Prefect Flows for API Testing

This script serves all test flows locally, making them available
for execution through the Prefect UI at http://localhost:4200

Usage:
    python -m testing.prefect.serve_flows

The flows will be available in the Prefect UI under Deployments.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prefect import serve

from testing.test_flows.flows.pipeline_flows import (
    sql_pipeline_test_flow,
    audio_pipeline_test_flow,
    query_pipeline_test_flow,
    git_pipeline_test_flow,
    code_flow_pipeline_test_flow,
    code_assistance_pipeline_test_flow,
    document_pipeline_test_flow,
    all_pipelines_test_flow,
    shared_test_flow,
)
from testing.test_flows.flows.combined_flow import (
    api_test_flow,
    full_test_flow,
    quick_smoke_test_flow,
)
from testing.test_flows.flows.granular_test_flow import (
    run_specific_tests,
    run_tests_by_marker,
    run_tests_by_pattern,
)


def main():
    """Serve all test flows."""
    print("=" * 60)
    print("Serving Prefect Test Flows")
    print("=" * 60)
    print("\nAvailable flows:")
    print("  - api-tests: Run all API tests")
    print("  - api-tests-smoke: Quick smoke test (SQL + Query)")
    print("  - sql-pipeline-tests: SQL pipeline tests")
    print("  - audio-pipeline-tests: Audio pipeline tests")
    print("  - query-pipeline-tests: Query/RAG pipeline tests")
    print("  - git-pipeline-tests: Git pipeline tests")
    print("  - code-flow-pipeline-tests: Code flow pipeline tests")
    print("  - code-assistance-pipeline-tests: Code assistance tests")
    print("  - document-pipeline-tests: Document pipeline tests")
    print("  - all-pipeline-tests: All pipeline tests")
    print("  - shared-tests: Shared/common tests (embeddings, streaming, vector search)")
    print("  - full-test-suite: Complete test suite")
    print("\nGranular Test Flows:")
    print("  - run-specific-tests: Run tests with path/pattern/marker filters")
    print("  - run-tests-by-marker: Run tests by pytest marker")
    print("  - run-tests-by-pattern: Run tests by name pattern (-k)")
    print("\nPrefect UI: http://localhost:4200")
    print("=" * 60)

    # Create deployments for serve
    deployments = [
        api_test_flow.to_deployment(
            name="api-tests",
            tags=["testing", "api"],
        ),
        quick_smoke_test_flow.to_deployment(
            name="api-tests-smoke",
            tags=["testing", "api", "smoke"],
        ),
        sql_pipeline_test_flow.to_deployment(
            name="sql-tests",
            tags=["testing", "pipeline", "sql"],
        ),
        audio_pipeline_test_flow.to_deployment(
            name="audio-tests",
            tags=["testing", "pipeline", "audio"],
        ),
        query_pipeline_test_flow.to_deployment(
            name="query-tests",
            tags=["testing", "pipeline", "query"],
        ),
        git_pipeline_test_flow.to_deployment(
            name="git-tests",
            tags=["testing", "pipeline", "git"],
        ),
        code_flow_pipeline_test_flow.to_deployment(
            name="code-flow-tests",
            tags=["testing", "pipeline", "code"],
        ),
        code_assistance_pipeline_test_flow.to_deployment(
            name="code-assistance-tests",
            tags=["testing", "pipeline", "code"],
        ),
        document_pipeline_test_flow.to_deployment(
            name="document-tests",
            tags=["testing", "pipeline", "document"],
        ),
        shared_test_flow.to_deployment(
            name="shared-tests",
            tags=["testing", "pipeline", "shared"],
        ),
        all_pipelines_test_flow.to_deployment(
            name="all-pipeline-tests",
            tags=["testing", "pipeline", "all"],
        ),
        full_test_flow.to_deployment(
            name="full-test-suite",
            tags=["testing", "full"],
        ),
        # Granular test flows
        run_specific_tests.to_deployment(
            name="run-specific-tests",
            tags=["testing", "granular"],
        ),
        run_tests_by_marker.to_deployment(
            name="run-tests-by-marker",
            tags=["testing", "granular", "marker"],
        ),
        run_tests_by_pattern.to_deployment(
            name="run-tests-by-pattern",
            tags=["testing", "granular", "pattern"],
        ),
    ]

    # Serve all deployments
    serve(*deployments)


if __name__ == "__main__":
    main()
