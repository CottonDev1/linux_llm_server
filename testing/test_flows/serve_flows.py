"""
Serve Prefect Flows for Testing

This script serves all test flows locally, making them available
for execution through the Prefect UI at http://localhost:4200

Usage:
    python -m testing.test_flows.serve_flows

Available Flows (4 total):
    - pipeline-tests: Run any/all pipeline tests (parameterized)
    - full-test-suite: Complete test suite
    - smoke-test: Quick validation (SQL + Query)
    - custom-tests: Fine-grained pytest control
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prefect import serve

from testing.test_flows.flows import (
    pipeline_test_flow,
    full_test_flow,
    smoke_test_flow,
    custom_test_flow,
)


def main():
    """Serve all test flows."""
    print("=" * 60)
    print("Prefect Test Flows")
    print("=" * 60)
    print("\nAvailable flows:")
    print("  1. pipeline-tests    - Run any/all pipeline tests")
    print("     Params: pipelines (list), parallel (bool)")
    print("     Pipelines: sql, audio, query, git, code_flow,")
    print("                code_assistance, document, shared")
    print()
    print("  2. full-test-suite   - Run ALL pipeline tests")
    print()
    print("  3. smoke-test        - Quick validation (SQL + Query)")
    print()
    print("  4. custom-tests      - Fine-grained test selection")
    print("     Params: test_path, pattern (-k), marker (-m), timeout")
    print()
    print("Prefect UI: http://localhost:4200")
    print("=" * 60)

    deployments = [
        pipeline_test_flow.to_deployment(
            name="pipeline-tests",
            tags=["testing", "pipeline"],
        ),
        full_test_flow.to_deployment(
            name="full-test-suite",
            tags=["testing", "full"],
        ),
        smoke_test_flow.to_deployment(
            name="smoke-test",
            tags=["testing", "smoke"],
        ),
        custom_test_flow.to_deployment(
            name="custom-tests",
            tags=["testing", "custom"],
        ),
    ]

    serve(*deployments)


if __name__ == "__main__":
    main()
