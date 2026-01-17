"""
Prefect Deployments for Playwright UI Tests
============================================

Creates deployments for Playwright E2E tests that can be triggered
from the Prefect dashboard at http://10.101.20.21:4200

Usage:
    # Start the deployment server (keeps running)
    python playwright_deployments.py

    # Then trigger from Prefect dashboard or CLI:
    prefect deployment run "playwright-full-test/full-suite"
    prefect deployment run "playwright-quick-test/smoke-test"
"""

from prefect import serve

from playwright_test_flow import (
    run_playwright_tests,
    run_quick_test,
    run_full_test,
    run_audio_test,
    run_audio_analysis_test,
    run_user_management_test,
    TEST_SUITES,
)


def main():
    """Deploy all Playwright test flows using serve()."""
    print("=" * 60)
    print("Registering Playwright Test Deployments")
    print("=" * 60)
    print()

    # Create deployments using the new flow.to_deployment() API
    full_suite = run_full_test.to_deployment(
        name="full-suite",
        description="Run ALL Playwright E2E tests (takes ~15-30 minutes)",
        tags=["playwright", "e2e", "full-suite"],
    )

    smoke_test = run_quick_test.to_deployment(
        name="smoke-test",
        description="Quick smoke test - admin pages, sidebar, knowledge base (~3-5 minutes)",
        tags=["playwright", "e2e", "smoke-test", "quick"],
    )

    custom_test = run_playwright_tests.to_deployment(
        name="custom",
        description="Run specific test suites (configure via parameters)",
        tags=["playwright", "e2e", "custom"],
        parameters={
            "suites": None,
            "parallel": False,
            "headless": True,
            "generate_report": True,
        },
    )

    audio_tests = run_audio_test.to_deployment(
        name="audio-tests",
        description="Audio processing E2E tests - single, bulk, and analysis workflow",
        tags=["playwright", "e2e", "audio"],
    )

    audio_analysis = run_audio_analysis_test.to_deployment(
        name="audio-analysis",
        description="Audio analysis workflow tests - staff monitoring, search, view/edit",
        tags=["playwright", "e2e", "audio", "analysis"],
    )

    user_management = run_user_management_test.to_deployment(
        name="user-management",
        description="User management E2E tests - creation, roles, staff dashboard",
        tags=["playwright", "e2e", "users", "rbac"],
    )

    print("Deployments configured:")
    print("  - playwright_full_test/full-suite")
    print("  - playwright_quick_test/smoke-test")
    print("  - playwright_test_flow/custom")
    print("  - playwright_audio_test/audio-tests")
    print("  - playwright_audio_analysis_test/audio-analysis")
    print("  - playwright_user_management_test/user-management")
    print()
    print("Available test suites:")
    for suite_name, suite_info in TEST_SUITES.items():
        print(f"  - {suite_name}: {suite_info['description']}")
    print()
    print("Starting deployment server...")
    print("Trigger tests from:")
    print("  - Prefect Dashboard: http://10.101.20.21:4200")
    print("  - CLI: prefect deployment run 'playwright_full_test/full-suite'")
    print()
    print("Press Ctrl+C to stop the server.")
    print()

    # Serve all deployments (this keeps running to handle triggers)
    # serve() must be called from a synchronous context
    serve(
        full_suite,
        smoke_test,
        custom_test,
        audio_tests,
        audio_analysis,
        user_management,
    )


if __name__ == "__main__":
    main()
