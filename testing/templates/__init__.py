"""
Test Templates Package
======================

Provides reusable, parameterized test templates for common error scenarios
that can be used across all pipeline tests.

Example Usage
-------------

1. Import and use error test cases with parametrize:

    from testing.templates import VALIDATION_ERROR_CASES, get_validation_test_ids

    @pytest.mark.parametrize("error_case", VALIDATION_ERROR_CASES, ids=get_validation_test_ids())
    async def test_validation_errors(api_client, error_case):
        response = await api_client.post("/api/endpoint", error_case.input_data)
        assert response.status_code == error_case.expected_status

2. Use convenience decorators:

    from testing.templates import validation_error_tests, service_error_tests

    @validation_error_tests()
    async def test_my_endpoint_validation(api_client, error_case):
        response = await api_client.post("/api/my-endpoint", error_case.input_data)
        assert response.status_code == error_case.expected_status

3. Use ErrorTestRunner for bulk test execution:

    from testing.templates import ErrorTestRunner

    async def test_all_validation_errors(api_client):
        runner = ErrorTestRunner(api_client)
        results = await runner.run_all_validation_tests("/api/endpoint")
        assert all(r.passed for r in results)

4. Create custom test cases:

    from testing.templates import ErrorTestCase, ErrorCategory

    MY_CUSTOM_CASES = [
        ErrorTestCase(
            name="custom_error",
            input_data={"field": "invalid"},
            expected_status=422,
            expected_error_contains="validation failed",
            category=ErrorCategory.VALIDATION
        ),
    ]

5. Use pipeline-specific error cases:

    from testing.templates import (
        get_sql_pipeline_error_cases,
        get_audio_pipeline_error_cases,
        get_document_pipeline_error_cases,
    )

    SQL_ERROR_CASES = get_sql_pipeline_error_cases()

Available Test Case Collections
-------------------------------

- VALIDATION_ERROR_CASES: Input validation errors (422)
- SERVICE_UNAVAILABLE_CASES: Service unavailable errors (503)
- TIMEOUT_CASES: Timeout errors (504)
- AUTH_ERROR_CASES: Authentication errors (401)
- RATE_LIMIT_CASES: Rate limiting errors (429)
- RESOURCE_ERROR_CASES: Resource not found/conflict (404, 409)
- PERMISSION_ERROR_CASES: Permission denied errors (403)
- BAD_REQUEST_CASES: Malformed request errors (400)
- ALL_CLIENT_ERROR_CASES: All 4xx errors combined
- ALL_SERVER_ERROR_CASES: All 5xx errors combined
- ALL_ERROR_CASES: All error cases combined

Available Decorators
--------------------

- validation_error_tests(): Parametrize with validation cases
- service_error_tests(): Parametrize with service/timeout cases
- auth_error_tests(): Parametrize with authentication cases
- rate_limit_tests(): Parametrize with rate limit cases
- resource_error_tests(): Parametrize with resource error cases
- all_client_error_tests(): Parametrize with all 4xx cases
- all_server_error_tests(): Parametrize with all 5xx cases
- error_tests_by_category(category): Parametrize by ErrorCategory

Factory Functions
-----------------

- create_validation_case(): Create 422 validation error case
- create_not_found_case(): Create 404 not found case
- create_service_error_case(): Create 503 service error case
- create_timeout_case(): Create 504 timeout case
"""

from .error_test_templates import (
    # Data classes and enums
    ErrorTestCase,
    ErrorCategory,
    TestResult,

    # Validation error cases (422)
    VALIDATION_ERROR_CASES,

    # Service unavailable cases (503)
    SERVICE_UNAVAILABLE_CASES,

    # Timeout cases (504)
    TIMEOUT_CASES,

    # Authentication error cases (401)
    AUTH_ERROR_CASES,

    # Rate limiting cases (429)
    RATE_LIMIT_CASES,

    # Resource error cases (404, 409)
    RESOURCE_ERROR_CASES,

    # Permission error cases (403)
    PERMISSION_ERROR_CASES,

    # Bad request cases (400)
    BAD_REQUEST_CASES,

    # Combined collections
    ALL_CLIENT_ERROR_CASES,
    ALL_SERVER_ERROR_CASES,
    ALL_ERROR_CASES,

    # Test runner class
    ErrorTestRunner,

    # Parametrize helpers
    get_validation_test_ids,
    get_validation_test_params,
    get_service_test_ids,
    get_timeout_test_ids,
    get_all_error_test_ids,

    # Parametrize decorators
    validation_error_tests,
    service_error_tests,
    auth_error_tests,
    rate_limit_tests,
    resource_error_tests,
    all_client_error_tests,
    all_server_error_tests,
    error_tests_by_category,

    # Factory functions
    create_validation_case,
    create_not_found_case,
    create_service_error_case,
    create_timeout_case,

    # Pipeline-specific case getters
    get_sql_pipeline_error_cases,
    get_audio_pipeline_error_cases,
    get_document_pipeline_error_cases,
)

__all__ = [
    # Data classes and enums
    "ErrorTestCase",
    "ErrorCategory",
    "TestResult",

    # Validation error cases (422)
    "VALIDATION_ERROR_CASES",

    # Service unavailable cases (503)
    "SERVICE_UNAVAILABLE_CASES",

    # Timeout cases (504)
    "TIMEOUT_CASES",

    # Authentication error cases (401)
    "AUTH_ERROR_CASES",

    # Rate limiting cases (429)
    "RATE_LIMIT_CASES",

    # Resource error cases (404, 409)
    "RESOURCE_ERROR_CASES",

    # Permission error cases (403)
    "PERMISSION_ERROR_CASES",

    # Bad request cases (400)
    "BAD_REQUEST_CASES",

    # Combined collections
    "ALL_CLIENT_ERROR_CASES",
    "ALL_SERVER_ERROR_CASES",
    "ALL_ERROR_CASES",

    # Test runner class
    "ErrorTestRunner",

    # Parametrize helpers
    "get_validation_test_ids",
    "get_validation_test_params",
    "get_service_test_ids",
    "get_timeout_test_ids",
    "get_all_error_test_ids",

    # Parametrize decorators
    "validation_error_tests",
    "service_error_tests",
    "auth_error_tests",
    "rate_limit_tests",
    "resource_error_tests",
    "all_client_error_tests",
    "all_server_error_tests",
    "error_tests_by_category",

    # Factory functions
    "create_validation_case",
    "create_not_found_case",
    "create_service_error_case",
    "create_timeout_case",

    # Pipeline-specific case getters
    "get_sql_pipeline_error_cases",
    "get_audio_pipeline_error_cases",
    "get_document_pipeline_error_cases",
]
