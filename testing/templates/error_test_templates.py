"""
Parameterized Error Test Templates
==================================

Reusable test templates for common error scenarios across all pipelines.
Use these with pytest.mark.parametrize for comprehensive error coverage.

Usage Examples:
--------------

1. Use pre-defined test cases with parametrize decorator:

    @pytest.mark.parametrize("error_case", VALIDATION_ERROR_CASES, ids=get_validation_test_ids())
    async def test_validation_errors(self, api_client, error_case):
        response = await api_client.post("/api/endpoint", error_case.input_data)
        assert response.status_code == error_case.expected_status

2. Use the convenience decorator:

    @validation_error_tests()
    async def test_my_endpoint_validation(self, api_client, error_case):
        response = await api_client.post("/api/my-endpoint", error_case.input_data)
        assert response.status_code == error_case.expected_status

3. Use ErrorTestRunner for automated test execution:

    async def test_all_validation_errors(self, error_test_runner):
        results = await error_test_runner.run_all_validation_tests("/api/endpoint")
        assert all(r["passed"] for r in results)

4. Customize test cases for specific pipeline:

    from testing.templates import ErrorTestCase, VALIDATION_ERROR_CASES

    SQL_VALIDATION_CASES = VALIDATION_ERROR_CASES + [
        ErrorTestCase(
            name="invalid_database",
            input_data={"query": "SELECT 1", "database": "nonexistent"},
            expected_status=400,
            expected_error_contains="database not found",
            description="Invalid database name"
        ),
    ]
"""

import pytest
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum


class ErrorCategory(Enum):
    """Categories of error test cases."""
    VALIDATION = "validation"
    SERVICE = "service"
    TIMEOUT = "timeout"
    AUTH = "authentication"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    PERMISSION = "permission"


@dataclass
class ErrorTestCase:
    """
    Definition of an error test case.

    Attributes:
        name: Unique identifier for the test case
        input_data: Request data to send (may include special _mock_* flags)
        expected_status: Expected HTTP status code
        expected_error_contains: Substring expected in error message (case-insensitive)
        description: Human-readable description of what this tests
        category: Category of error for grouping and filtering
        skip_condition: Optional callable that returns True to skip this test
        additional_headers: Extra headers to include in request
    """
    name: str
    input_data: Dict[str, Any]
    expected_status: int
    expected_error_contains: Optional[str] = None
    description: str = ""
    category: ErrorCategory = ErrorCategory.VALIDATION
    skip_condition: Optional[Callable[[], bool]] = None
    additional_headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate test case configuration."""
        if not self.name:
            raise ValueError("ErrorTestCase must have a name")
        if not isinstance(self.expected_status, int):
            raise ValueError("expected_status must be an integer")
        if self.expected_status < 100 or self.expected_status >= 600:
            raise ValueError("expected_status must be a valid HTTP status code")


# =============================================================================
# Validation Error Cases (422 Unprocessable Entity)
# =============================================================================

VALIDATION_ERROR_CASES = [
    ErrorTestCase(
        name="empty_request",
        input_data={},
        expected_status=422,
        expected_error_contains="required",
        description="Empty request body should fail validation",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="null_required_field",
        input_data={"query": None},
        expected_status=422,
        expected_error_contains="none is not allowed",
        description="Null value for required field",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="wrong_type_string_for_int",
        input_data={"limit": "not_a_number"},
        expected_status=422,
        expected_error_contains="type",
        description="Wrong type: string instead of int",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="negative_limit",
        input_data={"limit": -1},
        expected_status=422,
        expected_error_contains="greater than",
        description="Negative value for positive-only field",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="exceeds_max_length",
        input_data={"query": "x" * 100000},
        expected_status=422,
        expected_error_contains="length",
        description="Input exceeds maximum length",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="empty_string_required",
        input_data={"query": ""},
        expected_status=422,
        expected_error_contains="empty",
        description="Empty string for required non-empty field",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="whitespace_only",
        input_data={"query": "   "},
        expected_status=422,
        expected_error_contains="blank",
        description="Whitespace-only string for required field",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="invalid_enum_value",
        input_data={"format": "invalid_format"},
        expected_status=422,
        expected_error_contains="not a valid",
        description="Invalid enum value",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="array_instead_of_object",
        input_data={"options": ["a", "b"]},  # When object expected
        expected_status=422,
        expected_error_contains="type",
        description="Array provided when object expected",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="object_instead_of_array",
        input_data={"items": {"key": "value"}},  # When array expected
        expected_status=422,
        expected_error_contains="type",
        description="Object provided when array expected",
        category=ErrorCategory.VALIDATION
    ),
]


# =============================================================================
# Service Unavailable Cases (503 Service Unavailable)
# =============================================================================

SERVICE_UNAVAILABLE_CASES = [
    ErrorTestCase(
        name="llm_unavailable",
        input_data={"query": "test", "_mock_llm_down": True},
        expected_status=503,
        expected_error_contains="LLM service",
        description="LLM service is unavailable",
        category=ErrorCategory.SERVICE
    ),
    ErrorTestCase(
        name="mongodb_unavailable",
        input_data={"query": "test", "_mock_db_down": True},
        expected_status=503,
        expected_error_contains="database",
        description="MongoDB is unavailable",
        category=ErrorCategory.SERVICE
    ),
    ErrorTestCase(
        name="embedding_unavailable",
        input_data={"query": "test", "_mock_embedding_down": True},
        expected_status=503,
        expected_error_contains="embedding",
        description="Embedding service is unavailable",
        category=ErrorCategory.SERVICE
    ),
    ErrorTestCase(
        name="sql_server_unavailable",
        input_data={"query": "test", "_mock_sql_down": True},
        expected_status=503,
        expected_error_contains="SQL Server",
        description="SQL Server is unavailable",
        category=ErrorCategory.SERVICE
    ),
    ErrorTestCase(
        name="external_api_unavailable",
        input_data={"query": "test", "_mock_external_api_down": True},
        expected_status=503,
        expected_error_contains="external service",
        description="External API dependency is unavailable",
        category=ErrorCategory.SERVICE
    ),
]


# =============================================================================
# Timeout Cases (504 Gateway Timeout)
# =============================================================================

TIMEOUT_CASES = [
    ErrorTestCase(
        name="llm_timeout",
        input_data={"query": "test", "_mock_llm_timeout": True},
        expected_status=504,
        expected_error_contains="timeout",
        description="LLM request times out",
        category=ErrorCategory.TIMEOUT
    ),
    ErrorTestCase(
        name="db_timeout",
        input_data={"query": "test", "_mock_db_timeout": True},
        expected_status=504,
        expected_error_contains="timeout",
        description="Database query times out",
        category=ErrorCategory.TIMEOUT
    ),
    ErrorTestCase(
        name="embedding_timeout",
        input_data={"query": "test", "_mock_embedding_timeout": True},
        expected_status=504,
        expected_error_contains="timeout",
        description="Embedding generation times out",
        category=ErrorCategory.TIMEOUT
    ),
    ErrorTestCase(
        name="sql_query_timeout",
        input_data={"query": "test", "_mock_sql_timeout": True},
        expected_status=504,
        expected_error_contains="timeout",
        description="SQL query execution times out",
        category=ErrorCategory.TIMEOUT
    ),
]


# =============================================================================
# Authentication Error Cases (401 Unauthorized)
# =============================================================================

AUTH_ERROR_CASES = [
    ErrorTestCase(
        name="missing_auth",
        input_data={"_no_auth": True},
        expected_status=401,
        expected_error_contains="unauthorized",
        description="Missing authentication token",
        category=ErrorCategory.AUTH
    ),
    ErrorTestCase(
        name="invalid_token",
        input_data={"_auth_token": "invalid_token_12345"},
        expected_status=401,
        expected_error_contains="invalid",
        description="Invalid authentication token",
        category=ErrorCategory.AUTH
    ),
    ErrorTestCase(
        name="expired_token",
        input_data={"_auth_token": "expired_token_12345"},
        expected_status=401,
        expected_error_contains="expired",
        description="Expired authentication token",
        category=ErrorCategory.AUTH
    ),
    ErrorTestCase(
        name="malformed_token",
        input_data={"_auth_token": "not.a.valid.jwt.token"},
        expected_status=401,
        expected_error_contains="malformed",
        description="Malformed JWT token",
        category=ErrorCategory.AUTH
    ),
]


# =============================================================================
# Rate Limiting Cases (429 Too Many Requests)
# =============================================================================

RATE_LIMIT_CASES = [
    ErrorTestCase(
        name="rate_limited",
        input_data={"_trigger_rate_limit": True},
        expected_status=429,
        expected_error_contains="rate limit",
        description="Request rate limit exceeded",
        category=ErrorCategory.RATE_LIMIT
    ),
    ErrorTestCase(
        name="concurrent_limit",
        input_data={"_trigger_concurrent_limit": True},
        expected_status=429,
        expected_error_contains="concurrent",
        description="Concurrent request limit exceeded",
        category=ErrorCategory.RATE_LIMIT
    ),
    ErrorTestCase(
        name="daily_quota_exceeded",
        input_data={"_trigger_daily_limit": True},
        expected_status=429,
        expected_error_contains="quota",
        description="Daily quota exceeded",
        category=ErrorCategory.RATE_LIMIT
    ),
]


# =============================================================================
# Resource Error Cases (404 Not Found, 409 Conflict)
# =============================================================================

RESOURCE_ERROR_CASES = [
    ErrorTestCase(
        name="resource_not_found",
        input_data={"id": "nonexistent_id_12345"},
        expected_status=404,
        expected_error_contains="not found",
        description="Requested resource does not exist",
        category=ErrorCategory.RESOURCE
    ),
    ErrorTestCase(
        name="deleted_resource",
        input_data={"id": "deleted_resource_12345"},
        expected_status=404,
        expected_error_contains="not found",
        description="Resource was previously deleted",
        category=ErrorCategory.RESOURCE
    ),
    ErrorTestCase(
        name="conflict_duplicate",
        input_data={"id": "existing_id", "_create_duplicate": True},
        expected_status=409,
        expected_error_contains="already exists",
        description="Attempting to create duplicate resource",
        category=ErrorCategory.RESOURCE
    ),
    ErrorTestCase(
        name="conflict_version",
        input_data={"id": "resource_id", "version": 1, "_current_version": 5},
        expected_status=409,
        expected_error_contains="version",
        description="Version conflict during update",
        category=ErrorCategory.RESOURCE
    ),
]


# =============================================================================
# Permission Error Cases (403 Forbidden)
# =============================================================================

PERMISSION_ERROR_CASES = [
    ErrorTestCase(
        name="forbidden_resource",
        input_data={"resource_id": "protected_resource"},
        expected_status=403,
        expected_error_contains="forbidden",
        description="Access to resource is forbidden",
        category=ErrorCategory.PERMISSION
    ),
    ErrorTestCase(
        name="insufficient_permissions",
        input_data={"action": "admin_action", "_user_role": "viewer"},
        expected_status=403,
        expected_error_contains="permission",
        description="User lacks required permissions",
        category=ErrorCategory.PERMISSION
    ),
    ErrorTestCase(
        name="read_only_violation",
        input_data={"action": "write", "_resource_readonly": True},
        expected_status=403,
        expected_error_contains="read-only",
        description="Attempting to modify read-only resource",
        category=ErrorCategory.PERMISSION
    ),
]


# =============================================================================
# Bad Request Cases (400 Bad Request)
# =============================================================================

BAD_REQUEST_CASES = [
    ErrorTestCase(
        name="invalid_json",
        input_data={"_raw_body": "{ invalid json }"},
        expected_status=400,
        expected_error_contains="JSON",
        description="Request body is not valid JSON",
        category=ErrorCategory.VALIDATION
    ),
    ErrorTestCase(
        name="unsupported_content_type",
        input_data={"query": "test"},
        expected_status=400,
        expected_error_contains="content-type",
        description="Unsupported content type header",
        additional_headers={"Content-Type": "text/plain"}
    ),
    ErrorTestCase(
        name="missing_content_type",
        input_data={"query": "test"},
        expected_status=400,
        expected_error_contains="content-type",
        description="Missing content type header",
        additional_headers={"Content-Type": ""}
    ),
]


# =============================================================================
# Combined Error Case Collections
# =============================================================================

ALL_CLIENT_ERROR_CASES = (
    VALIDATION_ERROR_CASES +
    AUTH_ERROR_CASES +
    PERMISSION_ERROR_CASES +
    RATE_LIMIT_CASES +
    RESOURCE_ERROR_CASES +
    BAD_REQUEST_CASES
)

ALL_SERVER_ERROR_CASES = (
    SERVICE_UNAVAILABLE_CASES +
    TIMEOUT_CASES
)

ALL_ERROR_CASES = ALL_CLIENT_ERROR_CASES + ALL_SERVER_ERROR_CASES


# =============================================================================
# Error Test Runner
# =============================================================================

@dataclass
class TestResult:
    """Result of running an error test case."""
    test_name: str
    passed: bool
    errors: List[str]
    actual_status: Optional[int] = None
    actual_response: Optional[str] = None
    duration_ms: Optional[float] = None


class ErrorTestRunner:
    """
    Helper class to run error test cases against an API endpoint.

    Usage:
        runner = ErrorTestRunner(api_client)
        results = await runner.run_all_validation_tests("/api/sql/query")

        for result in results:
            if not result.passed:
                print(f"FAILED: {result.test_name}: {result.errors}")
    """

    def __init__(self, api_client):
        """
        Initialize the error test runner.

        Args:
            api_client: API client with async post() method
        """
        self.client = api_client

    def _extract_request_flags(self, input_data: Dict[str, Any]) -> tuple:
        """
        Extract internal test flags from input data.

        Returns:
            Tuple of (cleaned_data, flags_dict)
        """
        flags = {}
        cleaned = {}

        for key, value in input_data.items():
            if key.startswith("_"):
                flags[key] = value
            else:
                cleaned[key] = value

        return cleaned, flags

    async def run_test_case(
        self,
        endpoint: str,
        test_case: ErrorTestCase,
        base_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> TestResult:
        """
        Run a single error test case.

        Args:
            endpoint: API endpoint path
            test_case: The error test case to run
            base_data: Base request data to merge with test case data
            headers: Additional headers for the request

        Returns:
            TestResult with pass/fail status and details
        """
        import time

        # Check skip condition
        if test_case.skip_condition and test_case.skip_condition():
            return TestResult(
                test_name=test_case.name,
                passed=True,
                errors=["SKIPPED: " + (test_case.description or "condition met")]
            )

        # Merge base data with test case data
        request_data = {**(base_data or {}), **test_case.input_data}

        # Extract internal flags from request
        request_data, flags = self._extract_request_flags(request_data)

        # Merge headers
        request_headers = {**(headers or {}), **test_case.additional_headers}

        # Make request with timing
        start_time = time.time()
        try:
            response = await self.client.post(
                endpoint,
                json=request_data,
                headers=request_headers if request_headers else None
            )
            duration_ms = (time.time() - start_time) * 1000
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                passed=False,
                errors=[f"Request failed with exception: {str(e)}"],
                duration_ms=(time.time() - start_time) * 1000
            )

        # Validate response
        result = TestResult(
            test_name=test_case.name,
            passed=True,
            errors=[],
            actual_status=response.status_code,
            actual_response=str(getattr(response, 'json', lambda: response.text)()),
            duration_ms=duration_ms
        )

        # Check status code
        if response.status_code != test_case.expected_status:
            result.passed = False
            result.errors.append(
                f"Expected status {test_case.expected_status}, "
                f"got {response.status_code}"
            )

        # Check error message if specified
        if test_case.expected_error_contains:
            try:
                response_data = response.json() if hasattr(response, 'json') else {}
                error_text = str(response_data).lower()
            except Exception:
                error_text = str(response.text).lower() if hasattr(response, 'text') else ""

            if test_case.expected_error_contains.lower() not in error_text:
                result.passed = False
                result.errors.append(
                    f"Expected error containing '{test_case.expected_error_contains}', "
                    f"got: {error_text[:200]}"
                )

        return result

    async def run_test_cases(
        self,
        endpoint: str,
        test_cases: List[ErrorTestCase],
        base_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stop_on_first_failure: bool = False
    ) -> List[TestResult]:
        """
        Run multiple error test cases.

        Args:
            endpoint: API endpoint path
            test_cases: List of test cases to run
            base_data: Base request data for all tests
            headers: Headers for all requests
            stop_on_first_failure: Stop running after first failure

        Returns:
            List of TestResult objects
        """
        results = []

        for test_case in test_cases:
            result = await self.run_test_case(
                endpoint, test_case, base_data, headers
            )
            results.append(result)

            if stop_on_first_failure and not result.passed:
                break

        return results

    async def run_all_validation_tests(
        self,
        endpoint: str,
        base_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[TestResult]:
        """Run all validation error tests."""
        return await self.run_test_cases(
            endpoint, VALIDATION_ERROR_CASES, base_data, headers
        )

    async def run_all_service_tests(
        self,
        endpoint: str,
        base_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[TestResult]:
        """Run all service unavailable tests."""
        return await self.run_test_cases(
            endpoint, SERVICE_UNAVAILABLE_CASES, base_data, headers
        )

    async def run_all_timeout_tests(
        self,
        endpoint: str,
        base_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[TestResult]:
        """Run all timeout tests."""
        return await self.run_test_cases(
            endpoint, TIMEOUT_CASES, base_data, headers
        )

    async def run_by_category(
        self,
        endpoint: str,
        category: ErrorCategory,
        base_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[TestResult]:
        """Run all tests for a specific error category."""
        category_cases = [c for c in ALL_ERROR_CASES if c.category == category]
        return await self.run_test_cases(
            endpoint, category_cases, base_data, headers
        )


# =============================================================================
# Pytest Parametrize Helpers
# =============================================================================

def get_validation_test_ids() -> List[str]:
    """Get test IDs for validation error cases."""
    return [case.name for case in VALIDATION_ERROR_CASES]


def get_validation_test_params() -> List[tuple]:
    """Get parameters for validation error tests."""
    return [
        (case.input_data, case.expected_status, case.expected_error_contains)
        for case in VALIDATION_ERROR_CASES
    ]


def get_service_test_ids() -> List[str]:
    """Get test IDs for service error cases."""
    return [case.name for case in SERVICE_UNAVAILABLE_CASES]


def get_timeout_test_ids() -> List[str]:
    """Get test IDs for timeout error cases."""
    return [case.name for case in TIMEOUT_CASES]


def get_all_error_test_ids() -> List[str]:
    """Get test IDs for all error cases."""
    return [case.name for case in ALL_ERROR_CASES]


# =============================================================================
# Pytest Parametrize Decorators
# =============================================================================

def validation_error_tests():
    """
    Decorator for parameterized validation error tests.

    Usage:
        @validation_error_tests()
        async def test_validation(self, api_client, error_case):
            response = await api_client.post("/api/endpoint", error_case.input_data)
            assert response.status_code == error_case.expected_status
    """
    return pytest.mark.parametrize(
        "error_case",
        VALIDATION_ERROR_CASES,
        ids=get_validation_test_ids()
    )


def service_error_tests():
    """
    Decorator for parameterized service and timeout error tests.

    Usage:
        @service_error_tests()
        async def test_service_errors(self, api_client, error_case):
            response = await api_client.post("/api/endpoint", error_case.input_data)
            assert response.status_code == error_case.expected_status
    """
    combined_cases = SERVICE_UNAVAILABLE_CASES + TIMEOUT_CASES
    return pytest.mark.parametrize(
        "error_case",
        combined_cases,
        ids=[c.name for c in combined_cases]
    )


def auth_error_tests():
    """Decorator for parameterized authentication error tests."""
    return pytest.mark.parametrize(
        "error_case",
        AUTH_ERROR_CASES,
        ids=[c.name for c in AUTH_ERROR_CASES]
    )


def rate_limit_tests():
    """Decorator for parameterized rate limit error tests."""
    return pytest.mark.parametrize(
        "error_case",
        RATE_LIMIT_CASES,
        ids=[c.name for c in RATE_LIMIT_CASES]
    )


def resource_error_tests():
    """Decorator for parameterized resource error tests."""
    return pytest.mark.parametrize(
        "error_case",
        RESOURCE_ERROR_CASES,
        ids=[c.name for c in RESOURCE_ERROR_CASES]
    )


def all_client_error_tests():
    """Decorator for all client error tests (4xx status codes)."""
    return pytest.mark.parametrize(
        "error_case",
        ALL_CLIENT_ERROR_CASES,
        ids=[c.name for c in ALL_CLIENT_ERROR_CASES]
    )


def all_server_error_tests():
    """Decorator for all server error tests (5xx status codes)."""
    return pytest.mark.parametrize(
        "error_case",
        ALL_SERVER_ERROR_CASES,
        ids=[c.name for c in ALL_SERVER_ERROR_CASES]
    )


def error_tests_by_category(category: ErrorCategory):
    """
    Decorator for error tests filtered by category.

    Usage:
        @error_tests_by_category(ErrorCategory.VALIDATION)
        async def test_validation_only(self, api_client, error_case):
            ...
    """
    category_cases = [c for c in ALL_ERROR_CASES if c.category == category]
    return pytest.mark.parametrize(
        "error_case",
        category_cases,
        ids=[c.name for c in category_cases]
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def error_test_runner(api_client):
    """
    Fixture providing error test runner.

    Requires an api_client fixture to be defined in your conftest.py.

    Usage:
        async def test_errors(self, error_test_runner):
            results = await error_test_runner.run_all_validation_tests("/api/endpoint")
            assert all(r.passed for r in results)
    """
    return ErrorTestRunner(api_client)


@pytest.fixture
def validation_cases() -> List[ErrorTestCase]:
    """Fixture providing validation error cases."""
    return VALIDATION_ERROR_CASES.copy()


@pytest.fixture
def service_cases() -> List[ErrorTestCase]:
    """Fixture providing service error cases."""
    return SERVICE_UNAVAILABLE_CASES.copy()


@pytest.fixture
def timeout_cases() -> List[ErrorTestCase]:
    """Fixture providing timeout error cases."""
    return TIMEOUT_CASES.copy()


@pytest.fixture
def auth_cases() -> List[ErrorTestCase]:
    """Fixture providing authentication error cases."""
    return AUTH_ERROR_CASES.copy()


# =============================================================================
# Test Case Factory Functions
# =============================================================================

def create_validation_case(
    name: str,
    input_data: Dict[str, Any],
    expected_error: str,
    description: str = ""
) -> ErrorTestCase:
    """
    Factory function to create a validation error test case.

    Args:
        name: Unique test name
        input_data: Request data
        expected_error: Expected error message substring
        description: Optional description

    Returns:
        ErrorTestCase configured for 422 validation error
    """
    return ErrorTestCase(
        name=name,
        input_data=input_data,
        expected_status=422,
        expected_error_contains=expected_error,
        description=description,
        category=ErrorCategory.VALIDATION
    )


def create_not_found_case(
    name: str,
    input_data: Dict[str, Any],
    description: str = ""
) -> ErrorTestCase:
    """Factory function for 404 Not Found test case."""
    return ErrorTestCase(
        name=name,
        input_data=input_data,
        expected_status=404,
        expected_error_contains="not found",
        description=description,
        category=ErrorCategory.RESOURCE
    )


def create_service_error_case(
    name: str,
    input_data: Dict[str, Any],
    service_name: str,
    description: str = ""
) -> ErrorTestCase:
    """Factory function for 503 Service Unavailable test case."""
    return ErrorTestCase(
        name=name,
        input_data=input_data,
        expected_status=503,
        expected_error_contains=service_name,
        description=description or f"{service_name} is unavailable",
        category=ErrorCategory.SERVICE
    )


def create_timeout_case(
    name: str,
    input_data: Dict[str, Any],
    description: str = ""
) -> ErrorTestCase:
    """Factory function for 504 Gateway Timeout test case."""
    return ErrorTestCase(
        name=name,
        input_data=input_data,
        expected_status=504,
        expected_error_contains="timeout",
        description=description,
        category=ErrorCategory.TIMEOUT
    )


# =============================================================================
# Pipeline-Specific Error Case Extensions
# =============================================================================

def get_sql_pipeline_error_cases() -> List[ErrorTestCase]:
    """
    Get error cases specific to SQL pipeline.

    Extends base validation cases with SQL-specific scenarios.
    """
    sql_cases = VALIDATION_ERROR_CASES.copy()
    sql_cases.extend([
        ErrorTestCase(
            name="invalid_database_name",
            input_data={"query": "SELECT 1", "database": "nonexistent_db"},
            expected_status=400,
            expected_error_contains="database",
            description="Invalid database name specified",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="sql_injection_attempt",
            input_data={"query": "'; DROP TABLE users; --"},
            expected_status=400,
            expected_error_contains="invalid",
            description="Potential SQL injection attempt",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="destructive_query_blocked",
            input_data={"query": "DELETE FROM important_table"},
            expected_status=403,
            expected_error_contains="not allowed",
            description="Destructive query blocked by safety rules",
            category=ErrorCategory.PERMISSION
        ),
    ])
    return sql_cases


def get_audio_pipeline_error_cases() -> List[ErrorTestCase]:
    """
    Get error cases specific to audio pipeline.

    Extends base validation cases with audio-specific scenarios.
    """
    audio_cases = VALIDATION_ERROR_CASES.copy()
    audio_cases.extend([
        ErrorTestCase(
            name="unsupported_audio_format",
            input_data={"file_path": "/tmp/audio.xyz"},
            expected_status=400,
            expected_error_contains="format",
            description="Unsupported audio file format",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="audio_file_too_large",
            input_data={"file_path": "/tmp/huge_file.mp3", "_file_size_mb": 500},
            expected_status=400,
            expected_error_contains="size",
            description="Audio file exceeds maximum size limit",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="corrupted_audio_file",
            input_data={"file_path": "/tmp/corrupted.mp3"},
            expected_status=400,
            expected_error_contains="corrupt",
            description="Audio file is corrupted or unreadable",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="transcription_service_unavailable",
            input_data={"file_path": "/tmp/audio.mp3", "_mock_transcription_down": True},
            expected_status=503,
            expected_error_contains="transcription",
            description="Transcription service unavailable",
            category=ErrorCategory.SERVICE
        ),
    ])
    return audio_cases


def get_document_pipeline_error_cases() -> List[ErrorTestCase]:
    """
    Get error cases specific to document pipeline.

    Extends base validation cases with document-specific scenarios.
    """
    doc_cases = VALIDATION_ERROR_CASES.copy()
    doc_cases.extend([
        ErrorTestCase(
            name="unsupported_document_type",
            input_data={"file_path": "/tmp/file.xyz"},
            expected_status=400,
            expected_error_contains="format",
            description="Unsupported document format",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="document_too_large",
            input_data={"file_path": "/tmp/huge.pdf", "_file_size_mb": 100},
            expected_status=400,
            expected_error_contains="size",
            description="Document exceeds maximum size limit",
            category=ErrorCategory.VALIDATION
        ),
        ErrorTestCase(
            name="password_protected_document",
            input_data={"file_path": "/tmp/protected.pdf"},
            expected_status=400,
            expected_error_contains="password",
            description="Document is password protected",
            category=ErrorCategory.VALIDATION
        ),
    ])
    return doc_cases
