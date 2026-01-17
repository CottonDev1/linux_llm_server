"""
Custom Test Assertions
======================

Specialized assertions for pipeline testing.
"""

import re
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher


def assert_valid_sql(sql: str, database: str = None) -> None:
    """
    Assert that a SQL string is syntactically valid.

    Performs basic T-SQL syntax validation:
    - Contains required keywords
    - Balanced parentheses
    - No obvious syntax errors

    Args:
        sql: SQL query string to validate
        database: Optional database name for context

    Raises:
        AssertionError: If SQL is invalid
    """
    if not sql or not sql.strip():
        raise AssertionError("SQL query is empty")

    sql_upper = sql.upper()

    # Must contain at least one SQL keyword
    sql_keywords = [
        "SELECT", "INSERT", "UPDATE", "DELETE",
        "CREATE", "ALTER", "DROP", "EXEC", "EXECUTE"
    ]
    has_keyword = any(kw in sql_upper for kw in sql_keywords)
    if not has_keyword:
        raise AssertionError(f"SQL does not contain any valid SQL keywords: {sql[:100]}...")

    # Check balanced parentheses
    open_count = sql.count('(')
    close_count = sql.count(')')
    if open_count != close_count:
        raise AssertionError(
            f"Unbalanced parentheses in SQL: {open_count} open, {close_count} close"
        )

    # Check for common T-SQL issues
    forbidden_patterns = [
        (r"SELECT\s+FROM", "SELECT immediately followed by FROM (missing columns)"),
        (r"WHERE\s+(AND|OR)", "WHERE immediately followed by AND/OR"),
        (r"FROM\s+(WHERE|ORDER|GROUP)", "FROM immediately followed by clause (missing table)"),
    ]

    for pattern, message in forbidden_patterns:
        if re.search(pattern, sql_upper):
            raise AssertionError(f"SQL syntax error: {message}")


def assert_document_stored(
    collection,
    document_id: str,
    expected_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assert that a document was stored in MongoDB.

    Args:
        collection: MongoDB collection
        document_id: Document ID to find
        expected_fields: Optional list of fields that must be present

    Returns:
        The found document

    Raises:
        AssertionError: If document not found or missing fields
    """
    doc = collection.find_one({"_id": document_id})

    if doc is None:
        raise AssertionError(f"Document not found: {document_id}")

    if expected_fields:
        missing = [f for f in expected_fields if f not in doc]
        if missing:
            raise AssertionError(
                f"Document missing expected fields: {missing}. "
                f"Found fields: {list(doc.keys())}"
            )

    return doc


def assert_similar_text(
    actual: str,
    expected: str,
    threshold: float = 0.8,
    message: str = None,
) -> float:
    """
    Assert that two text strings are similar.

    Uses SequenceMatcher for similarity comparison.

    Args:
        actual: Actual text
        expected: Expected text
        threshold: Minimum similarity ratio (0.0 to 1.0)
        message: Optional custom error message

    Returns:
        Similarity ratio

    Raises:
        AssertionError: If similarity is below threshold
    """
    # Normalize texts
    actual_norm = ' '.join(actual.lower().split())
    expected_norm = ' '.join(expected.lower().split())

    ratio = SequenceMatcher(None, actual_norm, expected_norm).ratio()

    if ratio < threshold:
        default_msg = (
            f"Text similarity {ratio:.2f} is below threshold {threshold}.\n"
            f"Expected: {expected[:100]}...\n"
            f"Actual: {actual[:100]}..."
        )
        raise AssertionError(message or default_msg)

    return ratio


def assert_llm_response_valid(
    response,
    min_length: int = 1,
    max_length: Optional[int] = None,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
) -> None:
    """
    Assert that an LLM response is valid.

    Args:
        response: LLMResponse object from fixtures
        min_length: Minimum response length
        max_length: Optional maximum response length
        must_contain: Optional list of strings that must be in response
        must_not_contain: Optional list of strings that must not be in response

    Raises:
        AssertionError: If response is invalid
    """
    if not response.success:
        raise AssertionError(f"LLM request failed: {response.error}")

    text = response.text

    if len(text) < min_length:
        raise AssertionError(
            f"Response too short: {len(text)} chars (min: {min_length})"
        )

    if max_length and len(text) > max_length:
        raise AssertionError(
            f"Response too long: {len(text)} chars (max: {max_length})"
        )

    if must_contain:
        text_lower = text.lower()
        missing = [s for s in must_contain if s.lower() not in text_lower]
        if missing:
            raise AssertionError(
                f"Response missing required content: {missing}"
            )

    if must_not_contain:
        text_lower = text.lower()
        found = [s for s in must_not_contain if s.lower() in text_lower]
        if found:
            raise AssertionError(
                f"Response contains forbidden content: {found}"
            )


def assert_mongodb_document(
    doc: Dict[str, Any],
    schema: Dict[str, type],
    allow_extra: bool = True,
) -> None:
    """
    Assert that a MongoDB document matches expected schema.

    Args:
        doc: Document to validate
        schema: Dict mapping field names to expected types
        allow_extra: Whether to allow fields not in schema

    Raises:
        AssertionError: If document doesn't match schema
    """
    if doc is None:
        raise AssertionError("Document is None")

    # Check required fields
    for field, expected_type in schema.items():
        if field not in doc:
            raise AssertionError(f"Missing required field: {field}")

        actual_type = type(doc[field])

        # Handle None values
        if doc[field] is None:
            continue  # None is acceptable for optional fields

        # Handle type checking
        if expected_type is not None and not isinstance(doc[field], expected_type):
            raise AssertionError(
                f"Field '{field}' has wrong type. "
                f"Expected {expected_type.__name__}, got {actual_type.__name__}"
            )

    # Check for unexpected fields if not allowed
    if not allow_extra:
        extra = set(doc.keys()) - set(schema.keys()) - {"_id"}
        if extra:
            raise AssertionError(f"Unexpected fields in document: {extra}")


def assert_sql_contains_tables(
    sql: str,
    tables: List[str],
    message: str = None,
) -> None:
    """
    Assert that a SQL query references specific tables.

    Args:
        sql: SQL query string
        tables: List of table names that should be referenced
        message: Optional custom error message

    Raises:
        AssertionError: If any table is not referenced
    """
    sql_upper = sql.upper()
    missing = []

    def camel_to_snake(name: str) -> str:
        """Convert CamelCase to snake_case."""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).upper()

    for table in tables:
        table_upper = table.upper()
        snake_upper = camel_to_snake(table)

        # Check for table name in either CamelCase or snake_case format
        pattern_camel = rf'\b{table_upper}\b'
        pattern_snake = rf'\b{snake_upper}\b'

        if not (re.search(pattern_camel, sql_upper) or re.search(pattern_snake, sql_upper)):
            missing.append(table)

    if missing:
        default_msg = f"SQL does not reference expected tables: {missing}"
        raise AssertionError(message or default_msg)


def assert_execution_success(
    result: Dict[str, Any],
    min_rows: int = 0,
    max_rows: Optional[int] = None,
) -> None:
    """
    Assert that a SQL execution result indicates success.

    Args:
        result: Execution result dictionary
        min_rows: Minimum expected rows
        max_rows: Optional maximum expected rows

    Raises:
        AssertionError: If execution failed or row count mismatch
    """
    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        raise AssertionError(f"SQL execution failed: {error}")

    row_count = result.get("row_count", 0)

    if row_count < min_rows:
        raise AssertionError(
            f"Too few rows returned: {row_count} (min: {min_rows})"
        )

    if max_rows is not None and row_count > max_rows:
        raise AssertionError(
            f"Too many rows returned: {row_count} (max: {max_rows})"
        )


def assert_similarity_score(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
    message: str = None,
) -> None:
    """
    Assert that a similarity score is within expected range.

    Args:
        score: Similarity score to validate
        min_score: Minimum acceptable score
        max_score: Maximum acceptable score
        message: Optional custom error message

    Raises:
        AssertionError: If score is outside the expected range
    """
    if score < min_score or score > max_score:
        default_msg = f"Similarity score {score} is outside range [{min_score}, {max_score}]"
        raise AssertionError(message or default_msg)


def assert_field_types(
    doc: Dict[str, Any],
    field_types: Dict[str, type],
) -> None:
    """
    Assert that document fields have the expected types.

    Args:
        doc: Document to validate
        field_types: Dict mapping field names to expected types

    Raises:
        AssertionError: If any field has the wrong type
    """
    for field, expected_type in field_types.items():
        if field in doc:
            actual_value = doc[field]
            if actual_value is not None and not isinstance(actual_value, expected_type):
                raise AssertionError(
                    f"Field '{field}' has type {type(actual_value).__name__}, "
                    f"expected {expected_type.__name__}"
                )


def assert_required_fields(
    doc: Dict[str, Any],
    required: List[str],
) -> None:
    """
    Assert that document contains all required fields.

    Args:
        doc: Document to validate
        required: List of required field names

    Raises:
        AssertionError: If any required field is missing
    """
    missing = [f for f in required if f not in doc]
    if missing:
        raise AssertionError(f"Missing required fields: {missing}")
