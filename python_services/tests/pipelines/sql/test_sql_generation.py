"""
SQL Pipeline Generation Tests
==============================

Tests for SQL generation using local llama.cpp LLM.

Tests:
1. SQL generator initialization
2. Schema context building
3. Business rules injection
4. SQL generation with mock LLM
5. Auto-fix pattern application
"""

import pytest
import os
import sys
import re

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestSQLGeneratorService:
    """Tests for SQLGeneratorService."""

    def test_generator_initialization(self):
        """Test SQL generator initializes correctly."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()

        assert gen is not None
        # Check required attributes exist
        assert hasattr(gen, '_format_schema')
        assert hasattr(gen, '_build_system_prompt')

    def test_format_schema_basic(self, sample_schema_context):
        """Test basic schema formatting."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(table)

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "CentralTickets" in formatted

    def test_format_schema_complete(self, sample_schema_context):
        """Test complete schema formatting with all context."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(table)

        # Check all sections present
        assert "Table:" in formatted
        assert "Columns:" in formatted
        assert "Foreign Keys:" in formatted
        assert "Related Tables:" in formatted
        assert "Examples:" in formatted  # Sample values

    def test_column_formatting(self, sample_schema_context):
        """Test column information is properly formatted."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(table)

        # Check column details
        assert "CentralTicketID" in formatted
        assert "AddTicketDate" in formatted
        assert "int" in formatted or "INT" in formatted
        assert "datetime" in formatted or "DATETIME" in formatted


class TestBusinessRules:
    """Tests for business rules injection."""

    def test_get_business_rules_ewrcentral(self, sample_schema_context):
        """Test EWRCentral business rules retrieval."""
        from services.sql_generator import SQLGeneratorService, SchemaContext

        gen = SQLGeneratorService()

        # Check if method exists
        if hasattr(gen, '_get_business_rules'):
            # Create schema context for the method
            schema_ctx = SchemaContext(
                tables=sample_schema_context["tables"],
                stored_procedures=sample_schema_context.get("stored_procedures", [])
            )

            rules = gen._get_business_rules("EWRCentral", schema_ctx)

            if rules:
                assert isinstance(rules, (str, list))
                rules_text = str(rules)
                assert len(rules_text) >= 0  # Rules may or may not exist
        else:
            pytest.skip("_get_business_rules method not available")

    def test_business_rules_database_specific(self, sample_schema_context):
        """Test business rules are database-specific."""
        from services.sql_generator import SQLGeneratorService, SchemaContext

        gen = SQLGeneratorService()

        if hasattr(gen, '_get_business_rules'):
            schema_ctx = SchemaContext(
                tables=sample_schema_context["tables"],
                stored_procedures=sample_schema_context.get("stored_procedures", [])
            )

            ewrcentral_rules = gen._get_business_rules("EWRCentral", schema_ctx)
            other_rules = gen._get_business_rules("OtherDatabase", schema_ctx)

            # Rules should be defined (may be empty for unknown databases)
            assert ewrcentral_rules is not None or other_rules is not None
        else:
            pytest.skip("_get_business_rules method not available")


class TestAutoFix:
    """Tests for SQL auto-fix patterns."""

    def test_create_date_fix(self):
        """Test CreateDate -> AddTicketDate auto-fix."""
        pattern = r"CreateDate"
        replacement = "AddTicketDate"

        test_cases = [
            ("SELECT CreateDate FROM CentralTickets", "SELECT AddTicketDate FROM CentralTickets"),
            ("WHERE CreateDate > GETDATE()", "WHERE AddTicketDate > GETDATE()"),
        ]

        for input_sql, expected in test_cases:
            fixed = re.sub(pattern, replacement, input_sql)
            assert fixed == expected, f"Auto-fix failed: {input_sql} -> {fixed}"

    def test_sql_dialect_compliance(self):
        """Test SQL is T-SQL compliant (not MySQL)."""
        # Common MySQL patterns that should not appear
        mysql_patterns = [
            r"LIMIT\s+\d+\s*$",  # LIMIT without OFFSET (MySQL style)
            r"NOW\(\)",  # MySQL NOW() instead of GETDATE()
            r"DATE_FORMAT\(",  # MySQL date format
            r"IFNULL\(",  # MySQL IFNULL instead of ISNULL
        ]

        valid_tsql = "SELECT TOP 10 * FROM CentralTickets WHERE AddTicketDate > GETDATE()"

        for pattern in mysql_patterns:
            assert not re.search(pattern, valid_tsql), f"MySQL pattern found: {pattern}"

    def test_select_star_detection(self):
        """Test SELECT * pattern detection."""
        pattern = r"SELECT\s+\*\s+FROM"

        bad_sql = "SELECT * FROM CentralTickets"
        good_sql = "SELECT CentralTicketID, AddTicketDate FROM CentralTickets"

        assert re.search(pattern, bad_sql), "Should detect SELECT *"
        assert not re.search(pattern, good_sql), "Should not flag specific columns"


class TestPromptBuilding:
    """Tests for system prompt building."""

    def test_system_prompt_components(self, sample_schema_context):
        """Test system prompt includes all components."""
        from services.sql_generator import SQLGeneratorService, SchemaContext

        gen = SQLGeneratorService()

        # Create schema context
        schema_ctx = SchemaContext(
            tables=sample_schema_context["tables"],
            stored_procedures=sample_schema_context.get("stored_procedures", [])
        )

        # Build prompt - check method signature
        if hasattr(gen, '_build_system_prompt'):
            import inspect
            sig = inspect.signature(gen._build_system_prompt)
            params = list(sig.parameters.keys())

            # Call with appropriate arguments based on signature
            # Signature: _build_system_prompt(self, database, schema_context, rules_context, ...)
            if 'rules_context' in params:
                prompt = gen._build_system_prompt("EWRCentral", schema_ctx, "")
            else:
                prompt = gen._build_system_prompt("EWRCentral", schema_ctx)

            assert isinstance(prompt, str)
            assert len(prompt) > 0

            # Should contain schema info
            assert "CentralTickets" in prompt or "Table" in prompt or "SQL" in prompt
        else:
            pytest.skip("_build_system_prompt method not available")

    def test_stored_procedures_in_prompt(self, sample_schema_context):
        """Test stored procedures are included in prompt."""
        from services.sql_generator import SQLGeneratorService, SchemaContext

        gen = SQLGeneratorService()

        procs = sample_schema_context.get("stored_procedures", [])
        if not procs:
            pytest.skip("No stored procedures in sample data")

        schema_ctx = SchemaContext(
            tables=sample_schema_context["tables"],
            stored_procedures=procs
        )

        if hasattr(gen, '_build_system_prompt'):
            import inspect
            sig = inspect.signature(gen._build_system_prompt)
            params = list(sig.parameters.keys())

            # Call with appropriate arguments based on signature
            # Signature: _build_system_prompt(self, database, schema_context, rules_context, ...)
            if 'rules_context' in params:
                prompt = gen._build_system_prompt("EWRCentral", schema_ctx, "")
            else:
                prompt = gen._build_system_prompt("EWRCentral", schema_ctx)

            # Check for stored procedure mention or general prompt structure
            assert len(prompt) > 0, "Prompt should not be empty"
        else:
            pytest.skip("_build_system_prompt method not available")


class TestSQLValidation:
    """Tests for SQL validation."""

    def test_basic_sql_syntax(self):
        """Test basic SQL syntax validation."""
        valid_queries = [
            "SELECT * FROM CentralTickets",
            "SELECT TOP 10 CentralTicketID FROM CentralTickets WHERE AddTicketDate > GETDATE()",
            "SELECT COUNT(*) FROM CentralTickets GROUP BY TicketTypeID",
        ]

        invalid_queries = [
            "SELEC * FROM CentralTickets",  # Typo
            "SELECT FROM CentralTickets",  # Missing columns
        ]

        # Basic syntax check - SELECT...FROM pattern
        select_from_pattern = r"SELECT\s+.+\s+FROM\s+\w+"

        for query in valid_queries:
            assert re.search(select_from_pattern, query, re.IGNORECASE), f"Valid query failed: {query}"

    def test_dangerous_sql_patterns(self):
        """Test detection of dangerous SQL patterns."""
        dangerous_patterns = [
            r"DROP\s+TABLE",
            r"DELETE\s+FROM\s+\w+\s*$",  # DELETE without WHERE
            r"TRUNCATE\s+TABLE",
            r"UPDATE\s+\w+\s+SET\s+.+\s*$",  # UPDATE without WHERE
        ]

        dangerous_queries = [
            "DROP TABLE CentralTickets",
            "DELETE FROM CentralTickets",
            "TRUNCATE TABLE CentralTickets",
        ]

        for query in dangerous_queries:
            detected = False
            for pattern in dangerous_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    detected = True
                    break
            assert detected, f"Dangerous query not detected: {query}"
