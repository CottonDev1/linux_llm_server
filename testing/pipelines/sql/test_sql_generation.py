"""
SQL Generation Tests using local llama.cpp.

Tests SQL generation using local LLM endpoints including:
- Basic SQL generation from natural language
- Schema-guided generation
- Rule-guided generation
- Multi-table join generation
- Date/time query generation
- Aggregation query generation
- Token usage tracking

All tests use LOCAL llama.cpp only (port 8080 for SQL model).
NO external APIs permitted.
"""

import pytest
import re

from config.test_config import PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.shared_fixtures import TokenAssertions
from utils import (
    assert_valid_sql,
    assert_llm_response_valid,
    assert_sql_contains_tables,
)


class TestBasicSQLGeneration:
    """Test basic SQL generation from natural language."""

    @pytest.mark.requires_llm
    def test_generate_simple_select(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test generating a simple SELECT query."""
        prompt = """Generate a T-SQL query for: "Show all tickets from CentralTickets table"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), CustomerID (int), AddTicketDate (datetime)

Generate only the SQL query, no explanations.
SQL:"""

        max_tokens = 256
        response = llm_client.generate(
            prompt=prompt,
            model_type="sql",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Validate response
        assert_llm_response_valid(
            response,
            min_length=10,
            must_contain=["SELECT", "CentralTickets"],
        )

        # Validate SQL syntax
        assert_valid_sql(response.text)

        # Verify it's a SELECT
        assert "SELECT" in response.text.upper()
        assert "FROM CentralTickets" in response.text or "FROM [CentralTickets]" in response.text

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=10, max_tokens=500)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_generate_count_query(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test generating a COUNT aggregation query."""
        prompt = """Generate a T-SQL query for: "How many tickets are in the CentralTickets table?"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK)

Generate only the SQL query, no explanations.
SQL:"""

        max_tokens = 128
        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=max_tokens, temperature=0.0
        )

        assert_llm_response_valid(response, must_contain=["COUNT", "CentralTickets"])
        assert_valid_sql(response.text)

        # Should use COUNT(*)
        assert "COUNT" in response.text.upper()

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=10, max_tokens=300)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_generate_where_clause(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test generating query with WHERE clause."""
        prompt = """Generate a T-SQL query for: "Show tickets where CustomerID is 100"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), CustomerID (int), AddTicketDate (datetime)

Generate only the SQL query, no explanations.
SQL:"""

        max_tokens = 256
        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=max_tokens, temperature=0.0
        )

        assert_llm_response_valid(
            response, must_contain=["SELECT", "WHERE", "CustomerID"]
        )
        assert_valid_sql(response.text)

        # Should have WHERE clause
        assert "WHERE" in response.text.upper()
        assert "100" in response.text

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=10, max_tokens=400)
        token_assertions.assert_max_tokens_respected(response, max_tokens)


class TestSchemaGuidedGeneration:
    """Test SQL generation guided by database schema."""

    @pytest.mark.requires_llm
    def test_generate_with_column_names(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test that generated SQL uses correct column names from schema."""
        prompt = """Generate a T-SQL query for: "Show ticket ID and customer ID from tickets"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK), CustomerID (int), AddTicketDate (datetime)

IMPORTANT: Use the exact column names from the schema.

Generate only the SQL query, no explanations.
SQL:"""

        max_tokens = 256
        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=max_tokens, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should use exact column names
        text_upper = response.text.upper()
        assert "CENTRALTICKETID" in text_upper or "CENTRALTICKETID" in response.text
        assert "CUSTOMERID" in text_upper or "CUSTOMERID" in response.text

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=10, max_tokens=400)

    @pytest.mark.requires_llm
    def test_generate_join_query(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig,
        token_assertions: TokenAssertions
    ):
        """Test generating query with JOIN."""
        prompt = """Generate a T-SQL query for: "Show tickets with customer names"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK), CustomerID (int)
  Foreign Keys: CustomerID -> Customers.CustomerID

Table: Customers
  Columns: CustomerID (int, PK), CustomerName (nvarchar)

Generate only the SQL query, no explanations.
SQL:"""

        max_tokens = 512
        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=max_tokens, temperature=0.0
        )

        assert_llm_response_valid(
            response, must_contain=["SELECT", "JOIN", "CentralTickets", "Customers"]
        )
        assert_valid_sql(response.text)

        # Should have JOIN
        text_upper = response.text.upper()
        assert "JOIN" in text_upper
        assert_sql_contains_tables(
            response.text, ["CentralTickets", "Customers"]
        )

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=20, max_tokens=600)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_generate_with_primary_key(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test that generation uses primary key correctly."""
        prompt = """Generate a T-SQL query for: "Find ticket with ID 12345"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK), CustomerID (int)

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=256, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should filter by PK
        assert "CentralTicketID" in response.text or "CENTRALTICKETID" in response.text.upper()
        assert "12345" in response.text


class TestRuleGuidedGeneration:
    """Test SQL generation guided by rules."""

    @pytest.mark.requires_llm
    def test_generate_with_date_rule(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test generation following date comparison rule."""
        prompt = """Generate a T-SQL query for: "Show tickets created today"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), AddTicketDate (datetime)

RULES:
- For date comparisons, use CAST(column AS DATE) = CAST(GETDATE() AS DATE)
- The creation date column is AddTicketDate (NOT CreateDate)

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(
            response, must_contain=["SELECT", "AddTicketDate"]
        )
        assert_valid_sql(response.text)

        # Should use AddTicketDate
        assert "AddTicketDate" in response.text or "ADDTICKETDATE" in response.text.upper()

        # Should ideally use CAST for date comparison (soft check)
        text_upper = response.text.upper()
        has_cast = "CAST" in text_upper
        has_getdate = "GETDATE()" in text_upper

        # At minimum should have date comparison
        assert has_getdate or "DATEADD" in text_upper or "TODAY" in text_upper

    @pytest.mark.requires_llm
    def test_generate_with_column_name_rule(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test generation following column naming rule."""
        prompt = """Generate a T-SQL query for: "Show all tickets and their status"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), TicketStatusTypeID (int)

Table: Types
  Columns: TypeID (int), TypeName (nvarchar)

RULES:
- To get ticket status, JOIN with Types table using TicketStatusTypeID
- Use alias 'ts' for the Types table in status join
- Column is TicketStatusTypeID (NOT StatusID or TicketStatusID)

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should use correct column name
        assert "TicketStatusTypeID" in response.text or "TICKETSTATUSTYPEID" in response.text.upper()

        # Should JOIN with Types
        text_upper = response.text.upper()
        assert "JOIN" in text_upper
        assert "TYPES" in text_upper


class TestComplexSQLGeneration:
    """Test complex SQL generation scenarios."""

    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_generate_multi_table_join(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test generating complex multi-table JOIN query."""
        prompt = """Generate a T-SQL query for: "Show tickets with customer names and user names"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK), CustomerID (int), AddCentralUserID (int)
  Foreign Keys:
    CustomerID -> Customers.CustomerID
    AddCentralUserID -> Users.UserID

Table: Customers
  Columns: CustomerID (int, PK), CustomerName (nvarchar)

Table: Users
  Columns: UserID (int, PK), UserName (nvarchar)

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=1024, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should include all three tables
        assert_sql_contains_tables(
            response.text, ["CentralTickets", "Customers", "Users"]
        )

        # Should have multiple JOINs
        join_count = response.text.upper().count("JOIN")
        assert join_count >= 2, "Should have at least 2 JOINs"

    @pytest.mark.requires_llm
    def test_generate_group_by_query(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test generating GROUP BY aggregation query."""
        prompt = """Generate a T-SQL query for: "Count tickets by customer"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK), CustomerID (int)

Table: Customers
  Columns: CustomerID (int, PK), CustomerName (nvarchar)

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        text_upper = response.text.upper()
        # Should have COUNT and GROUP BY
        assert "COUNT" in text_upper
        assert "GROUP BY" in text_upper

    @pytest.mark.requires_llm
    def test_generate_order_by_query(
        self, llm_client: LocalLLMClient, pipeline_config: PipelineTestConfig
    ):
        """Test generating query with ORDER BY."""
        prompt = """Generate a T-SQL query for: "Show top 10 most recent tickets"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), AddTicketDate (datetime)

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        text_upper = response.text.upper()
        # Should have TOP and ORDER BY
        assert "TOP" in text_upper or "LIMIT" in text_upper
        assert "ORDER BY" in text_upper
        assert "DESC" in text_upper  # Most recent = descending


class TestLLMHealthAndErrors:
    """Test LLM endpoint health and error handling."""

    @pytest.mark.requires_llm
    def test_llm_endpoint_health(self, llm_client: LocalLLMClient):
        """Test that SQL LLM endpoint is healthy."""
        health = llm_client.health_check("sql")

        assert health["healthy"] is True
        assert health["endpoint"] == llm_client.sql_endpoint
        assert "models" in health

    @pytest.mark.requires_llm
    def test_empty_prompt_handling(self, llm_client: LocalLLMClient):
        """Test handling of empty prompt."""
        response = llm_client.generate(
            prompt="", model_type="sql", max_tokens=128, temperature=0.0
        )

        # Empty prompt might return empty or error
        # LLM should still respond (even if with empty/error)
        assert response is not None

    @pytest.mark.requires_llm
    def test_very_long_prompt(self, llm_client: LocalLLMClient):
        """Test handling of very long prompt."""
        # Create a very long prompt
        long_schema = "\n".join(
            [
                f"Table: Table{i}\n  Columns: Col1, Col2, Col3"
                for i in range(100)
            ]
        )

        prompt = f"""Generate a T-SQL query for: "Show all data"

DATABASE: EWRCentral
SCHEMA:
{long_schema}

Generate only the SQL query.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        # Should handle gracefully (might truncate or summarize)
        assert response is not None
        # If successful, should still be valid SQL
        if response.success and response.text:
            assert_valid_sql(response.text)

    @pytest.mark.requires_llm
    def test_invalid_sql_syntax_in_prompt(self, llm_client: LocalLLMClient):
        """Test that LLM can correct invalid examples in prompt."""
        prompt = """Generate a T-SQL query for: "Count tickets"

DATABASE: EWRCentral
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int)

BAD EXAMPLE (don't use this): SELCT COUNT(*) FORM CentralTickets

Generate the CORRECT SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=256, temperature=0.0
        )

        if response.success:
            # Should generate correct SQL despite bad example
            assert_valid_sql(response.text)
            text_upper = response.text.upper()
            assert "SELECT" in text_upper  # Not "SELCT"
            assert "FROM" in text_upper  # Not "FORM"
