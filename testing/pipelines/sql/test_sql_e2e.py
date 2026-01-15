"""
SQL Pipeline End-to-End Tests.

Tests complete SQL pipeline flow:
question → cache check → rule matching → schema loading → LLM generation → validation → storage

These tests exercise the full pipeline with all components integrated.
Uses local llama.cpp (port 8080) and MongoDB only.
"""

import pytest
import time

from config.test_config import PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.mongodb_fixtures import (
    create_mock_sql_query,
    cleanup_test_documents,
)
from utils import assert_valid_sql, assert_llm_response_valid


class TestSQLPipelineE2E:
    """End-to-end tests for complete SQL pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_cache_miss_to_generation(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete pipeline flow: cache miss → LLM generation → storage.

        Flow:
        1. Check cache (miss)
        2. Generate SQL with LLM
        3. Store result in agent_learning
        4. Verify stored correctly
        """
        collection = mongodb_database["agent_learning"]
        question = "How many tickets were created in the last 7 days?"
        database = "EWRCentral"

        # Step 1: Check cache (should miss)
        cached = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": database,
                "success": True,
            }
        )
        assert cached is None, "Cache should be empty initially"

        # Step 2: Generate SQL with LLM
        prompt = f"""Generate a T-SQL query for: "{question}"

DATABASE: {database}
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int, PK), AddTicketDate (datetime)

RULES:
- Use AddTicketDate for creation date
- For last 7 days: AddTicketDate >= DATEADD(DAY, -7, GETDATE())

Generate only the SQL query, no explanations.
SQL:"""

        start_time = time.time()
        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )
        generation_time = time.time() - start_time

        # Validate LLM response
        assert_llm_response_valid(
            response, must_contain=["SELECT", "CentralTickets"]
        )
        assert_valid_sql(response.text)

        # Step 3: Store result
        doc = create_mock_sql_query(
            question=question, sql=response.text, database=database, success=True
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["confidence"] = 0.85
        doc["processing_time"] = generation_time
        doc["matched_rules"] = []
        doc["rule_match_type"] = "none"

        collection.insert_one(doc)

        # Step 4: Verify storage
        stored = collection.find_one({"_id": doc["_id"]})
        assert stored is not None
        assert stored["question"] == question
        assert stored["sql"] == response.text
        assert stored["success"] is True

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_full_pipeline_cache_hit(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test pipeline with cache hit (bypass LLM).

        Flow:
        1. Pre-populate cache
        2. Query same question
        3. Return cached SQL (no LLM call)
        """
        collection = mongodb_database["agent_learning"]
        question = "Show all tickets created today"
        database = "EWRCentral"
        cached_sql = "SELECT * FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)"

        # Pre-populate cache
        doc = create_mock_sql_query(
            question=question, sql=cached_sql, database=database, success=True
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["confidence"] = 1.0
        collection.insert_one(doc)

        # Query (should hit cache)
        start_time = time.time()
        result = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": database,
                "success": True,
            }
        )
        cache_time = time.time() - start_time

        # Verify cache hit
        assert result is not None
        assert result["sql"] == cached_sql
        assert cache_time < 0.1, "Cache should be fast (<100ms)"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_pipeline_with_schema_context(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test pipeline with schema-guided generation.

        Flow:
        1. Load schema for tables
        2. Generate SQL with schema context
        3. Verify correct column names used
        """
        question = "Show ticket ID, customer ID, and creation date"
        database = "EWRCentral"

        schema = """Table: CentralTickets
  Columns:
    - CentralTicketID (int, PK)
    - CustomerID (int, FK -> Customers.CustomerID)
    - AddTicketDate (datetime)
    - TicketStatusTypeID (int)"""

        prompt = f"""Generate a T-SQL query for: "{question}"

DATABASE: {database}
SCHEMA:
{schema}

IMPORTANT: Use exact column names from schema.

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should use schema column names
        text_upper = response.text.upper()
        assert "CENTRALTICKETID" in text_upper or "CentralTicketID" in response.text
        assert "CUSTOMERID" in text_upper or "CustomerID" in response.text
        assert "ADDTICKETDATE" in text_upper or "AddTicketDate" in response.text

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_pipeline_with_rule_matching(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test pipeline with rule-based guidance.

        Flow:
        1. Match relevant rules for question
        2. Include rules in LLM prompt
        3. Generate SQL following rules
        4. Store with matched rules
        """
        collection = mongodb_database["agent_learning"]
        question = "Show tickets created today"
        database = "EWRCentral"

        # Simulate matched rules
        matched_rules = [
            {
                "rule_id": "use-addticketdate-not-createdate",
                "description": "Use AddTicketDate column for creation date",
                "rule_text": "The ticket creation date is stored in AddTicketDate, not CreateDate",
            },
            {
                "rule_id": "use-cast-for-date-comparison",
                "description": "Use CAST for date comparisons",
                "rule_text": "For date-only comparisons, use CAST(column AS DATE) = CAST(GETDATE() AS DATE)",
            },
        ]

        # Build prompt with rules
        rules_text = "\n".join(
            [f"- {r['rule_text']}" for r in matched_rules]
        )

        prompt = f"""Generate a T-SQL query for: "{question}"

DATABASE: {database}
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), AddTicketDate (datetime)

RULES:
{rules_text}

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should use AddTicketDate (not CreateDate)
        assert "AddTicketDate" in response.text or "ADDTICKETDATE" in response.text.upper()

        # Store with matched rules
        doc = create_mock_sql_query(
            question=question, sql=response.text, database=database, success=True
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["matched_rules"] = [r["rule_id"] for r in matched_rules]
        doc["rule_match_type"] = "keyword"
        collection.insert_one(doc)

        # Verify rules stored
        stored = collection.find_one({"_id": doc["_id"]})
        assert len(stored["matched_rules"]) == 2
        assert "use-addticketdate-not-createdate" in stored["matched_rules"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_pipeline_iterative_improvement(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test pipeline learning from feedback.

        Flow:
        1. Generate initial SQL
        2. Store with success=False (bad SQL)
        3. Regenerate with corrections
        4. Store with success=True
        5. Future queries should use successful version
        """
        collection = mongodb_database["agent_learning"]
        question = "Count tickets by status"
        database = "EWRCentral"

        # First attempt - wrong column name
        bad_sql = "SELECT StatusID, COUNT(*) FROM CentralTickets GROUP BY StatusID"
        doc_bad = create_mock_sql_query(
            question=question, sql=bad_sql, database=database, success=False
        )
        doc_bad["test_run_id"] = pipeline_config.test_run_id
        doc_bad["error"] = "Invalid column name 'StatusID'"
        collection.insert_one(doc_bad)

        # Second attempt - with correction
        prompt = f"""Generate a T-SQL query for: "{question}"

DATABASE: {database}
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), TicketStatusTypeID (int)

PREVIOUS ERROR: Invalid column name 'StatusID'
CORRECTION: Use TicketStatusTypeID, not StatusID

Generate only the SQL query, no explanations.
SQL:"""

        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=512, temperature=0.0
        )

        assert_llm_response_valid(response)
        assert_valid_sql(response.text)

        # Should use correct column
        assert "TicketStatusTypeID" in response.text or "TICKETSTATUSTYPEID" in response.text.upper()

        # Store successful version
        doc_good = create_mock_sql_query(
            question=question, sql=response.text, database=database, success=True
        )
        doc_good["test_run_id"] = pipeline_config.test_run_id
        doc_good["confidence"] = 0.9
        collection.insert_one(doc_good)

        # Future query should get successful version
        cached = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": database,
                "success": True,
            }
        )
        assert cached is not None
        assert cached["sql"] == response.text

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSQLPipelinePerformance:
    """Test performance characteristics of SQL pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_pipeline_generation_time(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """Test that SQL generation completes within reasonable time."""
        question = "Show top 10 customers by ticket count"
        database = "EWRCentral"

        prompt = f"""Generate a T-SQL query for: "{question}"

DATABASE: {database}
SCHEMA:
Table: CentralTickets
  Columns: CentralTicketID (int), CustomerID (int)
Table: Customers
  Columns: CustomerID (int), CustomerName (nvarchar)

Generate only the SQL query, no explanations.
SQL:"""

        start_time = time.time()
        response = llm_client.generate(
            prompt=prompt, model_type="sql", max_tokens=1024, temperature=0.0
        )
        generation_time = time.time() - start_time

        # Should complete in reasonable time (depends on hardware)
        # Typical: 1-10 seconds for local llama.cpp
        assert generation_time < 60, f"Generation too slow: {generation_time:.2f}s"

        if response.success:
            assert_valid_sql(response.text)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_cache_retrieval_performance(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that cache retrieval is fast (<100ms)."""
        collection = mongodb_database["agent_learning"]

        # Pre-populate with many entries
        docs = [
            create_mock_sql_query(
                question=f"Query {i}",
                sql=f"SELECT {i}",
                database="EWRCentral",
                success=True,
            )
            for i in range(100)
        ]

        for doc in docs:
            doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_many(docs)

        # Time cache lookup
        target = "Query 50"
        start_time = time.time()
        result = collection.find_one(
            {
                "question_normalized": target.lower().strip(),
                "database": "EWRCentral",
                "success": True,
            }
        )
        lookup_time = time.time() - start_time

        assert result is not None
        assert lookup_time < 0.1, f"Cache lookup too slow: {lookup_time:.3f}s"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSQLPipelineErrorHandling:
    """Test error handling in SQL pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_llm
    def test_llm_timeout_handling(self, llm_client: LocalLLMClient):
        """Test handling of LLM timeout."""
        # Create client with very short timeout
        from fixtures.llm_fixtures import LocalLLMClient as TimeoutClient

        timeout_client = TimeoutClient(timeout=1)  # 1 second timeout

        # Very complex prompt that might timeout
        complex_prompt = "Generate SQL query:\n" + ("Schema details...\n" * 1000)

        response = timeout_client.generate(
            prompt=complex_prompt, model_type="sql", max_tokens=2048
        )

        # Should either succeed or return timeout error (not crash)
        assert response is not None
        if not response.success:
            assert response.error is not None

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_invalid_database_handling(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test handling of invalid database name."""
        collection = mongodb_database["agent_learning"]
        question = "Show data"
        invalid_database = "NonExistentDB"

        # Store failed query for invalid database
        doc = create_mock_sql_query(
            question=question,
            sql="SELECT * FROM SomeTable",
            database=invalid_database,
            success=False,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["error"] = "Cannot open database 'NonExistentDB'"
        collection.insert_one(doc)

        # Verify error stored
        stored = collection.find_one({"_id": doc["_id"]})
        assert stored["success"] is False
        assert "error" in stored

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)
