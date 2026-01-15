"""
SQL Storage Tests for agent_learning collection.

Tests SQL query storage operations including:
- Storing successful queries
- Storing failed queries with error information
- Updating existing queries with feedback
- Retrieving stored queries for cache hits
- Data validation and schema enforcement

All tests use MongoDB only (no external dependencies).
All test data is prefixed with 'test_' for isolation.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_sql_query,
    insert_test_documents,
    cleanup_test_documents,
)
from utils import (
    assert_document_stored,
    assert_mongodb_document,
    generate_test_id,
)


class TestSQLStorage:
    """Test SQL query storage operations in agent_learning collection."""

    @pytest.mark.requires_mongodb
    def test_store_successful_query(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing a successful SQL query result."""
        collection = mongodb_database["agent_learning"]

        # Create test query document
        doc = create_mock_sql_query(
            question="How many tickets were created today?",
            sql="SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
            database="EWRCentral",
            success=True,
        )

        # Add test markers
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["confidence"] = 0.95
        doc["processing_time"] = 1.23

        # Insert document
        result = collection.insert_one(doc)
        doc_id = doc["_id"]

        # Verify stored
        stored_doc = assert_document_stored(
            collection,
            doc_id,
            expected_fields=[
                "question",
                "question_normalized",
                "sql",
                "database",
                "success",
                "confidence",
            ],
        )

        # Validate structure
        assert stored_doc["question"] == doc["question"]
        assert stored_doc["question_normalized"] == doc["question"].lower().strip()
        assert stored_doc["sql"] == doc["sql"]
        assert stored_doc["database"] == "EWRCentral"
        assert stored_doc["success"] is True
        assert stored_doc["confidence"] == 0.95

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_failed_query(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing a failed SQL query with error information."""
        collection = mongodb_database["agent_learning"]

        # Create failed query document
        doc = create_mock_sql_query(
            question="Show me invalid data",
            sql="SELECT * FROM NonExistentTable",
            database="EWRCentral",
            success=False,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["error"] = "Invalid object name 'NonExistentTable'."
        doc["confidence"] = 0.5

        # Insert document
        collection.insert_one(doc)
        doc_id = doc["_id"]

        # Verify stored with error
        stored_doc = assert_document_stored(collection, doc_id)
        assert stored_doc["success"] is False
        assert "error" in stored_doc
        assert "NonExistentTable" in stored_doc["error"]
        assert stored_doc["confidence"] < 0.8

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_query_with_matched_rules(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing query with matched rule information."""
        collection = mongodb_database["agent_learning"]

        doc = create_mock_sql_query(
            question="Show tickets created today",
            sql="SELECT * FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["matched_rules"] = [
            "use-addticketdate-not-createdate",
            "use-cast-for-date-comparison",
        ]
        doc["rule_match_type"] = "keyword"

        collection.insert_one(doc)

        # Verify rules stored
        stored_doc = assert_document_stored(collection, doc["_id"])
        assert "matched_rules" in stored_doc
        assert len(stored_doc["matched_rules"]) == 2
        assert "use-addticketdate-not-createdate" in stored_doc["matched_rules"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_multiple_queries_batch(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test bulk storage of multiple SQL queries."""
        collection = mongodb_database["agent_learning"]

        # Create multiple test queries
        queries = [
            create_mock_sql_query(
                question=f"Test question {i}",
                sql=f"SELECT * FROM Table{i}",
                database="EWRCentral",
                success=True,
            )
            for i in range(10)
        ]

        # Add test markers
        for query in queries:
            query["test_run_id"] = pipeline_config.test_run_id

        # Bulk insert
        doc_ids = insert_test_documents(
            collection, queries, pipeline_config.test_run_id
        )

        # Verify all stored
        assert len(doc_ids) == 10

        # Verify count
        count = collection.count_documents(
            {"test_run_id": pipeline_config.test_run_id}
        )
        assert count == 10

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_update_query_with_feedback(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test updating a stored query with user feedback."""
        collection = mongodb_database["agent_learning"]

        # Store initial query
        doc = create_mock_sql_query(
            question="Count tickets",
            sql="SELECT COUNT(*) FROM CentralTickets",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        # Update with feedback
        collection.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "user_feedback": "correct",
                    "feedback_timestamp": datetime.utcnow(),
                    "confidence": 1.0,
                }
            },
        )

        # Verify updated
        updated_doc = collection.find_one({"_id": doc["_id"]})
        assert updated_doc["user_feedback"] == "correct"
        assert updated_doc["confidence"] == 1.0
        assert "feedback_timestamp" in updated_doc

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_query_schema_validation(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that stored queries conform to expected schema."""
        collection = mongodb_database["agent_learning"]

        doc = create_mock_sql_query(
            question="Test query",
            sql="SELECT 1",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        stored_doc = collection.find_one({"_id": doc["_id"]})

        # Define expected schema
        schema = {
            "question": str,
            "question_normalized": str,
            "sql": str,
            "database": str,
            "success": bool,
            "is_test": bool,
            "test_marker": bool,
        }

        # Validate schema
        assert_mongodb_document(stored_doc, schema, allow_extra=True)

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_query_with_execution_results(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing query with execution results metadata."""
        collection = mongodb_database["agent_learning"]

        doc = create_mock_sql_query(
            question="Get ticket count",
            sql="SELECT COUNT(*) as TicketCount FROM CentralTickets",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["execution_result"] = {
            "row_count": 1,
            "columns": ["TicketCount"],
            "execution_time_ms": 15,
        }

        collection.insert_one(doc)

        # Verify execution result stored
        stored_doc = assert_document_stored(collection, doc["_id"])
        assert "execution_result" in stored_doc
        assert stored_doc["execution_result"]["row_count"] == 1
        assert "TicketCount" in stored_doc["execution_result"]["columns"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_normalized_question_indexing(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that normalized questions enable cache hits."""
        collection = mongodb_database["agent_learning"]

        # Store with normalized question
        doc1 = create_mock_sql_query(
            question="  Show me TICKETS created TODAY  ",  # Extra spaces, mixed case
            sql="SELECT * FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
            database="EWRCentral",
            success=True,
        )
        doc1["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc1)

        # Query with different formatting but same normalized form
        normalized = "show me tickets created today"
        result = collection.find_one(
            {
                "question_normalized": normalized,
                "database": "EWRCentral",
                "success": True,
            }
        )

        # Should find the match
        assert result is not None
        assert result["_id"] == doc1["_id"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSQLStorageEdgeCases:
    """Test edge cases and error conditions for SQL storage."""

    @pytest.mark.requires_mongodb
    def test_store_query_with_special_characters(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing queries with special characters in SQL."""
        collection = mongodb_database["agent_learning"]

        doc = create_mock_sql_query(
            question="Find customer 'O'Brien'",
            sql="SELECT * FROM Customers WHERE Name = 'O''Brien'",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        stored_doc = assert_document_stored(collection, doc["_id"])
        assert "O''Brien" in stored_doc["sql"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_large_sql_query(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing very large SQL queries."""
        collection = mongodb_database["agent_learning"]

        # Generate large SQL (multiple JOINs)
        large_sql = """
        SELECT ct.*, tt.TypeName as TicketType, ts.TypeName as Status,
               c.CustomerName, u.UserName
        FROM CentralTickets ct
        LEFT JOIN Types tt ON ct.TicketTypeID = tt.TypeID
        LEFT JOIN Types ts ON ct.TicketStatusTypeID = ts.TypeID
        LEFT JOIN Customers c ON ct.CustomerID = c.CustomerID
        LEFT JOIN Users u ON ct.AddCentralUserID = u.UserID
        WHERE CAST(ct.AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)
        """ * 10  # Repeat to make it large

        doc = create_mock_sql_query(
            question="Complex ticket query",
            sql=large_sql,
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        stored_doc = assert_document_stored(collection, doc["_id"])
        assert len(stored_doc["sql"]) > 1000

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_duplicate_question_handling(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test handling duplicate questions (cache scenario)."""
        collection = mongodb_database["agent_learning"]

        question = "How many tickets today?"
        sql1 = "SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)"
        sql2 = "SELECT COUNT(*) as TicketCount FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)"

        # Store first version
        doc1 = create_mock_sql_query(
            question=question, sql=sql1, database="EWRCentral", success=True
        )
        doc1["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc1)

        # Store second version (different SQL for same question)
        doc2 = create_mock_sql_query(
            question=question, sql=sql2, database="EWRCentral", success=True
        )
        doc2["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc2)

        # Both should be stored (learning multiple approaches)
        count = collection.count_documents(
            {
                "question_normalized": question.lower().strip(),
                "test_run_id": pipeline_config.test_run_id,
            }
        )
        assert count == 2

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)
