"""
SQL Retrieval Tests for cached queries and rule matching.

Tests SQL query retrieval operations including:
- Cache hit/miss scenarios
- Exact question matching
- Similar question matching
- Rule-based query retrieval
- Database-scoped retrieval
- Confidence-based filtering

All tests use MongoDB only (no external dependencies).
"""

import pytest
from datetime import datetime, timedelta

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_sql_query,
    insert_test_documents,
    cleanup_test_documents,
)
from utils import generate_test_id


class TestSQLCacheRetrieval:
    """Test SQL query cache retrieval operations."""

    @pytest.mark.requires_mongodb
    def test_exact_question_cache_hit(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test exact question match returns cached SQL."""
        collection = mongodb_database["agent_learning"]

        # Store successful query
        question = "How many tickets were created today?"
        sql = "SELECT COUNT(*) FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)"

        doc = create_mock_sql_query(
            question=question, sql=sql, database="EWRCentral", success=True
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        # Retrieve with exact question
        result = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": "EWRCentral",
                "success": True,
            }
        )

        assert result is not None
        assert result["sql"] == sql
        assert result["success"] is True

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_cache_miss_different_database(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that cache is database-scoped (same question, different DB)."""
        collection = mongodb_database["agent_learning"]

        question = "Count all records"
        sql_central = "SELECT COUNT(*) FROM CentralTickets"
        sql_gin = "SELECT COUNT(*) FROM Bales"

        # Store for EWRCentral
        doc1 = create_mock_sql_query(
            question=question, sql=sql_central, database="EWRCentral", success=True
        )
        doc1["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc1)

        # Store for Gin database
        doc2 = create_mock_sql_query(
            question=question, sql=sql_gin, database="EWR.Gin.Test", success=True
        )
        doc2["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc2)

        # Retrieve for EWRCentral
        result_central = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": "EWRCentral",
                "success": True,
            }
        )

        # Retrieve for Gin
        result_gin = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": "EWR.Gin.Test",
                "success": True,
            }
        )

        # Should return different SQL for different databases
        assert result_central["sql"] == sql_central
        assert result_gin["sql"] == sql_gin
        assert result_central["sql"] != result_gin["sql"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_cache_ignores_failed_queries(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that failed queries are not returned from cache."""
        collection = mongodb_database["agent_learning"]

        question = "Show invalid data"

        # Store failed query
        doc_fail = create_mock_sql_query(
            question=question,
            sql="SELECT * FROM InvalidTable",
            database="EWRCentral",
            success=False,
        )
        doc_fail["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc_fail)

        # Try to retrieve (should filter out failed)
        result = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": "EWRCentral",
                "success": True,  # Only successful queries
            }
        )

        # Should not find failed query
        assert result is None

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_retrieve_most_recent_on_duplicates(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieving most recent when multiple matches exist."""
        collection = mongodb_database["agent_learning"]

        question = "Count tickets"

        # Store older query
        doc_old = create_mock_sql_query(
            question=question,
            sql="SELECT COUNT(*) FROM CentralTickets",
            database="EWRCentral",
            success=True,
        )
        doc_old["test_run_id"] = pipeline_config.test_run_id
        doc_old["created_at"] = datetime.utcnow() - timedelta(days=7)
        collection.insert_one(doc_old)

        # Store newer query with better SQL
        doc_new = create_mock_sql_query(
            question=question,
            sql="SELECT COUNT(*) as TicketCount FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
            database="EWRCentral",
            success=True,
        )
        doc_new["test_run_id"] = pipeline_config.test_run_id
        doc_new["created_at"] = datetime.utcnow()
        collection.insert_one(doc_new)

        # Retrieve most recent
        result = collection.find_one(
            {
                "question_normalized": question.lower().strip(),
                "database": "EWRCentral",
                "success": True,
            },
            sort=[("created_at", -1)],  # Most recent first
        )

        assert result is not None
        assert result["_id"] == doc_new["_id"]
        assert "CAST(GETDATE() AS DATE)" in result["sql"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_case_insensitive_question_matching(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that question matching is case-insensitive."""
        collection = mongodb_database["agent_learning"]

        # Store with lowercase
        doc = create_mock_sql_query(
            question="show me tickets created today",
            sql="SELECT * FROM CentralTickets",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        # Query with different case
        variations = [
            "SHOW ME TICKETS CREATED TODAY",
            "Show Me Tickets Created Today",
            "sHoW mE tIcKeTs CrEaTeD tOdAy",
        ]

        for variation in variations:
            result = collection.find_one(
                {
                    "question_normalized": variation.lower().strip(),
                    "database": "EWRCentral",
                    "success": True,
                }
            )
            assert result is not None, f"Failed to match: {variation}"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_whitespace_normalization(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that extra whitespace is normalized."""
        collection = mongodb_database["agent_learning"]

        # Store with normal spacing
        doc = create_mock_sql_query(
            question="show me tickets",
            sql="SELECT * FROM CentralTickets",
            database="EWRCentral",
            success=True,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc)

        # Query with extra whitespace
        variations = [
            "  show   me   tickets  ",
            "\tshow\tme\ttickets\t",
            "show me tickets",  # Normal
        ]

        for variation in variations:
            normalized = " ".join(variation.lower().split())
            result = collection.find_one(
                {
                    "question_normalized": normalized,
                    "database": "EWRCentral",
                    "success": True,
                }
            )
            assert result is not None, f"Failed to match: {repr(variation)}"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSQLRuleBasedRetrieval:
    """Test rule-based query retrieval patterns."""

    @pytest.mark.requires_mongodb
    def test_retrieve_queries_by_matched_rules(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieving queries that used specific rules."""
        collection = mongodb_database["agent_learning"]

        rule_id = "use-addticketdate-not-createdate"

        # Store queries with rule
        docs = [
            create_mock_sql_query(
                question=f"Query {i}",
                sql=f"SELECT * FROM CentralTickets WHERE AddTicketDate > '{i}'",
                database="EWRCentral",
                success=True,
            )
            for i in range(5)
        ]

        for doc in docs:
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["matched_rules"] = [rule_id]

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Retrieve by rule
        results = list(
            collection.find(
                {
                    "matched_rules": rule_id,
                    "test_run_id": pipeline_config.test_run_id,
                }
            )
        )

        assert len(results) == 5
        for result in results:
            assert rule_id in result["matched_rules"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_retrieve_queries_with_high_confidence(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test filtering queries by confidence threshold."""
        collection = mongodb_database["agent_learning"]

        # Store queries with varying confidence
        queries = []
        for i, confidence in enumerate([0.5, 0.7, 0.85, 0.95, 0.99]):
            doc = create_mock_sql_query(
                question=f"Query {i}",
                sql=f"SELECT {i}",
                database="EWRCentral",
                success=True,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["confidence"] = confidence
            queries.append(doc)

        insert_test_documents(collection, queries, pipeline_config.test_run_id)

        # Retrieve high confidence (>= 0.8)
        high_confidence = list(
            collection.find(
                {
                    "test_run_id": pipeline_config.test_run_id,
                    "confidence": {"$gte": 0.8},
                }
            )
        )

        assert len(high_confidence) == 3  # 0.85, 0.95, 0.99
        for doc in high_confidence:
            assert doc["confidence"] >= 0.8

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_retrieve_queries_with_feedback(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieving queries that received positive feedback."""
        collection = mongodb_database["agent_learning"]

        # Store queries with different feedback
        doc_positive = create_mock_sql_query(
            question="Good query",
            sql="SELECT * FROM CentralTickets",
            database="EWRCentral",
            success=True,
        )
        doc_positive["test_run_id"] = pipeline_config.test_run_id
        doc_positive["user_feedback"] = "correct"
        collection.insert_one(doc_positive)

        doc_negative = create_mock_sql_query(
            question="Bad query",
            sql="SELECT * FROM WrongTable",
            database="EWRCentral",
            success=True,
        )
        doc_negative["test_run_id"] = pipeline_config.test_run_id
        doc_negative["user_feedback"] = "incorrect"
        collection.insert_one(doc_negative)

        doc_no_feedback = create_mock_sql_query(
            question="No feedback",
            sql="SELECT 1",
            database="EWRCentral",
            success=True,
        )
        doc_no_feedback["test_run_id"] = pipeline_config.test_run_id
        collection.insert_one(doc_no_feedback)

        # Retrieve only positive feedback
        positive_results = list(
            collection.find(
                {
                    "test_run_id": pipeline_config.test_run_id,
                    "user_feedback": "correct",
                }
            )
        )

        assert len(positive_results) == 1
        assert positive_results[0]["_id"] == doc_positive["_id"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSQLRetrievalPerformance:
    """Test performance-related aspects of SQL retrieval."""

    @pytest.mark.requires_mongodb
    def test_retrieve_from_large_dataset(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieval performance with large dataset."""
        collection = mongodb_database["agent_learning"]

        # Insert large batch of queries
        queries = [
            create_mock_sql_query(
                question=f"Test query number {i}",
                sql=f"SELECT {i}",
                database="EWRCentral",
                success=True,
            )
            for i in range(100)
        ]

        for doc in queries:
            doc["test_run_id"] = pipeline_config.test_run_id

        insert_test_documents(collection, queries, pipeline_config.test_run_id)

        # Retrieve specific query
        target_question = "Test query number 50"
        result = collection.find_one(
            {
                "question_normalized": target_question.lower().strip(),
                "database": "EWRCentral",
                "success": True,
            }
        )

        assert result is not None
        assert target_question in result["question"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_retrieve_with_index_hint(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieval using database indexes."""
        collection = mongodb_database["agent_learning"]

        # Create index on normalized question + database
        collection.create_index(
            [("question_normalized", 1), ("database", 1), ("success", 1)]
        )

        # Store test queries
        docs = [
            create_mock_sql_query(
                question=f"Indexed query {i}",
                sql=f"SELECT {i}",
                database="EWRCentral",
                success=True,
            )
            for i in range(50)
        ]

        for doc in docs:
            doc["test_run_id"] = pipeline_config.test_run_id

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Query using index
        result = collection.find_one(
            {
                "question_normalized": "indexed query 25",
                "database": "EWRCentral",
                "success": True,
            }
        )

        assert result is not None
        assert "25" in result["sql"]

        # Cleanup index
        collection.drop_index(
            [("question_normalized", 1), ("database", 1), ("success", 1)]
        )
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)
