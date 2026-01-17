"""
SQL End-to-End Tests - Using Real Data
=======================================

Tests SQL pipeline end-to-end functionality using REAL data.
Tests verify the complete data flow from storage to retrieval.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    verify_database_has_data,
    get_real_sql_example,
    get_real_sql_examples,
    get_real_sql_rule,
    get_real_stored_procedure,
    get_sql_test_questions,
)


class TestSQLPipelineDataFlow:
    """Test complete SQL data flow."""

    @pytest.mark.requires_mongodb
    def test_database_has_required_collections(self, mongodb_database):
        """Verify all required SQL collections exist with data."""
        required_collections = [
            "sql_examples",
            "sql_rules",
            "sql_stored_procedures",
            "sql_schema_context",
        ]

        for col_name in required_collections:
            collection = mongodb_database[col_name]
            count = collection.count_documents({})
            assert count > 0, f"{col_name} should have documents"

    @pytest.mark.requires_mongodb
    def test_sql_example_to_retrieval_flow(self, mongodb_database):
        """Test that stored SQL examples can be retrieved."""
        # Store reference
        example = get_real_sql_example(mongodb_database)
        assert example is not None

        # Retrieve by ID
        collection = mongodb_database["sql_examples"]
        retrieved = collection.find_one({"_id": example["_id"]})

        assert retrieved is not None
        assert retrieved["_id"] == example["_id"]
        assert retrieved.get("sql") == example.get("sql")

    @pytest.mark.requires_mongodb
    def test_rule_lookup_flow(self, mongodb_database):
        """Test that rules can be looked up for SQL generation."""
        # Get a rule
        rule = get_real_sql_rule(mongodb_database)
        assert rule is not None

        # Should be able to retrieve it
        collection = mongodb_database["sql_rules"]
        retrieved = collection.find_one({"_id": rule["_id"]})
        assert retrieved is not None

    @pytest.mark.requires_mongodb
    def test_schema_context_lookup(self, mongodb_database):
        """Test schema context can be retrieved for query generation."""
        collection = mongodb_database["sql_schema_context"]

        # Get a schema entry
        schema = collection.find_one()
        assert schema is not None

        # Should be queryable by table name if present
        if "table_name" in schema:
            result = collection.find_one({"table_name": schema["table_name"]})
            assert result is not None


class TestSQLDataIntegrity:
    """Test SQL data integrity across collections."""

    @pytest.mark.requires_mongodb
    def test_sql_examples_have_valid_sql(self, mongodb_database):
        """Verify SQL examples contain valid SQL statements."""
        examples = get_real_sql_examples(mongodb_database, limit=20)

        for example in examples:
            sql = example.get("sql", "")
            assert isinstance(sql, str)
            assert len(sql) > 0

            # Should contain SQL keywords
            sql_upper = sql.upper()
            has_sql_keyword = any(
                kw in sql_upper
                for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "EXEC"]
            )
            assert has_sql_keyword, f"SQL should have valid keyword: {sql[:50]}"

    @pytest.mark.requires_mongodb
    def test_stored_procedures_have_definition(self, mongodb_database):
        """Verify stored procedures have procedure definitions."""
        collection = mongodb_database["sql_stored_procedures"]

        # Check a sample of procedures
        procedures = list(collection.find().limit(10))

        for proc in procedures:
            assert "_id" in proc
            # Procedures should have some identifying information
            assert any(
                field in proc
                for field in ["procedure_name", "name", "definition", "sql"]
            )

    @pytest.mark.requires_mongodb
    def test_schema_context_has_structure(self, mongodb_database):
        """Verify schema context documents have proper structure."""
        collection = mongodb_database["sql_schema_context"]

        docs = list(collection.find().limit(10))

        for doc in docs:
            assert "_id" in doc
            # Should have table or database info
            assert any(
                field in doc
                for field in ["table_name", "database", "columns", "schema"]
            )


class TestSQLQueryPatterns:
    """Test common SQL query patterns work with real data."""

    @pytest.mark.requires_mongodb
    def test_query_by_database_name(self, mongodb_database):
        """Test querying multiple collections by database name."""
        # Find a database name that exists
        examples_col = mongodb_database["sql_examples"]
        databases = examples_col.distinct("database")

        if databases:
            db_name = databases[0]

            # Should be able to query examples by database
            examples = list(examples_col.find({"database": db_name}).limit(5))
            assert len(examples) > 0

    @pytest.mark.requires_mongodb
    def test_text_search_in_rules(self, mongodb_database):
        """Test text search capability in rules."""
        collection = mongodb_database["sql_rules"]

        # Search for common terms
        results = list(
            collection.find(
                {"description": {"$regex": "table", "$options": "i"}}
            ).limit(5)
        )

        # May or may not find results, just verify query works
        assert isinstance(results, list)

    @pytest.mark.requires_mongodb
    def test_aggregation_pipeline_works(self, mongodb_database):
        """Test that aggregation pipelines work on SQL data."""
        collection = mongodb_database["sql_examples"]

        pipeline = [
            {"$group": {"_id": "$database", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 0}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0


class TestSQLTestQuestions:
    """Test SQL test question generation from real data."""

    @pytest.mark.requires_mongodb
    def test_generate_test_questions(self, mongodb_database):
        """Test that test questions can be generated from real examples."""
        questions = get_sql_test_questions(mongodb_database)

        assert len(questions) > 0

        for q in questions:
            assert "database" in q or "question" in q or "expected_sql_pattern" in q

    @pytest.mark.requires_mongodb
    def test_questions_have_expected_structure(self, mongodb_database):
        """Test that generated questions have expected structure."""
        questions = get_sql_test_questions(mongodb_database)

        for q in questions[:5]:  # Check first 5
            # Should have some identifying information
            assert isinstance(q, dict)
            assert len(q) > 0


class TestSQLCollectionIndexes:
    """Test that SQL collections have proper indexes."""

    @pytest.mark.requires_mongodb
    def test_sql_examples_has_indexes(self, mongodb_database):
        """Verify sql_examples collection has indexes."""
        collection = mongodb_database["sql_examples"]
        indexes = list(collection.list_indexes())

        # Should have at least _id index
        index_names = [idx["name"] for idx in indexes]
        assert "_id_" in index_names

    @pytest.mark.requires_mongodb
    def test_sql_rules_has_indexes(self, mongodb_database):
        """Verify sql_rules collection has indexes."""
        collection = mongodb_database["sql_rules"]
        indexes = list(collection.list_indexes())

        assert len(indexes) >= 1  # At least _id

    @pytest.mark.requires_mongodb
    def test_stored_procedures_has_indexes(self, mongodb_database):
        """Verify sql_stored_procedures has indexes."""
        collection = mongodb_database["sql_stored_procedures"]
        indexes = list(collection.list_indexes())

        index_names = [idx["name"] for idx in indexes]
        assert "_id_" in index_names

    @pytest.mark.requires_mongodb
    def test_schema_context_has_indexes(self, mongodb_database):
        """Verify sql_schema_context has indexes."""
        collection = mongodb_database["sql_schema_context"]
        indexes = list(collection.list_indexes())

        assert len(indexes) >= 1


class TestSQLDataCounts:
    """Verify expected data counts in SQL collections."""

    @pytest.mark.requires_mongodb
    def test_sql_examples_count(self, mongodb_database):
        """Verify sql_examples has expected minimum count."""
        collection = mongodb_database["sql_examples"]
        count = collection.count_documents({})
        assert count >= 10, f"Expected at least 10 examples, got {count}"

    @pytest.mark.requires_mongodb
    def test_sql_rules_count(self, mongodb_database):
        """Verify sql_rules has expected minimum count."""
        collection = mongodb_database["sql_rules"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 rules, got {count}"

    @pytest.mark.requires_mongodb
    def test_stored_procedures_count(self, mongodb_database):
        """Verify sql_stored_procedures has expected minimum count."""
        collection = mongodb_database["sql_stored_procedures"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 procedures, got {count}"

    @pytest.mark.requires_mongodb
    def test_schema_context_count(self, mongodb_database):
        """Verify sql_schema_context has expected minimum count."""
        collection = mongodb_database["sql_schema_context"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 schema docs, got {count}"
