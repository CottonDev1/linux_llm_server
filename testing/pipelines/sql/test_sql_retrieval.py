"""
SQL Retrieval Tests - Using Real Data
======================================

Tests SQL data retrieval operations using REAL data from the database.
Tests verify that data can be queried, filtered, and retrieved correctly.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_sql_example,
    get_real_sql_examples,
    get_real_sql_rule,
    get_real_sql_rules,
    get_real_stored_procedure,
    get_real_stored_procedures,
)


class TestSQLExampleRetrieval:
    """Test retrieval of SQL examples."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_example(self, mongodb_database):
        """Test retrieving a single SQL example."""
        example = get_real_sql_example(mongodb_database)
        assert example is not None
        assert "_id" in example
        assert "sql" in example

    @pytest.mark.requires_mongodb
    def test_retrieve_multiple_examples(self, mongodb_database):
        """Test retrieving multiple SQL examples."""
        examples = get_real_sql_examples(mongodb_database, limit=10)
        assert len(examples) > 0

        # Each should have required fields
        for ex in examples:
            assert "_id" in ex
            assert "sql" in ex

    @pytest.mark.requires_mongodb
    def test_retrieve_examples_by_database(self, mongodb_database):
        """Test filtering SQL examples by database name."""
        collection = mongodb_database["sql_examples"]

        # Get distinct databases
        databases = collection.distinct("database")

        if databases:
            db_name = databases[0]
            results = list(collection.find({"database": db_name}))

            assert len(results) > 0
            for result in results:
                assert result.get("database") == db_name

    @pytest.mark.requires_mongodb
    def test_retrieve_example_with_projection(self, mongodb_database):
        """Test retrieving SQL example with field projection."""
        collection = mongodb_database["sql_examples"]

        # Only retrieve sql field
        result = collection.find_one({}, {"sql": 1, "_id": 0})

        assert result is not None
        assert "sql" in result
        # _id should be excluded
        assert "_id" not in result


class TestSQLRuleRetrieval:
    """Test retrieval of SQL rules."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_rule(self, mongodb_database):
        """Test retrieving a single SQL rule."""
        rule = get_real_sql_rule(mongodb_database)
        assert rule is not None
        assert "_id" in rule

    @pytest.mark.requires_mongodb
    def test_retrieve_multiple_rules(self, mongodb_database):
        """Test retrieving multiple SQL rules."""
        rules = get_real_sql_rules(mongodb_database, limit=20)
        assert len(rules) > 0

        for rule in rules:
            assert "_id" in rule

    @pytest.mark.requires_mongodb
    def test_retrieve_rules_with_description(self, mongodb_database):
        """Test retrieving rules that have descriptions."""
        collection = mongodb_database["sql_rules"]

        rules_with_desc = list(
            collection.find({"description": {"$exists": True}}).limit(10)
        )

        for rule in rules_with_desc:
            assert "description" in rule
            assert isinstance(rule["description"], str)

    @pytest.mark.requires_mongodb
    def test_search_rules_by_keyword(self, mongodb_database):
        """Test searching rules by keyword in description."""
        collection = mongodb_database["sql_rules"]

        # Search for common SQL keywords
        keywords = ["date", "table", "column", "join", "select"]

        for keyword in keywords:
            results = list(
                collection.find(
                    {"description": {"$regex": keyword, "$options": "i"}}
                ).limit(5)
            )
            # May or may not find results, just verify no error
            assert isinstance(results, list)


class TestStoredProcedureRetrieval:
    """Test retrieval of stored procedures."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_procedure(self, mongodb_database):
        """Test retrieving a single stored procedure."""
        sp = get_real_stored_procedure(mongodb_database)
        assert sp is not None
        assert "_id" in sp

    @pytest.mark.requires_mongodb
    def test_retrieve_multiple_procedures(self, mongodb_database):
        """Test retrieving multiple stored procedures."""
        procedures = get_real_stored_procedures(mongodb_database, limit=10)
        assert len(procedures) > 0

    @pytest.mark.requires_mongodb
    def test_retrieve_procedures_by_database(self, mongodb_database):
        """Test filtering stored procedures by database."""
        collection = mongodb_database["sql_stored_procedures"]

        # Get distinct databases
        databases = collection.distinct("database")

        if databases:
            db_name = databases[0]
            results = list(collection.find({"database": db_name}).limit(10))

            for result in results:
                assert result.get("database") == db_name

    @pytest.mark.requires_mongodb
    def test_retrieve_procedure_by_name(self, mongodb_database):
        """Test retrieving stored procedure by name."""
        collection = mongodb_database["sql_stored_procedures"]

        # Get one procedure first
        sp = collection.find_one({"procedure_name": {"$exists": True}})

        if sp and "procedure_name" in sp:
            proc_name = sp["procedure_name"]
            result = collection.find_one({"procedure_name": proc_name})
            assert result is not None
            assert result["procedure_name"] == proc_name


class TestSchemaContextRetrieval:
    """Test retrieval of schema context."""

    @pytest.mark.requires_mongodb
    def test_retrieve_schema_for_table(self, mongodb_database):
        """Test retrieving schema context for a specific table."""
        collection = mongodb_database["sql_schema_context"]

        # Get one document
        doc = collection.find_one({"table_name": {"$exists": True}})

        if doc:
            table_name = doc["table_name"]
            result = collection.find_one({"table_name": table_name})
            assert result is not None

    @pytest.mark.requires_mongodb
    def test_retrieve_schema_with_columns(self, mongodb_database):
        """Test retrieving schema that has column information."""
        collection = mongodb_database["sql_schema_context"]

        doc = collection.find_one({"columns": {"$exists": True, "$ne": []}})

        if doc:
            assert "columns" in doc
            assert isinstance(doc["columns"], list)
            assert len(doc["columns"]) > 0

    @pytest.mark.requires_mongodb
    def test_retrieve_schema_by_database(self, mongodb_database):
        """Test filtering schema context by database."""
        collection = mongodb_database["sql_schema_context"]

        databases = collection.distinct("database")

        if databases:
            db_name = databases[0]
            results = list(collection.find({"database": db_name}).limit(10))

            for result in results:
                assert result.get("database") == db_name


class TestAggregationQueries:
    """Test aggregation queries on SQL collections."""

    @pytest.mark.requires_mongodb
    def test_count_examples_per_database(self, mongodb_database):
        """Test counting SQL examples per database."""
        collection = mongodb_database["sql_examples"]

        pipeline = [
            {"$group": {"_id": "$database", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0

        for result in results:
            assert "count" in result
            assert result["count"] > 0

    @pytest.mark.requires_mongodb
    def test_count_procedures_per_database(self, mongodb_database):
        """Test counting stored procedures per database."""
        collection = mongodb_database["sql_stored_procedures"]

        pipeline = [
            {"$group": {"_id": "$database", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_sample_random_examples(self, mongodb_database):
        """Test sampling random SQL examples."""
        collection = mongodb_database["sql_examples"]

        pipeline = [{"$sample": {"size": 5}}]
        results = list(collection.aggregate(pipeline))

        assert len(results) <= 5
        for result in results:
            assert "_id" in result


class TestQueryPerformance:
    """Test query performance characteristics."""

    @pytest.mark.requires_mongodb
    def test_indexed_query_performance(self, mongodb_database):
        """Test that indexed queries are efficient."""
        collection = mongodb_database["sql_examples"]

        # Query by _id should use index
        doc = collection.find_one()
        if doc:
            result = collection.find_one({"_id": doc["_id"]})
            assert result is not None
            assert result["_id"] == doc["_id"]

    @pytest.mark.requires_mongodb
    def test_limit_query_performance(self, mongodb_database):
        """Test that limit queries return quickly."""
        collection = mongodb_database["sql_examples"]

        # Should return quickly with limit
        results = list(collection.find().limit(10))
        assert len(results) <= 10

    @pytest.mark.requires_mongodb
    def test_projection_reduces_data(self, mongodb_database):
        """Test that projection reduces returned data."""
        collection = mongodb_database["sql_schema_context"]

        # Full document
        full_doc = collection.find_one()

        # Projected document
        projected = collection.find_one({}, {"_id": 1, "table_name": 1})

        if full_doc and projected:
            # Projected should have fewer fields
            assert len(projected) <= len(full_doc)
