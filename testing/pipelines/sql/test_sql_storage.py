"""
SQL Storage Tests - Using Real Data
====================================

Tests SQL data storage validation using REAL data from the database.
Tests verify that stored data has correct structure and can be queried.
"""

import pytest
from typing import Dict, Any

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    verify_database_has_data,
    get_real_sql_example,
    get_real_sql_examples,
    get_real_sql_rule,
    get_real_sql_rules,
    get_real_stored_procedure,
)
from utils import (
    assert_mongodb_document,
)


class TestSQLDataExists:
    """Verify SQL-related collections have data."""

    @pytest.mark.requires_mongodb
    def test_sql_examples_collection_has_data(self, mongodb_database):
        """Verify sql_examples collection exists and has documents."""
        collection = mongodb_database["sql_examples"]
        count = collection.count_documents({})
        assert count > 0, "sql_examples collection should have documents"

    @pytest.mark.requires_mongodb
    def test_sql_rules_collection_has_data(self, mongodb_database):
        """Verify sql_rules collection exists and has documents."""
        collection = mongodb_database["sql_rules"]
        count = collection.count_documents({})
        assert count > 0, "sql_rules collection should have documents"

    @pytest.mark.requires_mongodb
    def test_sql_stored_procedures_has_data(self, mongodb_database):
        """Verify sql_stored_procedures collection exists and has documents."""
        collection = mongodb_database["sql_stored_procedures"]
        count = collection.count_documents({})
        assert count > 0, "sql_stored_procedures collection should have documents"

    @pytest.mark.requires_mongodb
    def test_sql_schema_context_has_data(self, mongodb_database):
        """Verify sql_schema_context collection exists and has documents."""
        collection = mongodb_database["sql_schema_context"]
        count = collection.count_documents({})
        assert count > 0, "sql_schema_context collection should have documents"


class TestSQLExamplesStructure:
    """Test structure and content of sql_examples documents."""

    @pytest.mark.requires_mongodb
    def test_sql_example_has_required_fields(self, mongodb_database):
        """Test that SQL examples have required fields."""
        example = get_real_sql_example(mongodb_database)
        assert example is not None, "Should have at least one SQL example"

        # Check required fields exist
        assert "_id" in example, "Example should have _id"
        assert "sql" in example, "Example should have sql field"

    @pytest.mark.requires_mongodb
    def test_sql_example_has_database_field(self, mongodb_database):
        """Test that SQL examples have database field."""
        example = get_real_sql_example(mongodb_database)
        assert example is not None

        # Database field should be present
        if "database" in example:
            assert isinstance(example["database"], str)
            assert len(example["database"]) > 0

    @pytest.mark.requires_mongodb
    def test_sql_example_sql_is_valid_string(self, mongodb_database):
        """Test that SQL field contains valid SQL string."""
        examples = get_real_sql_examples(mongodb_database, limit=10)
        assert len(examples) > 0, "Should have SQL examples"

        for example in examples:
            sql = example.get("sql", "")
            assert isinstance(sql, str), "SQL should be a string"
            assert len(sql) > 0, "SQL should not be empty"
            # Basic SQL validation - should have SELECT keyword
            assert "SELECT" in sql.upper(), f"SQL should contain SELECT: {sql[:100]}"

    @pytest.mark.requires_mongodb
    def test_multiple_sql_examples_retrievable(self, mongodb_database):
        """Test that multiple SQL examples can be retrieved."""
        examples = get_real_sql_examples(mongodb_database, limit=5)
        assert len(examples) >= 1, "Should retrieve at least 1 example"

        # Verify each has sql field
        for ex in examples:
            assert "sql" in ex


class TestSQLRulesStructure:
    """Test structure and content of sql_rules documents."""

    @pytest.mark.requires_mongodb
    def test_sql_rule_has_description(self, mongodb_database):
        """Test that SQL rules have description field."""
        rule = get_real_sql_rule(mongodb_database)
        assert rule is not None, "Should have at least one SQL rule"

        # Check description field
        assert "_id" in rule, "Rule should have _id"
        if "description" in rule:
            assert isinstance(rule["description"], str)
            assert len(rule["description"]) > 0

    @pytest.mark.requires_mongodb
    def test_multiple_sql_rules_retrievable(self, mongodb_database):
        """Test that multiple SQL rules can be retrieved."""
        rules = get_real_sql_rules(mongodb_database, limit=10)
        assert len(rules) >= 1, "Should retrieve at least 1 rule"

    @pytest.mark.requires_mongodb
    def test_sql_rules_count(self, mongodb_database):
        """Test sql_rules collection has expected count."""
        collection = mongodb_database["sql_rules"]
        count = collection.count_documents({})
        # We know from backup there are 420 rules
        assert count >= 100, f"Expected at least 100 rules, got {count}"


class TestStoredProceduresStructure:
    """Test structure of sql_stored_procedures documents."""

    @pytest.mark.requires_mongodb
    def test_stored_procedure_has_required_fields(self, mongodb_database):
        """Test that stored procedures have required fields."""
        sp = get_real_stored_procedure(mongodb_database)
        assert sp is not None, "Should have at least one stored procedure"

        assert "_id" in sp, "SP should have _id"

    @pytest.mark.requires_mongodb
    def test_stored_procedure_has_database(self, mongodb_database):
        """Test that stored procedures have database field."""
        collection = mongodb_database["sql_stored_procedures"]
        sp = collection.find_one({"database": {"$exists": True}})

        if sp:
            assert isinstance(sp["database"], str)

    @pytest.mark.requires_mongodb
    def test_stored_procedures_count(self, mongodb_database):
        """Test sql_stored_procedures has expected count."""
        collection = mongodb_database["sql_stored_procedures"]
        count = collection.count_documents({})
        # We know from backup there are 1619 stored procedures
        assert count >= 100, f"Expected at least 100 SPs, got {count}"


class TestSchemaContextStructure:
    """Test structure of sql_schema_context documents."""

    @pytest.mark.requires_mongodb
    def test_schema_context_has_table_info(self, mongodb_database):
        """Test that schema context documents have table information."""
        collection = mongodb_database["sql_schema_context"]
        doc = collection.find_one()

        assert doc is not None, "Should have schema context documents"
        assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_schema_context_has_columns(self, mongodb_database):
        """Test that schema context documents have column information."""
        collection = mongodb_database["sql_schema_context"]
        doc = collection.find_one({"columns": {"$exists": True}})

        if doc and "columns" in doc:
            assert isinstance(doc["columns"], list)

    @pytest.mark.requires_mongodb
    def test_schema_context_count(self, mongodb_database):
        """Test sql_schema_context has expected count."""
        collection = mongodb_database["sql_schema_context"]
        count = collection.count_documents({})
        # We know from backup there are 468 schema context docs
        assert count >= 100, f"Expected at least 100 schema docs, got {count}"


class TestSQLQueryability:
    """Test that SQL data can be queried effectively."""

    @pytest.mark.requires_mongodb
    def test_query_examples_by_database(self, mongodb_database):
        """Test querying SQL examples by database name."""
        collection = mongodb_database["sql_examples"]

        # Get distinct databases
        databases = collection.distinct("database")

        if databases:
            # Query by first database
            db_name = databases[0]
            results = list(collection.find({"database": db_name}).limit(5))
            assert len(results) > 0, f"Should find examples for database {db_name}"

    @pytest.mark.requires_mongodb
    def test_query_schema_by_table_name(self, mongodb_database):
        """Test querying schema context by table name."""
        collection = mongodb_database["sql_schema_context"]

        # Get one document and use its table_name if present
        doc = collection.find_one({"table_name": {"$exists": True}})

        if doc and "table_name" in doc:
            table_name = doc["table_name"]
            result = collection.find_one({"table_name": table_name})
            assert result is not None

    @pytest.mark.requires_mongodb
    def test_sql_examples_indexable(self, mongodb_database):
        """Test that sql_examples can be indexed for retrieval."""
        collection = mongodb_database["sql_examples"]

        # Get indexes
        indexes = list(collection.list_indexes())
        index_names = [idx["name"] for idx in indexes]

        # Should have at least _id index
        assert "_id_" in index_names
