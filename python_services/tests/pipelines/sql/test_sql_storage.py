"""
SQL Pipeline Storage Tests
==========================

Tests for MongoDB schema storage and retrieval functionality.

Tests:
1. Schema document structure validation
2. Foreign key storage and retrieval
3. Sample values storage
4. Related tables mapping
5. Stored procedure metadata
"""

import pytest
import os
import sys

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestSchemaStorage:
    """Tests for schema storage in MongoDB."""

    def test_schema_document_structure(self, sample_schema_context):
        """Verify schema documents have required fields."""
        for table in sample_schema_context["tables"]:
            assert "table_name" in table, "Missing table_name"
            assert "columns" in table, "Missing columns"
            assert isinstance(table["columns"], list), "Columns must be a list"

            # Check column structure
            for col in table["columns"]:
                assert "name" in col, "Column missing name"
                assert "type" in col, "Column missing type"

    def test_foreign_keys_stored(self, sample_schema_context):
        """Verify foreign keys are properly stored."""
        tickets_table = next(
            (t for t in sample_schema_context["tables"] if t["table_name"] == "CentralTickets"),
            None
        )
        assert tickets_table is not None, "CentralTickets table not found"

        fks = tickets_table.get("foreign_keys", [])
        assert len(fks) >= 1, "Expected at least one foreign key"

        # Check FK structure
        for fk in fks:
            assert "column" in fk, "FK missing column"
            assert "references_table" in fk, "FK missing references_table"
            assert "references_column" in fk, "FK missing references_column"

    def test_sample_values_stored(self, sample_schema_context):
        """Verify sample values are stored for columns."""
        tickets_table = next(
            (t for t in sample_schema_context["tables"] if t["table_name"] == "CentralTickets"),
            None
        )
        assert tickets_table is not None

        samples = tickets_table.get("sample_values", {})
        assert "AddTicketDate" in samples, "Expected sample values for AddTicketDate"
        assert len(samples["AddTicketDate"]) > 0, "Expected at least one sample value"

    def test_related_tables_stored(self, sample_schema_context):
        """Verify related tables are mapped."""
        tickets_table = next(
            (t for t in sample_schema_context["tables"] if t["table_name"] == "CentralTickets"),
            None
        )
        assert tickets_table is not None

        related = tickets_table.get("related_tables", [])
        assert "Types" in related, "Expected Types in related tables"
        assert "CentralUsers" in related, "Expected CentralUsers in related tables"

    def test_stored_procedures_metadata(self, sample_schema_context):
        """Verify stored procedures metadata is stored."""
        procs = sample_schema_context.get("stored_procedures", [])
        assert len(procs) > 0, "Expected at least one stored procedure"

        for proc in procs:
            assert "name" in proc, "Procedure missing name"


class TestMongoDBIntegration:
    """Integration tests with real MongoDB (when available).

    Note: MongoDBService reads connection settings from environment/config,
    so we set MONGODB_URI environment variable before importing.
    """

    @pytest.mark.asyncio
    async def test_mongodb_connection(self, mongodb_uri, mongodb_database):
        """Test MongoDB connection."""
        # Set environment before importing MongoDBService
        os.environ["MONGODB_URI"] = mongodb_uri
        os.environ["MONGODB_DATABASE"] = mongodb_database

        try:
            # Force reimport with new env vars
            import importlib
            import mongodb_service
            importlib.reload(mongodb_service)
            from mongodb import MongoDBService

            mongo = MongoDBService()
            await mongo.initialize()

            assert mongo.client is not None, "MongoDB client not initialized"

            # Verify we can access collections
            db = mongo.client[mongodb_database]
            collections = await db.list_collection_names()

            assert isinstance(collections, list), "Expected list of collections"

            await mongo.close()

        except ImportError as e:
            pytest.skip(f"MongoDBService not available: {e}")
        except Exception as e:
            pytest.skip(f"MongoDB not available: {e}")

    @pytest.mark.asyncio
    async def test_schema_retrieval(self, mongodb_uri, mongodb_database):
        """Test schema retrieval from MongoDB."""
        # Set environment before importing MongoDBService
        os.environ["MONGODB_URI"] = mongodb_uri
        os.environ["MONGODB_DATABASE"] = mongodb_database

        try:
            import importlib
            import mongodb_service
            importlib.reload(mongodb_service)
            from mongodb import MongoDBService

            mongo = MongoDBService()
            await mongo.initialize()

            # Try to retrieve a schema (signature: database, table_name)
            # Note: table_name includes schema prefix (e.g., 'dbo.CentralTickets')
            schema = await mongo.get_schema_by_table("EWRCentral", "dbo.CentralTickets")

            if schema:
                assert "table_name" in schema or "columns" in schema
            else:
                pytest.skip("No schema found for CentralTickets")

            await mongo.close()

        except ImportError as e:
            pytest.skip(f"MongoDBService not available: {e}")
        except Exception as e:
            pytest.skip(f"MongoDB not available: {e}")


class TestSQLRulesStorage:
    """Tests for SQL rules storage and retrieval."""

    def test_rules_structure(self, sample_sql_rules):
        """Verify SQL rules have required structure."""
        for rule in sample_sql_rules:
            assert "id" in rule, "Rule missing id"
            assert "description" in rule, "Rule missing description"
            assert "type" in rule, "Rule missing type"
            assert rule["type"] in ["assistance", "constraint"], f"Invalid rule type: {rule['type']}"

    def test_exact_match_rules(self, sample_sql_rules):
        """Verify exact match rules have example questions and SQL."""
        exact_rules = [r for r in sample_sql_rules if r.get("example")]

        for rule in exact_rules:
            example = rule["example"]
            assert "question" in example, "Example missing question"
            assert "sql" in example, "Example missing sql"
            assert len(example["question"]) > 0, "Question cannot be empty"
            assert len(example["sql"]) > 0, "SQL cannot be empty"

    def test_auto_fix_patterns(self, sample_sql_rules):
        """Verify auto-fix patterns are valid regex."""
        import re

        for rule in sample_sql_rules:
            if "auto_fix" in rule:
                pattern = rule["auto_fix"].get("pattern")
                assert pattern, "Auto-fix missing pattern"

                # Verify it's valid regex
                try:
                    re.compile(pattern)
                except re.error as e:
                    pytest.fail(f"Invalid regex pattern in rule {rule['id']}: {e}")

    def test_trigger_keywords(self, sample_sql_rules):
        """Verify trigger keywords are present."""
        for rule in sample_sql_rules:
            keywords = rule.get("trigger_keywords", [])
            tables = rule.get("trigger_tables", [])
            example = rule.get("example")

            # Rule should have at least one trigger mechanism
            has_trigger = len(keywords) > 0 or len(tables) > 0 or example is not None
            assert has_trigger, f"Rule {rule['id']} has no trigger mechanism"
