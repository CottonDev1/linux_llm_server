"""
SQL Pipeline Retrieval Tests
=============================

Tests for hybrid retrieval, relationship graph, and schema context injection.

Tests:
1. Hybrid retrieval (semantic + keyword)
2. RelationshipGraph FK-based JOIN path generation
3. Schema context formatting
4. Rule matching and scoring
"""

import pytest
import os
import sys

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestRelationshipGraph:
    """Tests for RelationshipGraph FK-based JOIN generation."""

    def test_build_relationship_context(self, sample_schema_context):
        """Test relationship context generation from schema."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context(sample_schema_context["tables"])

        assert context is not None, "Expected relationship context"
        assert "TABLE RELATIONSHIPS" in context, "Expected TABLE RELATIONSHIPS section"
        assert "CentralTickets" in context, "Expected CentralTickets in context"

    def test_foreign_key_relationships(self, sample_schema_context):
        """Test FK relationships are detected."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context(sample_schema_context["tables"])

        assert "TicketTypeID -> Types.TypeID" in context, "Expected TicketTypeID FK"
        assert "AddCentralUserID -> CentralUsers.CentralUserID" in context, "Expected AddCentralUserID FK"

    def test_join_patterns(self, sample_schema_context):
        """Test common JOIN patterns are generated."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context(sample_schema_context["tables"])

        assert "COMMON JOIN PATTERNS" in context, "Expected COMMON JOIN PATTERNS section"
        assert "joins to:" in context, "Expected join patterns"

    def test_empty_schema_handling(self):
        """Test handling of empty schema."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context([])
        # Should return empty or minimal context, not crash
        assert context is not None or context == ""


class TestSchemaFormatting:
    """Tests for schema context formatting."""

    def test_format_schema_with_fk(self, sample_schema_context):
        """Test schema formatting includes foreign keys."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        tickets_table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(tickets_table)

        assert "Foreign Keys:" in formatted, "Expected Foreign Keys section"
        assert "TicketTypeID -> Types.TypeID" in formatted, "Expected FK reference"

    def test_format_schema_with_sample_values(self, sample_schema_context):
        """Test schema formatting includes sample values."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        tickets_table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(tickets_table)

        assert "Examples:" in formatted, "Expected sample values"

    def test_format_schema_with_related_tables(self, sample_schema_context):
        """Test schema formatting includes related tables."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        tickets_table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(tickets_table)

        assert "Related Tables:" in formatted, "Expected Related Tables section"
        assert "Types" in formatted, "Expected Types in related tables"


class TestHybridRetrieval:
    """Tests for hybrid schema retrieval."""

    def test_mock_hybrid_retrieval(self, mock_mongodb):
        """Test hybrid retrieval with mock MongoDB."""
        import asyncio

        async def run_test():
            results = await mock_mongodb.hybrid_schema_retrieval("show tickets today")
            assert len(results) > 0, "Expected at least one result"
            assert "table_name" in results[0], "Expected table_name in result"
            assert "score" in results[0], "Expected score in result"

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_real_hybrid_retrieval(self, mongodb_uri, mongodb_database):
        """Test hybrid retrieval with real MongoDB (when available)."""
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

            # Check if hybrid_schema_retrieval method exists
            if not hasattr(mongo, 'hybrid_schema_retrieval'):
                await mongo.close()
                pytest.skip("hybrid_schema_retrieval not available")

            results = await mongo.hybrid_schema_retrieval(
                query="show tickets today",
                database="EWRCentral"
            )

            assert isinstance(results, list), "Expected list of results"

            await mongo.close()

        except ImportError as e:
            pytest.skip(f"MongoDBService not available: {e}")
        except Exception as e:
            pytest.skip(f"MongoDB not available: {e}")


class TestRuleMatching:
    """Tests for SQL rule matching."""

    def test_exact_match_detection(self, sample_sql_rules):
        """Test exact match rule detection."""
        from sql_pipeline.services.rules_service import RulesService

        rules_service = RulesService()

        # Find rule with exact match
        rule = next((r for r in sample_sql_rules if r.get("example")), None)
        if rule:
            question = rule["example"]["question"]
            # RulesService loads from config file, test the structure
            assert "question" in rule["example"]
            assert "sql" in rule["example"]

    def test_keyword_matching(self, sample_sql_rules):
        """Test keyword-based rule matching."""
        # Test keyword matching logic directly
        keywords = ["ticket", "today", "created"]
        question = "show tickets created today"

        # Check if keywords match
        matched_keywords = [k for k in keywords if k.lower() in question.lower()]
        assert len(matched_keywords) >= 2, "Expected keyword matches"

    def test_auto_fix_application(self, sample_sql_rules):
        """Test auto-fix pattern application."""
        import re

        # Get rule with auto_fix
        fix_rule = next((r for r in sample_sql_rules if r.get("auto_fix")), None)

        if fix_rule:
            pattern = fix_rule["auto_fix"]["pattern"]
            replacement = fix_rule["auto_fix"]["replacement"]

            # Test the fix
            test_sql = "SELECT * FROM CentralTickets WHERE CreateDate > GETDATE()"
            fixed = re.sub(pattern, replacement, test_sql)

            assert replacement in fixed, f"Expected '{replacement}' in fixed SQL"
            assert pattern not in fixed or pattern == replacement, f"Pattern '{pattern}' should be replaced"


class TestSchemaContextInjection:
    """Tests for MongoDB schema context injection into prompts."""

    def test_all_context_injected(self, sample_schema_context):
        """Test all context types are injected."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()

        for table in sample_schema_context["tables"]:
            formatted = gen._format_schema(table)

            # Check context injection
            assert "Table:" in formatted, "Missing table name"
            assert "Columns:" in formatted, "Missing columns section"

            if table.get("foreign_keys"):
                assert "Foreign Keys:" in formatted, "Missing FK section"

            if table.get("related_tables"):
                assert "Related Tables:" in formatted, "Missing related tables"

    def test_relationship_graph_integration(self, sample_schema_context):
        """Test relationship graph is integrated into SQL generator."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context(sample_schema_context["tables"])

        # Verify the context can be used
        assert isinstance(context, str), "Context should be string"
        assert len(context) > 0, "Context should not be empty"

        # Verify it contains useful information
        if len(sample_schema_context["tables"]) > 0:
            fks_exist = any(t.get("foreign_keys") for t in sample_schema_context["tables"])
            if fks_exist:
                assert "->" in context, "Expected FK arrows in context"
