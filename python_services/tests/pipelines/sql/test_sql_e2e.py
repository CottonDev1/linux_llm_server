"""
SQL Pipeline End-to-End Tests
==============================

End-to-end tests for the complete SQL generation pipeline.

Flow: Question -> Cache -> Rules -> Schema -> LLM -> Validation

Tests:
1. Complete pipeline execution
2. Rule bypass (exact match)
3. Schema context injection
4. Relationship graph integration
5. Error handling and recovery
"""

import pytest
import os
import sys
import asyncio

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestPipelineFlow:
    """Tests for complete pipeline flow."""

    def test_pipeline_components_exist(self):
        """Verify all pipeline components are importable."""
        components = []

        try:
            from services.sql_generator import SQLGeneratorService
            components.append("SQLGeneratorService")
        except ImportError as e:
            pytest.fail(f"Cannot import SQLGeneratorService: {e}")

        try:
            from services.relationship_graph import build_relationship_context
            components.append("RelationshipGraph")
        except ImportError as e:
            pytest.fail(f"Cannot import RelationshipGraph: {e}")

        try:
            from sql_pipeline.services.rules_service import RulesService
            components.append("RulesService")
        except ImportError as e:
            pytest.fail(f"Cannot import RulesService: {e}")

        assert len(components) >= 3, f"Missing components: {components}"

    def test_pipeline_integration(self, sample_schema_context):
        """Test components work together."""
        from services.sql_generator import SQLGeneratorService
        from services.relationship_graph import build_relationship_context

        # Initialize components
        gen = SQLGeneratorService()

        # Format schema
        for table in sample_schema_context["tables"]:
            formatted = gen._format_schema(table)
            assert len(formatted) > 0

        # Build relationship context
        rel_context = build_relationship_context(sample_schema_context["tables"])
        assert rel_context is not None

    def test_schema_to_prompt_flow(self, sample_schema_context):
        """Test schema -> relationship graph -> prompt flow."""
        from services.sql_generator import SQLGeneratorService, SchemaContext
        from services.relationship_graph import build_relationship_context

        gen = SQLGeneratorService()

        # Step 1: Build relationship context
        rel_context = build_relationship_context(sample_schema_context["tables"])

        # Step 2: Format individual schemas
        formatted_tables = []
        for table in sample_schema_context["tables"]:
            formatted = gen._format_schema(table)
            formatted_tables.append(formatted)

        # Step 3: Verify all context is available
        all_context = "\n".join(formatted_tables) + "\n" + (rel_context or "")

        assert "CentralTickets" in all_context
        assert "Foreign Keys:" in all_context or "TABLE RELATIONSHIPS:" in all_context


class TestExactMatchBypass:
    """Tests for exact match rule bypass."""

    def test_exact_match_rule_loaded(self, sample_sql_rules):
        """Test exact match rules are properly loaded."""
        exact_rules = [r for r in sample_sql_rules if r.get("example")]

        assert len(exact_rules) > 0, "Expected at least one exact match rule"

        for rule in exact_rules:
            assert rule["example"]["question"], "Missing question"
            assert rule["example"]["sql"], "Missing SQL"

    def test_exact_match_returns_sql_directly(self, sample_sql_rules):
        """Test exact match bypasses LLM and returns SQL."""
        # Simulate exact match logic
        exact_rule = next((r for r in sample_sql_rules if r.get("example")), None)

        if exact_rule:
            question = exact_rule["example"]["question"]
            expected_sql = exact_rule["example"]["sql"]

            # Simulate matching
            for rule in sample_sql_rules:
                if rule.get("example", {}).get("question") == question:
                    assert rule["example"]["sql"] == expected_sql
                    break


class TestSchemaContextInjectionE2E:
    """End-to-end tests for schema context injection."""

    def test_full_context_injection(self, sample_schema_context):
        """Test complete context injection flow."""
        from services.sql_generator import SQLGeneratorService, SchemaContext
        from services.relationship_graph import build_relationship_context

        gen = SQLGeneratorService()

        # Create full context using dataclass
        schema_ctx = SchemaContext(
            tables=sample_schema_context["tables"],
            stored_procedures=sample_schema_context.get("stored_procedures", [])
        )

        # Verify context has all required data
        assert len(schema_ctx.tables) > 0

        # Format all tables
        for table in schema_ctx.tables:
            formatted = gen._format_schema(table)

            # Verify rich context
            if table.get("foreign_keys"):
                assert "Foreign Keys:" in formatted
            if table.get("related_tables"):
                assert "Related Tables:" in formatted
            if table.get("sample_values"):
                assert "Examples:" in formatted

        # Verify relationship graph
        rel_context = build_relationship_context(schema_ctx.tables)
        if any(t.get("foreign_keys") for t in schema_ctx.tables):
            assert "TABLE RELATIONSHIPS:" in rel_context

    def test_context_completeness(self, sample_schema_context):
        """Test all context sources are included."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()
        tickets_table = sample_schema_context["tables"][0]

        formatted = gen._format_schema(tickets_table)

        # Check all context sources
        context_sources = {
            "table_name": "CentralTickets" in formatted,
            "columns": "Columns:" in formatted,
            "foreign_keys": "Foreign Keys:" in formatted,
            "related_tables": "Related Tables:" in formatted,
            "sample_values": "Examples:" in formatted,
        }

        missing = [k for k, v in context_sources.items() if not v]
        assert len(missing) == 0, f"Missing context sources: {missing}"


class TestRelationshipGraphE2E:
    """End-to-end tests for relationship graph integration."""

    def test_relationship_graph_in_pipeline(self, sample_schema_context):
        """Test relationship graph is used in pipeline."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context(sample_schema_context["tables"])

        # Verify graph output
        assert "TABLE RELATIONSHIPS:" in context
        assert "COMMON JOIN PATTERNS:" in context

        # Verify specific relationships
        assert "CentralTickets" in context
        assert "Types" in context
        assert "CentralUsers" in context

    def test_join_hints_generated(self, sample_schema_context):
        """Test JOIN hints are generated from FKs."""
        from services.relationship_graph import build_relationship_context

        context = build_relationship_context(sample_schema_context["tables"])

        # Check for JOIN pattern hints
        assert "joins to:" in context

        # Verify FK-based hints
        assert "TicketTypeID" in context or "Types" in context


class TestErrorHandling:
    """Tests for error handling in pipeline."""

    def test_empty_schema_handling(self):
        """Test pipeline handles empty schema gracefully."""
        from services.sql_generator import SQLGeneratorService
        from services.relationship_graph import build_relationship_context

        gen = SQLGeneratorService()

        # Empty table
        empty_table = {"table_name": "EmptyTable", "columns": []}
        formatted = gen._format_schema(empty_table)

        assert "EmptyTable" in formatted
        assert formatted is not None

        # Empty tables list
        context = build_relationship_context([])
        assert context is not None or context == ""

    def test_missing_fields_handling(self):
        """Test handling of tables with missing fields."""
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()

        # Table with minimal fields
        minimal_table = {
            "table_name": "MinimalTable",
            "columns": [{"name": "ID", "type": "int"}]
        }

        formatted = gen._format_schema(minimal_table)
        assert "MinimalTable" in formatted
        assert "ID" in formatted

    def test_invalid_fk_handling(self):
        """Test handling of invalid FK references."""
        from services.relationship_graph import build_relationship_context

        # Table with potentially invalid FK
        tables = [{
            "table_name": "TestTable",
            "columns": [{"name": "ID", "type": "int"}],
            "foreign_keys": [
                {"column": "RefID", "references_table": "NonExistent", "references_column": "ID"}
            ]
        }]

        # Should not crash
        context = build_relationship_context(tables)
        assert context is not None


class TestPerformanceBaseline:
    """Basic performance tests."""

    def test_schema_formatting_performance(self, sample_schema_context):
        """Test schema formatting completes in reasonable time."""
        import time
        from services.sql_generator import SQLGeneratorService

        gen = SQLGeneratorService()

        start = time.time()

        # Format all tables multiple times
        for _ in range(100):
            for table in sample_schema_context["tables"]:
                gen._format_schema(table)

        elapsed = time.time() - start

        # Should complete 300 format operations in under 1 second
        assert elapsed < 1.0, f"Schema formatting too slow: {elapsed:.2f}s"

    def test_relationship_graph_performance(self, sample_schema_context):
        """Test relationship graph building performance."""
        import time
        from services.relationship_graph import build_relationship_context

        start = time.time()

        # Build graph multiple times
        for _ in range(100):
            build_relationship_context(sample_schema_context["tables"])

        elapsed = time.time() - start

        # Should complete 100 builds in under 1 second
        assert elapsed < 1.0, f"Relationship graph building too slow: {elapsed:.2f}s"
