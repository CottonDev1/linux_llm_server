"""
Code Flow End-to-End Tests - Using Real Data
=============================================

Tests complete code flow analysis pipeline using REAL data.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
)
from utils import assert_llm_response_valid


class TestCodeFlowDataFlow:
    """Test complete code flow data flow."""

    @pytest.mark.requires_mongodb
    def test_database_has_required_collections(self, mongodb_database):
        """Verify all required code flow collections exist with data."""
        required_collections = [
            "code_methods",
            "code_callgraph",
            "code_classes",
        ]

        for col_name in required_collections:
            collection = mongodb_database[col_name]
            count = collection.count_documents({})
            assert count > 0, f"{col_name} should have documents"

    @pytest.mark.requires_mongodb
    def test_code_method_to_retrieval_flow(self, mongodb_database):
        """Test that stored code methods can be retrieved."""
        method = get_real_code_method(mongodb_database)
        assert method is not None

        collection = mongodb_database["code_methods"]
        retrieved = collection.find_one({"_id": method["_id"]})

        assert retrieved is not None
        assert retrieved["_id"] == method["_id"]

    @pytest.mark.requires_mongodb
    def test_callgraph_lookup_flow(self, mongodb_database):
        """Test that callgraph entries can be looked up."""
        collection = mongodb_database["code_callgraph"]
        doc = collection.find_one()
        assert doc is not None

        retrieved = collection.find_one({"_id": doc["_id"]})
        assert retrieved is not None


class TestCodeFlowE2E:
    """End-to-end tests for code flow pipeline with LLM."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_code_analysis(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete code flow pipeline.

        Flow:
        1. Retrieve real code methods
        2. Build context from code
        3. Analyze with LLM
        4. Validate response
        """
        # Step 1: Retrieve real code methods
        methods = get_real_code_methods(mongodb_database, limit=3)
        assert len(methods) > 0, "Should have code methods in database"

        # Step 2: Build context from real methods
        code_contexts = []
        for method in methods:
            method_name = method.get("method_name", "Method")
            class_name = method.get("class_name", "Class")
            content = method.get("content", method.get("code", ""))[:500]
            if content:
                code_contexts.append(f"{method_name} in {class_name}:\n{content}")

        code_text = "\n\n".join(code_contexts)

        # Step 3: Analyze with LLM
        prompt = f"""Analyze these code methods and identify:
1. What each method does
2. Any database operations
3. Method relationships

CODE:
{code_text}

ANALYSIS:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=512,
            temperature=0.0,
        )

        # Step 4: Validate response
        assert_llm_response_valid(
            response,
            min_length=30,
        )

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_pipeline_with_project_filtering(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test pipeline with project-scoped search.

        Verifies that search correctly filters by project.
        """
        collection = mongodb_database["code_methods"]

        # Get available projects
        projects = collection.distinct("project")

        if projects:
            target_project = projects[0]

            # Retrieve methods from specific project
            results = list(
                collection.find({"project": target_project}).limit(3)
            )

            # Should only get methods from target project
            assert len(results) > 0
            for result in results:
                assert result["project"] == target_project

            # Build context from project methods
            context = "\n".join([
                r.get("content", r.get("code", ""))[:300]
                for r in results if r.get("content") or r.get("code")
            ])

            if context:
                prompt = f"""Analyze these methods from the {target_project} project.

CODE:
{context}

Provide a brief analysis.

ANALYSIS:"""

                response = llm_client.generate(
                    prompt=prompt,
                    model_type="code",
                    max_tokens=256,
                    temperature=0.0,
                )

                assert_llm_response_valid(response, min_length=20)


class TestCallGraphTraversal:
    """Test call graph traversal on real data."""

    @pytest.mark.requires_mongodb
    def test_build_call_graph_from_callgraph(self, mongodb_database):
        """Test building a call graph from callgraph collection."""
        collection = mongodb_database["code_callgraph"]

        # Get a starting method
        doc = collection.find_one({"method_name": {"$exists": True}})

        if doc and "method_name" in doc:
            start_method = doc["method_name"]

            # Find methods called by this method
            called_methods = list(
                collection.find({"caller_method": start_method}).limit(10)
            )

            # The result should be a list (may be empty)
            assert isinstance(called_methods, list)

    @pytest.mark.requires_mongodb
    def test_find_callers_of_method(self, mongodb_database):
        """Test finding all callers of a method."""
        collection = mongodb_database["code_callgraph"]

        # Get a method that might be called
        doc = collection.find_one({"callee_method": {"$exists": True}})

        if doc and "callee_method" in doc:
            target_method = doc["callee_method"]

            # Find all methods that call this one
            callers = list(
                collection.find({"callee_method": target_method}).limit(10)
            )

            assert isinstance(callers, list)


class TestCodeFlowPerformance:
    """Test performance characteristics of code flow pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_retrieval_scalability(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieval with larger result sets."""
        collection = mongodb_database["code_methods"]

        # Perform search with limit
        import time
        start_time = time.time()
        results = list(collection.find().limit(100))
        search_time = time.time() - start_time

        assert len(results) <= 100
        assert search_time < 2.0, f"Search should be fast: {search_time:.2f}s"

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_callgraph_traversal_performance(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test callgraph traversal performance."""
        collection = mongodb_database["code_callgraph"]

        import time
        start_time = time.time()
        results = list(collection.find().limit(100))
        search_time = time.time() - start_time

        assert len(results) <= 100
        assert search_time < 2.0, f"Callgraph query should be fast: {search_time:.2f}s"
