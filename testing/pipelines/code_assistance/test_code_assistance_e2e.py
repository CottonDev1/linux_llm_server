"""
Code Assistance End-to-End Tests - Using Real Data
===================================================

Tests complete code assistance pipeline using REAL data.
Uses local llama.cpp (port 8082) and MongoDB only.
"""

import pytest
import time
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
)
from utils import assert_llm_response_valid


class TestCodeAssistanceDataFlow:
    """Test complete code assistance data flow."""

    @pytest.mark.requires_mongodb
    def test_database_has_required_collections(self, mongodb_database):
        """Verify all required code collections exist with data."""
        required_collections = [
            "code_methods",
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


class TestCodeAssistanceE2E:
    """End-to-end tests for code assistance pipeline with LLM."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_code_question(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete code assistance pipeline.

        Flow:
        1. Retrieve real code methods
        2. Build context from code
        3. Generate answer with LLM
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

        # Step 3: Generate answer
        user_question = "What do these code methods do?"

        prompt = f"""You are a helpful code assistant. Answer questions about the code.

CODE CONTEXT:
{code_text}

QUESTION: {user_question}

Provide a clear, helpful answer.

ANSWER:"""

        start_time = time.time()
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=512,
            temperature=0.2,
        )
        generation_time = time.time() - start_time

        # Step 4: Validate response
        assert_llm_response_valid(
            response,
            min_length=30,
        )

        # Response should be coherent
        assert len(response.text) > 20

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
        Test pipeline with project-scoped context.

        Verifies that context correctly filters by project.
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
                prompt = f"""Explain these methods from the {target_project} project.

CODE:
{context}

Provide a brief explanation.

EXPLANATION:"""

                response = llm_client.generate(
                    prompt=prompt,
                    model_type="code",
                    max_tokens=256,
                    temperature=0.2,
                )

                assert_llm_response_valid(response, min_length=20)


class TestCodeAssistancePerformance:
    """Test performance characteristics of code assistance pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_end_to_end_latency(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """Test total pipeline latency."""
        # Measure total pipeline time
        start_time = time.time()

        # 1. Code retrieval
        method = get_real_code_method(mongodb_database)
        retrieval_time = time.time() - start_time

        assert method is not None

        # 2. Generate response
        gen_start = time.time()
        content = method.get("content", method.get("code", "Code here"))[:500]
        method_name = method.get("method_name", "Method")

        prompt = f"""Explain this method:

{method_name}:
{content}

EXPLANATION:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=256,
            temperature=0.2,
        )
        gen_time = time.time() - gen_start

        total_time = time.time() - start_time

        # Validate performance
        assert retrieval_time < 1.0, f"Retrieval too slow: {retrieval_time:.2f}s"
        assert total_time < 60.0, f"Total pipeline too slow: {total_time:.2f}s"

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_retrieval_scalability(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieval with larger result sets."""
        collection = mongodb_database["code_methods"]

        # Perform search with limit
        start_time = time.time()
        results = list(collection.find().limit(100))
        search_time = time.time() - start_time

        assert len(results) <= 100
        assert search_time < 2.0, f"Search should be fast: {search_time:.2f}s"


class TestCodeAssistanceQuality:
    """Test quality of code assistance outputs."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_answer_coherence(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """Test that generated answers are coherent."""
        # Get a real code method
        method = get_real_code_method(mongodb_database)
        assert method is not None

        content = method.get("content", method.get("code", ""))[:500]
        method_name = method.get("method_name", "Method")
        class_name = method.get("class_name", "Class")

        prompt = f"""Summarize what this method does in 2-3 sentences.

METHOD: {method_name} in {class_name}

CODE:
{content}

SUMMARY:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=256,
            temperature=0.2,
        )

        assert_llm_response_valid(
            response,
            min_length=20,
        )

        # Response should be a coherent summary
        text = response.text
        assert len(text) > 20
        # Should contain complete sentences
        assert "." in text or "?" in text or "!" in text
