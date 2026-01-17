"""
Query/RAG Pipeline End-to-End Tests - Using Real Data
======================================================

Tests complete RAG pipeline flow using REAL data.
Uses local llama.cpp (port 8081) and MongoDB only.
"""

import pytest
import time
import math
from typing import List, Dict, Any

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_document,
    get_real_documents,
    get_real_code_method,
    get_real_code_methods,
)
from utils import assert_llm_response_valid


class TestRAGPipelineDataFlow:
    """Test complete RAG data flow."""

    @pytest.mark.requires_mongodb
    def test_database_has_required_collections(self, mongodb_database):
        """Verify all required RAG collections exist with data."""
        required_collections = [
            "documents",
            "code_methods",
        ]

        for col_name in required_collections:
            collection = mongodb_database[col_name]
            count = collection.count_documents({})
            assert count > 0, f"{col_name} should have documents"

    @pytest.mark.requires_mongodb
    def test_document_to_retrieval_flow(self, mongodb_database):
        """Test that stored documents can be retrieved."""
        doc = get_real_document(mongodb_database)
        assert doc is not None

        collection = mongodb_database["documents"]
        retrieved = collection.find_one({"_id": doc["_id"]})

        assert retrieved is not None
        assert retrieved["_id"] == doc["_id"]

    @pytest.mark.requires_mongodb
    def test_code_method_to_retrieval_flow(self, mongodb_database):
        """Test that stored code methods can be retrieved."""
        method = get_real_code_method(mongodb_database)
        assert method is not None

        collection = mongodb_database["code_methods"]
        retrieved = collection.find_one({"_id": method["_id"]})

        assert retrieved is not None
        assert retrieved["_id"] == method["_id"]


class TestRAGPipelineE2E:
    """End-to-end tests for complete RAG pipeline with LLM."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_document_query(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete RAG pipeline for document query.

        Flow:
        1. Retrieve real documents
        2. Use document content as context
        3. Generate answer with LLM
        4. Validate response quality
        """
        # Step 1: Retrieve real documents
        docs = get_real_documents(mongodb_database, limit=3)
        assert len(docs) > 0, "Should have documents in database"

        # Step 2: Build context from real documents
        contexts = []
        for doc in docs:
            title = doc.get("title", "Document")
            content = doc.get("content", "")[:500]  # Limit content length
            if content:
                contexts.append(f"{title}: {content}")

        context_text = "\n\n".join(contexts)

        # Step 3: Generate answer
        prompt = f"""Answer the question using the provided documentation.

DOCUMENTATION:
{context_text}

QUESTION: What information is provided in these documents?

Provide a clear, comprehensive summary.

ANSWER:"""

        start_time = time.time()
        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )
        generation_time = time.time() - start_time

        # Step 4: Validate response
        assert_llm_response_valid(
            response,
            min_length=30,
        )

        # Response should be coherent (not empty or error)
        assert len(response.text) > 20

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_code_query(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete RAG pipeline for code-related query.

        Flow:
        1. Retrieve real code methods
        2. Use code as context
        3. Generate explanation
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

        # Step 3: Generate explanation
        prompt = f"""Analyze the following code methods.

CODE:
{code_text}

QUESTION: What do these methods do?

Provide a brief explanation of each method's purpose.

EXPLANATION:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

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

            # Generate answer using project-specific context
            context = "\n".join([
                r.get("content", r.get("code", ""))[:300]
                for r in results if r.get("content") or r.get("code")
            ])

            if context:
                prompt = f"""Describe these methods from the {target_project} project.

CODE:
{context}

Provide a brief summary.

SUMMARY:"""

                response = llm_client.generate(
                    prompt=prompt,
                    model_type="general",
                    max_tokens=256,
                    temperature=0.3,
                )

                assert_llm_response_valid(response, min_length=20)


class TestRAGPipelinePerformance:
    """Test performance characteristics of RAG pipeline."""

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

        # 1. Document retrieval
        doc = get_real_document(mongodb_database)
        retrieval_time = time.time() - start_time

        assert doc is not None

        # 2. Generate response
        gen_start = time.time()
        content = doc.get("content", "Test content")[:500]

        prompt = f"""Summarize this content:

{content}

SUMMARY:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
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


class TestRAGPipelineQuality:
    """Test quality of RAG pipeline outputs."""

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
        # Get a real document
        doc = get_real_document(mongodb_database)
        assert doc is not None

        content = doc.get("content", "")[:500]
        title = doc.get("title", "Document")

        prompt = f"""Summarize the following document in 2-3 sentences.

TITLE: {title}

CONTENT:
{content}

SUMMARY:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
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


class TestVectorSearchIntegration:
    """Test vector search integration with real data."""

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    @pytest.mark.requires_mongodb
    def test_vector_similarity_on_real_data(self, mongodb_database):
        """Test vector similarity calculation on real documents."""
        collection = mongodb_database["documents"]

        # Get documents with vectors
        docs = list(collection.find({"vector": {"$exists": True}}).limit(5))

        if len(docs) >= 2:
            # Calculate similarities between pairs
            for i in range(len(docs) - 1):
                vec1 = docs[i]["vector"]
                vec2 = docs[i + 1]["vector"]

                similarity = self._cosine_similarity(vec1, vec2)

                # Similarity should be valid
                assert -1.0 <= similarity <= 1.0

    @pytest.mark.requires_mongodb
    def test_retrieve_similar_documents(self, mongodb_database):
        """Test retrieving similar documents by vector."""
        collection = mongodb_database["documents"]

        # Get a document with vector
        query_doc = collection.find_one({"vector": {"$exists": True}})

        if query_doc and "vector" in query_doc:
            query_vector = query_doc["vector"]

            # Get all documents with vectors
            docs = list(collection.find({"vector": {"$exists": True}}).limit(20))

            # Calculate similarities
            for doc in docs:
                doc["similarity"] = self._cosine_similarity(
                    query_vector, doc["vector"]
                )

            # Sort by similarity
            docs.sort(key=lambda x: x["similarity"], reverse=True)

            # Top result should be the query document itself (similarity = 1.0)
            assert docs[0]["_id"] == query_doc["_id"]
            assert abs(docs[0]["similarity"] - 1.0) < 0.001
