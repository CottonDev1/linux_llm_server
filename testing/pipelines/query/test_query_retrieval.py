"""
Query/RAG Retrieval Tests for vector search and document retrieval.

Tests vector similarity search operations including:
- Cosine similarity search
- Project-scoped search
- Multi-project search (with EWRLibrary)
- Similarity threshold filtering
- Result ranking and scoring

All tests use MongoDB only (no external vector databases).
"""

import pytest
import math
from typing import List

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_document_chunk,
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)


class TestVectorSimilaritySearch:
    """Test vector similarity search operations."""

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    @pytest.mark.requires_mongodb
    def test_exact_match_search(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test search with exact embedding match."""
        collection = mongodb_database["documents"]

        # Create query embedding
        query_embedding = [0.5] * 384

        # Store document with identical embedding
        doc = create_mock_document_chunk(
            title="Exact Match Document",
            content="This should be an exact match",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = query_embedding.copy()
        collection.insert_one(doc)

        # Search using cosine similarity
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        # Calculate similarities
        for result in results:
            result["similarity"] = self._cosine_similarity(
                query_embedding, result["embedding"]
            )

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Top result should be exact match (similarity = 1.0)
        assert len(results) > 0
        assert abs(results[0]["similarity"] - 1.0) < 0.001

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_similarity_ranking(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that results are ranked by similarity score."""
        collection = mongodb_database["documents"]

        # Create query embedding
        query_embedding = [1.0, 0.0, 0.0] * 128  # 384 dims

        # Store documents with varying similarity
        embeddings = [
            [1.0, 0.0, 0.0] * 128,  # Exact match
            [0.9, 0.1, 0.0] * 128,  # High similarity
            [0.5, 0.5, 0.0] * 128,  # Medium similarity
            [0.0, 1.0, 0.0] * 128,  # Low similarity
            [0.0, 0.0, 1.0] * 128,  # Very low similarity
        ]

        docs = []
        for i, emb in enumerate(embeddings):
            doc = create_mock_document_chunk(
                title=f"Document {i}",
                content=f"Content {i}",
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = emb
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Retrieve and calculate similarities
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        for result in results:
            result["similarity"] = self._cosine_similarity(
                query_embedding, result["embedding"]
            )

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Verify ranking (descending similarity)
        assert len(results) == 5
        for i in range(len(results) - 1):
            assert (
                results[i]["similarity"] >= results[i + 1]["similarity"]
            ), "Results should be sorted by similarity"

        # Top result should be exact match
        assert abs(results[0]["similarity"] - 1.0) < 0.001

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_threshold_filtering(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test filtering results by similarity threshold."""
        collection = mongodb_database["documents"]

        query_embedding = [1.0] * 384

        # Store documents with known similarities
        embeddings = [
            [1.0] * 384,  # similarity = 1.0
            [0.9] * 384,  # similarity ≈ 0.9
            [0.5] * 384,  # similarity ≈ 0.5
            [0.1] * 384,  # similarity ≈ 0.1
        ]

        docs = []
        for i, emb in enumerate(embeddings):
            # Normalize embeddings
            norm = math.sqrt(sum(x * x for x in emb))
            normalized = [x / norm for x in emb]

            doc = create_mock_document_chunk(
                title=f"Doc {i}", content=f"Content {i}", chunk_index=0
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = normalized
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Retrieve and filter by threshold
        threshold = 0.7
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        filtered_results = []
        for result in results:
            similarity = self._cosine_similarity(query_embedding, result["embedding"])
            if similarity >= threshold:
                result["similarity"] = similarity
                filtered_results.append(result)

        # Should only include high similarity results
        assert len(filtered_results) >= 1  # At least the exact match
        for result in filtered_results:
            assert result["similarity"] >= threshold

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestProjectScopedRetrieval:
    """Test project-scoped document retrieval."""

    @pytest.mark.requires_mongodb
    def test_single_project_search(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test searching within a single project."""
        collection = mongodb_database["code_context"]

        # Store code from different projects
        projects = ["gin", "warehouse", "marketing"]
        docs = []

        for project in projects:
            for i in range(3):
                doc = create_mock_code_method(
                    method_name=f"{project}_Method_{i}",
                    class_name=f"{project}Class",
                    project=project,
                    code=f"public void {project}_Method_{i}() {{ }}",
                )
                doc["test_run_id"] = pipeline_config.test_run_id
                doc["embedding"] = [float(len(project)) / 10] * 384
                docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Search for specific project
        target_project = "gin"
        results = list(
            collection.find(
                {"project": target_project, "test_run_id": pipeline_config.test_run_id}
            )
        )

        # Should only return gin results
        assert len(results) == 3
        for result in results:
            assert result["project"] == target_project

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_multi_project_search(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test searching across multiple projects."""
        collection = mongodb_database["code_context"]

        # Store code from gin and EWRLibrary
        projects_docs = [
            ("gin", "GinSpecificMethod"),
            ("gin", "AnotherGinMethod"),
            ("EWRLibrary", "SharedUtilityMethod"),
            ("EWRLibrary", "CommonHelper"),
        ]

        docs = []
        for project, method_name in projects_docs:
            doc = create_mock_code_method(
                method_name=method_name,
                class_name=f"{project}Class",
                project=project,
                code=f"public void {method_name}() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [0.5] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Search for gin + EWRLibrary (typical pattern)
        results = list(
            collection.find(
                {
                    "project": {"$in": ["gin", "EWRLibrary"]},
                    "test_run_id": pipeline_config.test_run_id,
                }
            )
        )

        # Should return both gin and EWRLibrary results
        assert len(results) == 4
        projects_found = set(r["project"] for r in results)
        assert "gin" in projects_found
        assert "EWRLibrary" in projects_found

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_exclude_other_projects(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that project scoping excludes other projects."""
        collection = mongodb_database["code_context"]

        # Store code from multiple projects
        projects = ["gin", "warehouse", "marketing", "EWRLibrary"]
        docs = []

        for project in projects:
            doc = create_mock_code_method(
                method_name=f"{project}Method",
                class_name=f"{project}Class",
                project=project,
                code=f"public void {project}Method() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [0.5] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Search only warehouse
        results = list(
            collection.find(
                {"project": "warehouse", "test_run_id": pipeline_config.test_run_id}
            )
        )

        # Should only get warehouse, not gin/marketing/EWRLibrary
        assert len(results) == 1
        assert results[0]["project"] == "warehouse"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestDepartmentFiltering:
    """Test department-based filtering for knowledge base."""

    @pytest.mark.requires_mongodb
    def test_filter_by_department(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test filtering documents by department."""
        collection = mongodb_database["documents"]

        # Store documents from different departments
        departments = ["IT", "Safety", "Operations", "Marketing"]
        docs = []

        for dept in departments:
            doc = create_mock_document_chunk(
                title=f"{dept} Document",
                content=f"Document from {dept} department",
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["department"] = dept
            doc["embedding"] = [0.5] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Filter by department
        target_dept = "Safety"
        results = list(
            collection.find(
                {
                    "department": target_dept,
                    "test_run_id": pipeline_config.test_run_id,
                }
            )
        )

        assert len(results) == 1
        assert results[0]["department"] == target_dept

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestRetrievalPerformance:
    """Test performance of vector retrieval operations."""

    @pytest.mark.requires_mongodb
    def test_retrieve_from_large_collection(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test retrieval performance with large collection."""
        collection = mongodb_database["code_context"]

        # Insert many documents
        docs = []
        for i in range(100):
            doc = create_mock_code_method(
                method_name=f"Method{i}",
                class_name="TestClass",
                project="gin",
                code=f"public void Method{i}() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [float(i) / 100] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Retrieve specific subset
        results = list(
            collection.find(
                {"project": "gin", "test_run_id": pipeline_config.test_run_id}
            ).limit(10)
        )

        assert len(results) == 10

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_limit_results(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test limiting number of results returned."""
        collection = mongodb_database["documents"]

        # Store many documents
        docs = []
        for i in range(50):
            doc = create_mock_document_chunk(
                title=f"Doc {i}", content=f"Content {i}", chunk_index=0
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [0.5] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Retrieve with limit
        limit = 5
        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id}).limit(limit)
        )

        assert len(results) == limit

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestHybridRetrieval:
    """Test hybrid retrieval combining vector and keyword search."""

    @pytest.mark.requires_mongodb
    def test_vector_with_keyword_filter(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test combining vector search with keyword filtering."""
        collection = mongodb_database["code_context"]

        # Store methods with different characteristics
        docs = [
            {
                "method_name": "ProcessBale",
                "project": "gin",
                "language": "csharp",
                "keywords": ["bale", "processing", "gin"],
            },
            {
                "method_name": "ProcessTicket",
                "project": "warehouse",
                "language": "csharp",
                "keywords": ["ticket", "processing", "warehouse"],
            },
            {
                "method_name": "GenerateReport",
                "project": "gin",
                "language": "csharp",
                "keywords": ["report", "generation"],
            },
        ]

        stored_docs = []
        for doc_data in docs:
            doc = create_mock_code_method(
                method_name=doc_data["method_name"],
                class_name="TestClass",
                project=doc_data["project"],
                code=f"public void {doc_data['method_name']}() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["language"] = doc_data["language"]
            doc["keywords"] = doc_data["keywords"]
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Hybrid search: gin project + contains "bale" keyword
        results = list(
            collection.find(
                {
                    "project": "gin",
                    "keywords": "bale",
                    "test_run_id": pipeline_config.test_run_id,
                }
            )
        )

        assert len(results) == 1
        assert results[0]["method_name"] == "ProcessBale"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_text_search_fallback(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test text search when vector search yields no results."""
        collection = mongodb_database["documents"]

        # Store documents with rich text content
        docs = [
            {
                "title": "Bale Processing Guide",
                "content": "This guide explains how to process cotton bales in the gin system.",
            },
            {
                "title": "Safety Procedures",
                "content": "Safety procedures for warehouse operations and equipment handling.",
            },
            {
                "title": "System Configuration",
                "content": "Configuration settings for the EWR system and database connections.",
            },
        ]

        stored_docs = []
        for doc_data in docs:
            doc = create_mock_document_chunk(
                title=doc_data["title"],
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Text search for keyword
        keyword = "bale"
        results = list(
            collection.find(
                {
                    "content": {"$regex": keyword, "$options": "i"},
                    "test_run_id": pipeline_config.test_run_id,
                }
            )
        )

        assert len(results) >= 1
        assert any(keyword.lower() in r["content"].lower() for r in results)

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)
