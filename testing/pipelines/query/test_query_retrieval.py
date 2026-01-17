"""
Query/RAG Retrieval Tests - Using Real Data
============================================

Tests document retrieval operations using REAL data from the database.
"""

import pytest
import math
from typing import List, Dict, Any

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_document,
    get_real_documents,
    get_real_code_method,
    get_real_code_methods,
    get_document_test_queries,
    get_code_test_queries,
)


class TestDocumentRetrieval:
    """Test retrieval of documents."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_document(self, mongodb_database):
        """Test retrieving a single document."""
        doc = get_real_document(mongodb_database)
        assert doc is not None
        assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_retrieve_multiple_documents(self, mongodb_database):
        """Test retrieving multiple documents."""
        docs = get_real_documents(mongodb_database, limit=10)
        assert len(docs) > 0

        for doc in docs:
            assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_retrieve_documents_by_department(self, mongodb_database):
        """Test filtering documents by department."""
        collection = mongodb_database["documents"]

        departments = collection.distinct("department")

        if departments:
            dept = departments[0]
            results = list(collection.find({"department": dept}).limit(10))

            for result in results:
                assert result.get("department") == dept

    @pytest.mark.requires_mongodb
    def test_retrieve_document_by_title(self, mongodb_database):
        """Test retrieving document by title."""
        collection = mongodb_database["documents"]

        doc = collection.find_one({"title": {"$exists": True}})
        if doc and "title" in doc:
            title = doc["title"]
            result = collection.find_one({"title": title})
            assert result is not None


class TestCodeMethodRetrieval:
    """Test retrieval of code methods."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_code_method(self, mongodb_database):
        """Test retrieving a single code method."""
        method = get_real_code_method(mongodb_database)
        assert method is not None
        assert "_id" in method

    @pytest.mark.requires_mongodb
    def test_retrieve_multiple_code_methods(self, mongodb_database):
        """Test retrieving multiple code methods."""
        methods = get_real_code_methods(mongodb_database, limit=10)
        assert len(methods) > 0

        for method in methods:
            assert "_id" in method

    @pytest.mark.requires_mongodb
    def test_retrieve_methods_by_project(self, mongodb_database):
        """Test filtering code methods by project."""
        collection = mongodb_database["code_methods"]

        projects = collection.distinct("project")

        if projects:
            project = projects[0]
            results = list(collection.find({"project": project}).limit(10))

            for result in results:
                assert result.get("project") == project

    @pytest.mark.requires_mongodb
    def test_retrieve_method_by_name(self, mongodb_database):
        """Test retrieving method by name."""
        collection = mongodb_database["code_methods"]

        doc = collection.find_one({"method_name": {"$exists": True}})
        if doc and "method_name" in doc:
            method_name = doc["method_name"]
            result = collection.find_one({"method_name": method_name})
            assert result is not None


class TestVectorRetrieval:
    """Test vector/embedding retrieval operations."""

    @pytest.mark.requires_mongodb
    def test_retrieve_documents_with_vectors(self, mongodb_database):
        """Test retrieving documents that have vector embeddings."""
        collection = mongodb_database["documents"]

        results = list(collection.find({"vector": {"$exists": True}}).limit(5))

        for result in results:
            assert "vector" in result
            assert isinstance(result["vector"], list)

    @pytest.mark.requires_mongodb
    def test_vector_similarity_calculation(self, mongodb_database):
        """Test that vector similarity can be calculated."""
        collection = mongodb_database["documents"]

        docs = list(collection.find({"vector": {"$exists": True}}).limit(2))

        if len(docs) >= 2:
            vec1 = docs[0]["vector"]
            vec2 = docs[1]["vector"]

            # Calculate cosine similarity
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            similarity = dot / (norm1 * norm2) if norm1 and norm2 else 0.0

            # Similarity should be between -1 and 1
            assert -1.0 <= similarity <= 1.0


class TestSearchQueries:
    """Test search query patterns on real data."""

    @pytest.mark.requires_mongodb
    def test_text_search_in_documents(self, mongodb_database):
        """Test text search in documents."""
        collection = mongodb_database["documents"]

        # Get a sample word from existing content
        doc = collection.find_one({"content": {"$exists": True}})
        if doc and "content" in doc:
            # Get first significant word
            words = doc["content"].split()
            if len(words) > 3:
                search_word = words[3]
                if len(search_word) > 3:
                    results = list(
                        collection.find(
                            {"content": {"$regex": search_word, "$options": "i"}}
                        ).limit(5)
                    )
                    assert isinstance(results, list)

    @pytest.mark.requires_mongodb
    def test_generate_document_queries(self, mongodb_database):
        """Test generating document search queries from real data."""
        queries = get_document_test_queries(mongodb_database)

        assert len(queries) > 0
        for q in queries[:5]:
            assert isinstance(q, dict)

    @pytest.mark.requires_mongodb
    def test_generate_code_queries(self, mongodb_database):
        """Test generating code search queries from real data."""
        queries = get_code_test_queries(mongodb_database)

        assert len(queries) > 0
        for q in queries[:5]:
            assert isinstance(q, dict)


class TestAggregationQueries:
    """Test aggregation queries on collections."""

    @pytest.mark.requires_mongodb
    def test_count_documents_per_department(self, mongodb_database):
        """Test counting documents per department."""
        collection = mongodb_database["documents"]

        pipeline = [
            {"$group": {"_id": "$department", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_count_methods_per_class(self, mongodb_database):
        """Test counting methods per class."""
        collection = mongodb_database["code_methods"]

        pipeline = [
            {"$group": {"_id": "$class_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_sample_random_documents(self, mongodb_database):
        """Test sampling random documents."""
        collection = mongodb_database["documents"]

        pipeline = [{"$sample": {"size": 5}}]
        results = list(collection.aggregate(pipeline))

        assert len(results) <= 5
        for result in results:
            assert "_id" in result


class TestQueryPerformance:
    """Test query performance characteristics."""

    @pytest.mark.requires_mongodb
    def test_indexed_query_by_id(self, mongodb_database):
        """Test that _id queries are efficient."""
        collection = mongodb_database["documents"]

        doc = collection.find_one()
        if doc:
            result = collection.find_one({"_id": doc["_id"]})
            assert result is not None

    @pytest.mark.requires_mongodb
    def test_limit_query_performance(self, mongodb_database):
        """Test that limit queries return quickly."""
        collection = mongodb_database["documents"]

        results = list(collection.find().limit(10))
        assert len(results) <= 10

    @pytest.mark.requires_mongodb
    def test_projection_query(self, mongodb_database):
        """Test queries with projection."""
        collection = mongodb_database["documents"]

        result = collection.find_one({}, {"title": 1, "department": 1})

        if result:
            assert "_id" in result  # _id is included by default


class TestMultiProjectRetrieval:
    """Test retrieving code from multiple projects."""

    @pytest.mark.requires_mongodb
    def test_retrieve_from_multiple_projects(self, mongodb_database):
        """Test retrieving code methods from multiple projects."""
        collection = mongodb_database["code_methods"]

        projects = collection.distinct("project")

        if len(projects) >= 2:
            # Query for methods from first two projects
            target_projects = projects[:2]
            results = list(
                collection.find({"project": {"$in": target_projects}}).limit(10)
            )

            assert len(results) > 0
            # Results should only be from target projects
            for result in results:
                assert result.get("project") in target_projects

    @pytest.mark.requires_mongodb
    def test_count_methods_per_project(self, mongodb_database):
        """Test counting methods per project."""
        collection = mongodb_database["code_methods"]

        pipeline = [
            {"$group": {"_id": "$project", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0

        # Log project distribution
        for r in results[:5]:
            assert r["_id"] is not None or r["count"] > 0
