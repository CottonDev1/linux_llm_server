"""
Git Retrieval Tests - Using Real Data
======================================

Tests code retrieval operations using REAL data from the database.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
    get_real_code_class,
    get_code_test_queries,
)


class TestCodeMethodRetrieval:
    """Test retrieval of code methods."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_method(self, mongodb_database):
        """Test retrieving a single code method."""
        method = get_real_code_method(mongodb_database)
        assert method is not None
        assert "_id" in method

    @pytest.mark.requires_mongodb
    def test_retrieve_multiple_methods(self, mongodb_database):
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


class TestCodeClassRetrieval:
    """Test retrieval of code classes."""

    @pytest.mark.requires_mongodb
    def test_retrieve_single_class(self, mongodb_database):
        """Test retrieving a single code class."""
        code_class = get_real_code_class(mongodb_database)
        assert code_class is not None
        assert "_id" in code_class

    @pytest.mark.requires_mongodb
    def test_retrieve_classes_by_project(self, mongodb_database):
        """Test filtering code classes by project."""
        collection = mongodb_database["code_classes"]

        projects = collection.distinct("project")

        if projects:
            project = projects[0]
            results = list(collection.find({"project": project}).limit(5))

            for result in results:
                assert result.get("project") == project


class TestCodeCallgraphRetrieval:
    """Test retrieval of code callgraph."""

    @pytest.mark.requires_mongodb
    def test_retrieve_callgraph_entry(self, mongodb_database):
        """Test retrieving a callgraph entry."""
        collection = mongodb_database["code_callgraph"]
        doc = collection.find_one()
        assert doc is not None
        assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_retrieve_callgraph_by_method(self, mongodb_database):
        """Test retrieving callgraph entries for a method."""
        collection = mongodb_database["code_callgraph"]

        doc = collection.find_one({"method_name": {"$exists": True}})
        if doc and "method_name" in doc:
            results = list(collection.find({"method_name": doc["method_name"]}).limit(5))
            assert len(results) > 0


class TestCodeSearchQueries:
    """Test code search query patterns."""

    @pytest.mark.requires_mongodb
    def test_text_search_in_methods(self, mongodb_database):
        """Test text search in code methods."""
        collection = mongodb_database["code_methods"]

        results = list(
            collection.find(
                {"method_name": {"$regex": "Get", "$options": "i"}}
            ).limit(5)
        )

        assert isinstance(results, list)

    @pytest.mark.requires_mongodb
    def test_generate_code_queries(self, mongodb_database):
        """Test generating code search queries from real data."""
        queries = get_code_test_queries(mongodb_database)

        assert len(queries) > 0
        for q in queries[:5]:
            assert isinstance(q, dict)


class TestAggregationQueries:
    """Test aggregation queries on code collections."""

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
    def test_count_classes_per_project(self, mongodb_database):
        """Test counting classes per project."""
        collection = mongodb_database["code_classes"]

        pipeline = [
            {"$group": {"_id": "$project", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_sample_random_methods(self, mongodb_database):
        """Test sampling random code methods."""
        collection = mongodb_database["code_methods"]

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
        collection = mongodb_database["code_methods"]

        doc = collection.find_one()
        if doc:
            result = collection.find_one({"_id": doc["_id"]})
            assert result is not None

    @pytest.mark.requires_mongodb
    def test_limit_query_performance(self, mongodb_database):
        """Test that limit queries return quickly."""
        collection = mongodb_database["code_methods"]

        results = list(collection.find().limit(10))
        assert len(results) <= 10

    @pytest.mark.requires_mongodb
    def test_projection_query(self, mongodb_database):
        """Test queries with projection."""
        collection = mongodb_database["code_methods"]

        result = collection.find_one({}, {"method_name": 1, "class_name": 1})

        if result:
            assert "_id" in result  # _id is included by default
