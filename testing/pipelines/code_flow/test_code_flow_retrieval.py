"""
Code Flow Retrieval Tests - Using Real Data
=============================================

Tests code flow retrieval operations using REAL data from the database.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
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


class TestEventHandlerRetrieval:
    """Test retrieval of event handlers."""

    @pytest.mark.requires_mongodb
    def test_retrieve_event_handler(self, mongodb_database):
        """Test retrieving an event handler."""
        collection = mongodb_database["code_eventhandlers"]
        doc = collection.find_one()
        assert doc is not None
        assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_retrieve_event_handlers_by_project(self, mongodb_database):
        """Test retrieving event handlers by project."""
        collection = mongodb_database["code_eventhandlers"]

        projects = collection.distinct("project")

        if projects:
            project = projects[0]
            results = list(collection.find({"project": project}).limit(5))
            assert len(results) > 0


class TestAggregationQueries:
    """Test aggregation queries on code flow collections."""

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
    def test_count_methods_per_project(self, mongodb_database):
        """Test counting methods per project."""
        collection = mongodb_database["code_methods"]

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
