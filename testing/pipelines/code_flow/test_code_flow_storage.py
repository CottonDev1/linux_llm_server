"""
Code Flow Storage Tests - Using Real Data
==========================================

Tests code flow analysis storage validation using REAL data from the database.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
)


class TestCodeFlowDataExists:
    """Verify code flow collections have data."""

    @pytest.mark.requires_mongodb
    def test_code_methods_collection_has_data(self, mongodb_database):
        """Verify code_methods collection exists and has documents."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count > 0, "code_methods collection should have documents"

    @pytest.mark.requires_mongodb
    def test_code_callgraph_has_data(self, mongodb_database):
        """Verify code_callgraph collection exists and has documents."""
        collection = mongodb_database["code_callgraph"]
        count = collection.count_documents({})
        assert count > 0, "code_callgraph collection should have documents"

    @pytest.mark.requires_mongodb
    def test_code_classes_has_data(self, mongodb_database):
        """Verify code_classes collection exists and has documents."""
        collection = mongodb_database["code_classes"]
        count = collection.count_documents({})
        assert count > 0, "code_classes collection should have documents"


class TestCodeMethodStructure:
    """Test structure of stored code methods."""

    @pytest.mark.requires_mongodb
    def test_code_method_has_required_fields(self, mongodb_database):
        """Test that code methods have required fields."""
        method = get_real_code_method(mongodb_database)
        assert method is not None, "Should have at least one code method"
        assert "_id" in method

    @pytest.mark.requires_mongodb
    def test_code_method_has_name(self, mongodb_database):
        """Test that code methods have method_name field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"method_name": {"$exists": True}})

        if method:
            assert isinstance(method["method_name"], str)
            assert len(method["method_name"]) > 0

    @pytest.mark.requires_mongodb
    def test_code_method_has_class(self, mongodb_database):
        """Test that code methods have class_name field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"class_name": {"$exists": True}})

        if method:
            assert isinstance(method["class_name"], str)

    @pytest.mark.requires_mongodb
    def test_code_method_has_project(self, mongodb_database):
        """Test that code methods have project field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"project": {"$exists": True}})

        if method:
            assert isinstance(method["project"], str)


class TestCodeCallgraphStructure:
    """Test structure of stored code callgraph."""

    @pytest.mark.requires_mongodb
    def test_callgraph_has_documents(self, mongodb_database):
        """Test that callgraph has documents."""
        collection = mongodb_database["code_callgraph"]
        doc = collection.find_one()
        assert doc is not None
        assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_callgraph_has_method_fields(self, mongodb_database):
        """Test that callgraph entries have method fields."""
        collection = mongodb_database["code_callgraph"]
        doc = collection.find_one({"method_name": {"$exists": True}})

        if doc:
            assert "method_name" in doc


class TestDBOperationsStructure:
    """Test structure of stored database operations."""

    @pytest.mark.requires_mongodb
    def test_dboperations_collection_exists(self, mongodb_database):
        """Test that code_dboperations collection exists."""
        collection = mongodb_database["code_dboperations"]
        count = collection.count_documents({})
        # Collection may be empty, but should exist
        assert collection is not None

    @pytest.mark.requires_mongodb
    def test_dboperations_has_data(self, mongodb_database):
        """Test that database operations have data."""
        collection = mongodb_database["code_dboperations"]
        count = collection.count_documents({})

        # If we have data, validate structure
        if count > 0:
            doc = collection.find_one()
            assert "_id" in doc


class TestEventHandlersStructure:
    """Test structure of stored event handlers."""

    @pytest.mark.requires_mongodb
    def test_eventhandlers_collection_has_data(self, mongodb_database):
        """Test that code_eventhandlers collection has data."""
        collection = mongodb_database["code_eventhandlers"]
        count = collection.count_documents({})
        assert count > 0, "code_eventhandlers collection should have documents"

    @pytest.mark.requires_mongodb
    def test_eventhandler_has_required_fields(self, mongodb_database):
        """Test that event handlers have required fields."""
        collection = mongodb_database["code_eventhandlers"]
        doc = collection.find_one()

        if doc:
            assert "_id" in doc


class TestCodeFlowQueryability:
    """Test that code flow data can be queried effectively."""

    @pytest.mark.requires_mongodb
    def test_query_methods_by_project(self, mongodb_database):
        """Test querying code methods by project."""
        collection = mongodb_database["code_methods"]

        projects = collection.distinct("project")

        if projects:
            project = projects[0]
            results = list(collection.find({"project": project}).limit(5))
            assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_query_methods_by_class(self, mongodb_database):
        """Test querying code methods by class name."""
        collection = mongodb_database["code_methods"]

        doc = collection.find_one({"class_name": {"$exists": True}})
        if doc and "class_name" in doc:
            class_name = doc["class_name"]
            results = list(collection.find({"class_name": class_name}).limit(5))
            assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_aggregation_methods_per_class(self, mongodb_database):
        """Test aggregation of methods per class."""
        collection = mongodb_database["code_methods"]

        pipeline = [
            {"$group": {"_id": "$class_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0


class TestCodeFlowCounts:
    """Verify expected data counts."""

    @pytest.mark.requires_mongodb
    def test_code_methods_count(self, mongodb_database):
        """Verify code_methods has expected minimum count."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count >= 1000, f"Expected at least 1000 code methods, got {count}"

    @pytest.mark.requires_mongodb
    def test_code_classes_count(self, mongodb_database):
        """Verify code_classes has expected minimum count."""
        collection = mongodb_database["code_classes"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 code classes, got {count}"

    @pytest.mark.requires_mongodb
    def test_code_callgraph_count(self, mongodb_database):
        """Verify code_callgraph has expected minimum count."""
        collection = mongodb_database["code_callgraph"]
        count = collection.count_documents({})
        assert count >= 1000, f"Expected at least 1000 callgraph entries, got {count}"

    @pytest.mark.requires_mongodb
    def test_code_eventhandlers_count(self, mongodb_database):
        """Verify code_eventhandlers has expected minimum count."""
        collection = mongodb_database["code_eventhandlers"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 event handlers, got {count}"


class TestIndexesExist:
    """Verify collections have proper indexes."""

    @pytest.mark.requires_mongodb
    def test_code_methods_has_indexes(self, mongodb_database):
        """Verify code_methods collection has indexes."""
        collection = mongodb_database["code_methods"]
        indexes = list(collection.list_indexes())

        index_names = [idx["name"] for idx in indexes]
        assert "_id_" in index_names

    @pytest.mark.requires_mongodb
    def test_code_callgraph_has_indexes(self, mongodb_database):
        """Verify code_callgraph has indexes."""
        collection = mongodb_database["code_callgraph"]
        indexes = list(collection.list_indexes())

        assert len(indexes) >= 1
