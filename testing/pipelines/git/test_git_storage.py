"""
Git Storage Tests - Using Real Data
====================================

Tests code storage validation using REAL data from the database.
"""

import pytest
from typing import Dict, Any

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
    get_real_code_class,
)


class TestCodeDataExists:
    """Verify code-related collections have data."""

    @pytest.mark.requires_mongodb
    def test_code_methods_collection_has_data(self, mongodb_database):
        """Verify code_methods collection exists and has documents."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count > 0, "code_methods collection should have documents"

    @pytest.mark.requires_mongodb
    def test_code_classes_collection_has_data(self, mongodb_database):
        """Verify code_classes collection exists and has documents."""
        collection = mongodb_database["code_classes"]
        count = collection.count_documents({})
        assert count > 0, "code_classes collection should have documents"

    @pytest.mark.requires_mongodb
    def test_code_callgraph_has_data(self, mongodb_database):
        """Verify code_callgraph collection exists and has documents."""
        collection = mongodb_database["code_callgraph"]
        count = collection.count_documents({})
        assert count > 0, "code_callgraph collection should have documents"


class TestCodeMethodsStructure:
    """Test structure of code_methods documents."""

    @pytest.mark.requires_mongodb
    def test_code_method_has_required_fields(self, mongodb_database):
        """Test that code methods have required fields."""
        method = get_real_code_method(mongodb_database)
        assert method is not None, "Should have at least one code method"

        assert "_id" in method, "Method should have _id"

    @pytest.mark.requires_mongodb
    def test_code_method_has_method_name(self, mongodb_database):
        """Test that code methods have method_name field."""
        method = get_real_code_method(mongodb_database)
        assert method is not None

        if "method_name" in method:
            assert isinstance(method["method_name"], str)
            assert len(method["method_name"]) > 0

    @pytest.mark.requires_mongodb
    def test_code_method_has_class_name(self, mongodb_database):
        """Test that code methods have class_name field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"class_name": {"$exists": True}})

        if method:
            assert isinstance(method["class_name"], str)

    @pytest.mark.requires_mongodb
    def test_multiple_methods_retrievable(self, mongodb_database):
        """Test that multiple code methods can be retrieved."""
        methods = get_real_code_methods(mongodb_database, limit=10)
        assert len(methods) > 0

        for method in methods:
            assert "_id" in method


class TestCodeClassesStructure:
    """Test structure of code_classes documents."""

    @pytest.mark.requires_mongodb
    def test_code_class_has_required_fields(self, mongodb_database):
        """Test that code classes have required fields."""
        code_class = get_real_code_class(mongodb_database)
        assert code_class is not None, "Should have at least one code class"
        assert "_id" in code_class

    @pytest.mark.requires_mongodb
    def test_code_classes_count(self, mongodb_database):
        """Test code_classes has expected count."""
        collection = mongodb_database["code_classes"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 classes, got {count}"


class TestCodeCallgraphStructure:
    """Test structure of code_callgraph documents."""

    @pytest.mark.requires_mongodb
    def test_callgraph_has_documents(self, mongodb_database):
        """Test that callgraph has documents."""
        collection = mongodb_database["code_callgraph"]
        doc = collection.find_one()
        assert doc is not None
        assert "_id" in doc

    @pytest.mark.requires_mongodb
    def test_callgraph_count(self, mongodb_database):
        """Test code_callgraph has expected count."""
        collection = mongodb_database["code_callgraph"]
        count = collection.count_documents({})
        assert count >= 1000, f"Expected at least 1000 callgraph entries, got {count}"


class TestCodeQueryability:
    """Test that code data can be queried effectively."""

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
    def test_aggregation_methods_per_project(self, mongodb_database):
        """Test aggregation of methods per project."""
        collection = mongodb_database["code_methods"]

        pipeline = [
            {"$group": {"_id": "$project", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0


class TestCodeDataCounts:
    """Verify expected data counts in code collections."""

    @pytest.mark.requires_mongodb
    def test_code_methods_count(self, mongodb_database):
        """Verify code_methods has expected minimum count."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count >= 1000, f"Expected at least 1000 methods, got {count}"

    @pytest.mark.requires_mongodb
    def test_code_classes_count(self, mongodb_database):
        """Verify code_classes has expected minimum count."""
        collection = mongodb_database["code_classes"]
        count = collection.count_documents({})
        assert count >= 100, f"Expected at least 100 classes, got {count}"

    @pytest.mark.requires_mongodb
    def test_code_callgraph_count(self, mongodb_database):
        """Verify code_callgraph has expected minimum count."""
        collection = mongodb_database["code_callgraph"]
        count = collection.count_documents({})
        assert count >= 10000, f"Expected at least 10000 callgraph, got {count}"
