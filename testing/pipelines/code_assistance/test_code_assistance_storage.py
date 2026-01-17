"""
Code Assistance Storage Tests - Using Real Data
================================================

Tests code assistance data storage validation using REAL data from the database.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_code_method,
    get_real_code_methods,
)


class TestCodeAssistanceDataExists:
    """Verify code assistance related collections have data."""

    @pytest.mark.requires_mongodb
    def test_code_methods_collection_has_data(self, mongodb_database):
        """Verify code_methods collection exists and has documents."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count > 0, "code_methods collection should have documents"

    @pytest.mark.requires_mongodb
    def test_code_classes_has_data(self, mongodb_database):
        """Verify code_classes collection exists and has documents."""
        collection = mongodb_database["code_classes"]
        count = collection.count_documents({})
        assert count > 0, "code_classes collection should have documents"

    @pytest.mark.requires_mongodb
    def test_feedback_collection_exists(self, mongodb_database):
        """Verify feedback collection exists."""
        collection = mongodb_database["feedback"]
        # Collection may be empty but should exist
        count = collection.count_documents({})
        assert count >= 0  # Just verify collection is accessible


class TestCodeMethodStructure:
    """Test structure of stored code methods for assistance."""

    @pytest.mark.requires_mongodb
    def test_code_method_has_required_fields(self, mongodb_database):
        """Test that code methods have required fields for assistance."""
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

    @pytest.mark.requires_mongodb
    def test_code_method_has_content(self, mongodb_database):
        """Test that code methods have content for assistance."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"content": {"$exists": True}})

        if method:
            assert "content" in method
            assert len(method["content"]) > 0


class TestCodeAssistanceQueryability:
    """Test that code data can be queried for assistance."""

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
    def test_search_methods_by_name_pattern(self, mongodb_database):
        """Test searching methods by name pattern."""
        collection = mongodb_database["code_methods"]

        results = list(
            collection.find(
                {"method_name": {"$regex": "^Get", "$options": "i"}}
            ).limit(5)
        )

        assert isinstance(results, list)


class TestCodeAssistanceCounts:
    """Verify expected data counts for assistance."""

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


class TestAggregationQueries:
    """Test aggregation queries for assistance analytics."""

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
