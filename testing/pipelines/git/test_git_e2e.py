"""
Git End-to-End Tests - Using Real Data
=======================================

Tests code pipeline end-to-end functionality using REAL data.
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


class TestCodePipelineDataFlow:
    """Test complete code data flow."""

    @pytest.mark.requires_mongodb
    def test_database_has_required_collections(self, mongodb_database):
        """Verify all required code collections exist with data."""
        required_collections = [
            "code_methods",
            "code_classes",
            "code_callgraph",
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
    def test_class_lookup_flow(self, mongodb_database):
        """Test that classes can be looked up."""
        code_class = get_real_code_class(mongodb_database)
        assert code_class is not None

        collection = mongodb_database["code_classes"]
        retrieved = collection.find_one({"_id": code_class["_id"]})
        assert retrieved is not None


class TestCodeDataIntegrity:
    """Test code data integrity across collections."""

    @pytest.mark.requires_mongodb
    def test_code_methods_have_names(self, mongodb_database):
        """Verify code methods have method names."""
        methods = get_real_code_methods(mongodb_database, limit=20)

        methods_with_name = [m for m in methods if "method_name" in m]
        assert len(methods_with_name) > 0, "Some methods should have names"

    @pytest.mark.requires_mongodb
    def test_code_methods_have_class_info(self, mongodb_database):
        """Verify code methods have class information."""
        collection = mongodb_database["code_methods"]

        methods_with_class = list(
            collection.find({"class_name": {"$exists": True}}).limit(10)
        )

        for method in methods_with_class:
            assert "class_name" in method
            assert isinstance(method["class_name"], str)

    @pytest.mark.requires_mongodb
    def test_code_classes_have_structure(self, mongodb_database):
        """Verify code classes have proper structure."""
        collection = mongodb_database["code_classes"]

        docs = list(collection.find().limit(10))

        for doc in docs:
            assert "_id" in doc


class TestCodeQueryPatterns:
    """Test common code query patterns work with real data."""

    @pytest.mark.requires_mongodb
    def test_query_by_project_name(self, mongodb_database):
        """Test querying code by project name."""
        collection = mongodb_database["code_methods"]
        projects = collection.distinct("project")

        if projects:
            project = projects[0]
            results = list(collection.find({"project": project}).limit(5))
            assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_search_methods_by_pattern(self, mongodb_database):
        """Test searching methods by name pattern."""
        collection = mongodb_database["code_methods"]

        results = list(
            collection.find(
                {"method_name": {"$regex": "^Get", "$options": "i"}}
            ).limit(5)
        )

        assert isinstance(results, list)

    @pytest.mark.requires_mongodb
    def test_aggregation_pipeline_works(self, mongodb_database):
        """Test that aggregation pipelines work on code data."""
        collection = mongodb_database["code_methods"]

        pipeline = [
            {"$group": {"_id": "$project", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 0}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0


class TestCodeTestQueries:
    """Test code test query generation from real data."""

    @pytest.mark.requires_mongodb
    def test_generate_test_queries(self, mongodb_database):
        """Test that test queries can be generated from real methods."""
        queries = get_code_test_queries(mongodb_database)

        assert len(queries) > 0

    @pytest.mark.requires_mongodb
    def test_queries_have_expected_structure(self, mongodb_database):
        """Test that generated queries have expected structure."""
        queries = get_code_test_queries(mongodb_database)

        for q in queries[:5]:
            assert isinstance(q, dict)
            assert len(q) > 0


class TestCodeCollectionIndexes:
    """Test that code collections have proper indexes."""

    @pytest.mark.requires_mongodb
    def test_code_methods_has_indexes(self, mongodb_database):
        """Verify code_methods collection has indexes."""
        collection = mongodb_database["code_methods"]
        indexes = list(collection.list_indexes())

        index_names = [idx["name"] for idx in indexes]
        assert "_id_" in index_names

    @pytest.mark.requires_mongodb
    def test_code_classes_has_indexes(self, mongodb_database):
        """Verify code_classes collection has indexes."""
        collection = mongodb_database["code_classes"]
        indexes = list(collection.list_indexes())

        assert len(indexes) >= 1

    @pytest.mark.requires_mongodb
    def test_code_callgraph_has_indexes(self, mongodb_database):
        """Verify code_callgraph has indexes."""
        collection = mongodb_database["code_callgraph"]
        indexes = list(collection.list_indexes())

        assert len(indexes) >= 1


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
        assert count >= 10000, f"Expected at least 10000 callgraph entries, got {count}"
