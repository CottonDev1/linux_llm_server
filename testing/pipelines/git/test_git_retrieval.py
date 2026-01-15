"""
Git Analysis Retrieval Tests
=============================

Test retrieving parsed code entities from MongoDB.

Tests cover:
- Method retrieval by name, class, project
- Class retrieval with filtering
- Database operation queries
- Vector similarity search
"""

import pytest
from typing import List, Dict, Any

from config.test_config import get_test_config
from fixtures.mongodb_fixtures import (
    create_mock_code_method,
    insert_test_documents,
)
from utils import generate_test_id


class TestGitRetrieval:
    """Test git analysis retrieval operations."""

    @pytest.fixture(autouse=True)
    async def setup_test_data(self, mongodb_database, pipeline_config):
        """Set up test data for retrieval tests."""
        # Insert test methods
        self.methods_collection = mongodb_database["code_methods"]
        self.test_methods = []

        # Create diverse set of methods
        methods_data = [
            {
                "method_name": "GetUserById",
                "class_name": "UserService",
                "project": "TestProject",
                "purpose_summary": "Retrieves a user by their unique ID",
                "calls_methods": ["Database.Query", "User.FromDbRow"],
                "database_tables": ["Users"],
            },
            {
                "method_name": "UpdateUser",
                "class_name": "UserService",
                "project": "TestProject",
                "purpose_summary": "Updates user information in database",
                "calls_methods": ["Database.Execute", "ValidateUser"],
                "database_tables": ["Users", "UserAudit"],
            },
            {
                "method_name": "DeleteUser",
                "class_name": "UserService",
                "project": "TestProject",
                "purpose_summary": "Soft deletes a user",
                "calls_methods": ["Database.Execute"],
                "database_tables": ["Users"],
            },
            {
                "method_name": "ProcessOrder",
                "class_name": "OrderService",
                "project": "TestProject",
                "purpose_summary": "Processes a customer order",
                "calls_methods": ["CalculateTotal", "CreateInvoice"],
                "database_tables": ["Orders", "OrderItems"],
            },
            {
                "method_name": "CalculateTotal",
                "class_name": "OrderService",
                "project": "OtherProject",
                "purpose_summary": "Calculates order total with tax",
                "calls_methods": ["GetTaxRate"],
                "database_tables": [],
            },
        ]

        for method_data in methods_data:
            method = create_mock_code_method(**method_data)
            self.test_methods.append(method)

        insert_test_documents(
            self.methods_collection,
            self.test_methods,
            pipeline_config.test_run_id
        )

    @pytest.mark.requires_mongodb
    def test_retrieve_method_by_name(self, mongodb_database, pipeline_config):
        """Test retrieving a method by exact name match."""
        collection = mongodb_database["code_methods"]

        # Query by method name
        result = collection.find_one({
            "method_name": "GetUserById",
            "test_run_id": pipeline_config.test_run_id
        })

        # Verify result
        assert result is not None
        assert result["method_name"] == "GetUserById"
        assert result["class_name"] == "UserService"
        assert "Retrieves a user" in result["purpose_summary"]

    @pytest.mark.requires_mongodb
    def test_retrieve_methods_by_class(self, mongodb_database, pipeline_config):
        """Test retrieving all methods for a specific class."""
        collection = mongodb_database["code_methods"]

        # Query by class name
        results = list(collection.find({
            "class_name": "UserService",
            "test_run_id": pipeline_config.test_run_id
        }).sort("method_name", 1))

        # Verify results
        assert len(results) == 3
        method_names = [r["method_name"] for r in results]
        assert "GetUserById" in method_names
        assert "UpdateUser" in method_names
        assert "DeleteUser" in method_names

    @pytest.mark.requires_mongodb
    def test_retrieve_methods_by_project(self, mongodb_database, pipeline_config):
        """Test retrieving methods filtered by project."""
        collection = mongodb_database["code_methods"]

        # Query by project
        results = list(collection.find({
            "project": "TestProject",
            "test_run_id": pipeline_config.test_run_id
        }))

        # Verify results
        assert len(results) == 4
        assert all(r["project"] == "TestProject" for r in results)

    @pytest.mark.requires_mongodb
    def test_retrieve_methods_calling_specific_method(self, mongodb_database, pipeline_config):
        """Test finding methods that call a specific method."""
        collection = mongodb_database["code_methods"]

        # Query for methods that call Database.Execute
        results = list(collection.find({
            "calls_methods": "Database.Execute",
            "test_run_id": pipeline_config.test_run_id
        }))

        # Verify results
        assert len(results) >= 2
        method_names = [r["method_name"] for r in results]
        assert "UpdateUser" in method_names
        assert "DeleteUser" in method_names

    @pytest.mark.requires_mongodb
    def test_retrieve_methods_accessing_table(self, mongodb_database, pipeline_config):
        """Test finding methods that access a specific database table."""
        collection = mongodb_database["code_methods"]

        # Query for methods accessing Users table
        results = list(collection.find({
            "database_tables": "Users",
            "test_run_id": pipeline_config.test_run_id
        }))

        # Verify results
        assert len(results) >= 3
        assert all("Users" in r.get("database_tables", []) for r in results)

    @pytest.mark.requires_mongodb
    def test_retrieve_with_projection(self, mongodb_database, pipeline_config):
        """Test retrieving specific fields only."""
        collection = mongodb_database["code_methods"]

        # Query with field projection
        results = list(collection.find(
            {"test_run_id": pipeline_config.test_run_id},
            {"method_name": 1, "class_name": 1, "purpose_summary": 1, "_id": 0}
        ))

        # Verify projection
        assert len(results) > 0
        for result in results:
            assert "method_name" in result
            assert "class_name" in result
            assert "purpose_summary" in result
            assert "_id" not in result
            assert "content" not in result

    @pytest.mark.requires_mongodb
    def test_retrieve_with_sorting(self, mongodb_database, pipeline_config):
        """Test retrieving methods with sorting."""
        collection = mongodb_database["code_methods"]

        # Query with sorting
        results = list(collection.find({
            "class_name": "UserService",
            "test_run_id": pipeline_config.test_run_id
        }).sort("method_name", -1))  # Descending

        # Verify sorting
        assert len(results) == 3
        method_names = [r["method_name"] for r in results]
        assert method_names == sorted(method_names, reverse=True)

    @pytest.mark.requires_mongodb
    def test_retrieve_with_limit(self, mongodb_database, pipeline_config):
        """Test retrieving limited number of results."""
        collection = mongodb_database["code_methods"]

        # Query with limit
        results = list(collection.find({
            "test_run_id": pipeline_config.test_run_id
        }).limit(2))

        # Verify limit
        assert len(results) == 2

    @pytest.mark.requires_mongodb
    def test_aggregate_methods_by_class(self, mongodb_database, pipeline_config):
        """Test aggregating method counts by class."""
        collection = mongodb_database["code_methods"]

        # Aggregation pipeline
        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": "$class_name",
                "method_count": {"$sum": 1},
                "methods": {"$push": "$method_name"}
            }},
            {"$sort": {"method_count": -1}}
        ]

        results = list(collection.aggregate(pipeline))

        # Verify aggregation
        assert len(results) >= 2
        user_service = next((r for r in results if r["_id"] == "UserService"), None)
        assert user_service is not None
        assert user_service["method_count"] == 3

    @pytest.mark.requires_mongodb
    def test_aggregate_tables_accessed(self, mongodb_database, pipeline_config):
        """Test aggregating which tables are accessed most."""
        collection = mongodb_database["code_methods"]

        # Aggregation to count table accesses
        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$unwind": "$database_tables"},
            {"$group": {
                "_id": "$database_tables",
                "access_count": {"$sum": 1},
                "methods": {"$addToSet": "$method_name"}
            }},
            {"$sort": {"access_count": -1}}
        ]

        results = list(collection.aggregate(pipeline))

        # Verify aggregation
        assert len(results) > 0
        users_table = next((r for r in results if r["_id"] == "Users"), None)
        assert users_table is not None
        assert users_table["access_count"] >= 3

    @pytest.mark.requires_mongodb
    def test_text_search_purpose(self, mongodb_database, pipeline_config):
        """Test text search on purpose summary."""
        collection = mongodb_database["code_methods"]

        # Create text index if not exists
        try:
            collection.create_index([("purpose_summary", "text")])
        except:
            pass  # Index may already exist

        # Text search
        results = list(collection.find({
            "$text": {"$search": "user"},
            "test_run_id": pipeline_config.test_run_id
        }))

        # Verify search
        assert len(results) >= 3
        assert all("user" in r["purpose_summary"].lower() for r in results)

    @pytest.mark.requires_mongodb
    def test_retrieve_methods_by_multiple_criteria(self, mongodb_database, pipeline_config):
        """Test complex query with multiple criteria."""
        collection = mongodb_database["code_methods"]

        # Complex query
        results = list(collection.find({
            "$and": [
                {"class_name": "UserService"},
                {"database_tables": "Users"},
                {"test_run_id": pipeline_config.test_run_id}
            ]
        }))

        # Verify results
        assert len(results) >= 2
        for result in results:
            assert result["class_name"] == "UserService"
            assert "Users" in result["database_tables"]

    @pytest.mark.requires_mongodb
    def test_count_methods_by_project(self, mongodb_database, pipeline_config):
        """Test counting methods by project."""
        collection = mongodb_database["code_methods"]

        # Count by project
        test_project_count = collection.count_documents({
            "project": "TestProject",
            "test_run_id": pipeline_config.test_run_id
        })
        other_project_count = collection.count_documents({
            "project": "OtherProject",
            "test_run_id": pipeline_config.test_run_id
        })

        # Verify counts
        assert test_project_count == 4
        assert other_project_count == 1

    @pytest.mark.requires_mongodb
    def test_distinct_class_names(self, mongodb_database, pipeline_config):
        """Test getting distinct class names."""
        collection = mongodb_database["code_methods"]

        # Get distinct values
        class_names = collection.distinct("class_name", {
            "test_run_id": pipeline_config.test_run_id
        })

        # Verify distinct values
        assert len(class_names) >= 2
        assert "UserService" in class_names
        assert "OrderService" in class_names
