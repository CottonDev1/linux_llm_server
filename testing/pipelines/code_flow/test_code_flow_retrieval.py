"""
Code Flow Retrieval Tests
==========================

Test retrieving code flow analysis data including database operations,
call chains, and UI event mappings.
"""

import pytest
from datetime import datetime

from fixtures.mongodb_fixtures import insert_test_documents
from utils import generate_test_id


class TestCodeFlowRetrieval:
    """Test code flow retrieval operations."""

    @pytest.fixture(autouse=True)
    async def setup_test_data(self, mongodb_database, pipeline_config):
        """Set up test data for retrieval tests."""
        # Create test database operations
        self.db_ops_collection = mongodb_database["code_dboperations"]

        db_ops = [
            {
                "_id": f"test_{generate_test_id()}",
                "operation_type": "SELECT",
                "method_name": "GetBale",
                "class_name": "BaleService",
                "project": "Gin",
                "tables_accessed": ["Bales"],
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "operation_type": "INSERT",
                "method_name": "SaveBale",
                "class_name": "BaleService",
                "project": "Gin",
                "tables_accessed": ["Bales", "BaleHistory"],
                "is_transaction": True,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "operation_type": "UPDATE",
                "method_name": "UpdateBaleStatus",
                "class_name": "BaleService",
                "project": "Gin",
                "tables_accessed": ["Bales"],
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
        ]

        insert_test_documents(self.db_ops_collection, db_ops, pipeline_config.test_run_id)

        # Create test call relationships
        self.relationships_collection = mongodb_database["code_relationships"]

        relationships = [
            {
                "_id": f"test_{generate_test_id()}",
                "source_method": "ProcessBale",
                "target_method": "ValidateBale",
                "project": "Gin",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "source_method": "ProcessBale",
                "target_method": "SaveBale",
                "project": "Gin",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
        ]

        insert_test_documents(self.relationships_collection, relationships, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_retrieve_operations_by_type(self, mongodb_database, pipeline_config):
        """Test retrieving operations by type."""
        collection = mongodb_database["code_dboperations"]

        results = list(collection.find({
            "operation_type": "INSERT",
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 1
        assert all(r["operation_type"] == "INSERT" for r in results)

    @pytest.mark.requires_mongodb
    def test_retrieve_operations_by_table(self, mongodb_database, pipeline_config):
        """Test retrieving operations accessing a specific table."""
        collection = mongodb_database["code_dboperations"]

        results = list(collection.find({
            "tables_accessed": "Bales",
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 3
        assert all("Bales" in r["tables_accessed"] for r in results)

    @pytest.mark.requires_mongodb
    def test_retrieve_transactions(self, mongodb_database, pipeline_config):
        """Test retrieving operations that use transactions."""
        collection = mongodb_database["code_dboperations"]

        results = list(collection.find({
            "is_transaction": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 1

    @pytest.mark.requires_mongodb
    def test_retrieve_method_callers(self, mongodb_database, pipeline_config):
        """Test finding all methods that call a specific method."""
        collection = mongodb_database["code_relationships"]

        results = list(collection.find({
            "target_method": "SaveBale",
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 1
        callers = [r["source_method"] for r in results]
        assert "ProcessBale" in callers

    @pytest.mark.requires_mongodb
    def test_aggregate_table_access_frequency(self, mongodb_database, pipeline_config):
        """Test aggregating table access frequency."""
        collection = mongodb_database["code_dboperations"]

        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$unwind": "$tables_accessed"},
            {"$group": {
                "_id": "$tables_accessed",
                "access_count": {"$sum": 1},
                "operations": {"$addToSet": "$operation_type"}
            }},
            {"$sort": {"access_count": -1}}
        ]

        results = list(collection.aggregate(pipeline))

        assert len(results) > 0
        bales_table = next((r for r in results if r["_id"] == "Bales"), None)
        assert bales_table is not None
        assert bales_table["access_count"] >= 3
