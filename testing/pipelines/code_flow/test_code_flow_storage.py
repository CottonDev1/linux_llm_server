"""
Code Flow Storage Tests
========================

Test storing code flow analysis data in MongoDB.

Collections tested:
- code_dboperations: Database operation patterns
- code_relationships: Method call relationships
- code_ui_events: UI event handler mappings
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List

from config.test_config import get_test_config
from fixtures.mongodb_fixtures import insert_test_documents
from utils import (
    assert_document_stored,
    assert_mongodb_document,
    generate_test_id,
)


class TestCodeFlowStorage:
    """Test code flow pipeline storage operations."""

    @pytest.mark.requires_mongodb
    def test_store_database_operation(self, mongodb_database, pipeline_config):
        """Test storing a database operation pattern."""
        collection = mongodb_database["code_dboperations"]

        # Create DB operation document
        db_op = {
            "_id": f"test_{generate_test_id()}",
            "operation_id": f"op_{generate_test_id()}",
            "method_name": "SaveBale",
            "class_name": "BaleService",
            "project": "Gin",
            "operation_type": "INSERT",
            "tables_accessed": ["Bales", "BaleHistory"],
            "sql_pattern": "INSERT INTO Bales (BaleNumber, Weight) VALUES (@BaleNumber, @Weight)",
            "parameters": ["BaleNumber", "Weight"],
            "file_path": "/Services/BaleService.cs",
            "start_line": 125,
            "is_transaction": True,
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_flow",
        }

        # Insert document
        collection.insert_one(db_op)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            db_op["_id"],
            expected_fields=["operation_type", "tables_accessed", "method_name"]
        )

        # Validate schema
        assert_mongodb_document(
            stored_doc,
            {
                "operation_type": str,
                "tables_accessed": list,
                "method_name": str,
                "class_name": str,
                "is_transaction": bool,
            }
        )

        # Verify content
        assert stored_doc["operation_type"] == "INSERT"
        assert "Bales" in stored_doc["tables_accessed"]
        assert stored_doc["is_transaction"] is True

    @pytest.mark.requires_mongodb
    def test_store_method_relationship(self, mongodb_database, pipeline_config):
        """Test storing method call relationships."""
        collection = mongodb_database["code_relationships"]

        # Create relationship document
        relationship = {
            "_id": f"test_{generate_test_id()}",
            "relationship_type": "calls",
            "source_method": "ProcessTicket",
            "source_class": "TicketService",
            "target_method": "ValidateTicket",
            "target_class": "ValidationService",
            "project": "Gin",
            "call_count": 1,
            "is_async": False,
            "parameters_passed": ["ticketId", "userId"],
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_flow",
        }

        # Insert document
        collection.insert_one(relationship)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            relationship["_id"],
            expected_fields=["source_method", "target_method", "relationship_type"]
        )

        # Verify content
        assert stored_doc["relationship_type"] == "calls"
        assert stored_doc["source_method"] == "ProcessTicket"
        assert stored_doc["target_method"] == "ValidateTicket"

    @pytest.mark.requires_mongodb
    def test_store_ui_event_mapping(self, mongodb_database, pipeline_config):
        """Test storing UI event to method mappings."""
        collection = mongodb_database["code_ui_events"]

        # Create UI event document
        ui_event = {
            "_id": f"test_{generate_test_id()}",
            "event_type": "button_click",
            "control_name": "btnSaveBale",
            "event_handler": "btnSaveBale_Click",
            "class_name": "BaleEntryForm",
            "project": "Gin",
            "entry_method": "btnSaveBale_Click",
            "methods_called": ["ValidateBale", "SaveBale", "RefreshGrid"],
            "ui_framework": "WPF",
            "is_async": True,
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_flow",
        }

        # Insert document
        collection.insert_one(ui_event)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            ui_event["_id"],
            expected_fields=["event_type", "control_name", "event_handler", "methods_called"]
        )

        # Verify content
        assert stored_doc["event_type"] == "button_click"
        assert stored_doc["control_name"] == "btnSaveBale"
        assert "SaveBale" in stored_doc["methods_called"]

    @pytest.mark.requires_mongodb
    def test_store_call_chain(self, mongodb_database, pipeline_config):
        """Test storing a complete call chain."""
        collection = mongodb_database["code_call_chains"]

        # Create call chain document
        call_chain = {
            "_id": f"test_{generate_test_id()}",
            "chain_id": f"chain_{generate_test_id()}",
            "entry_point": "btnProcessOrder_Click",
            "project": "Warehouse",
            "chain_depth": 4,
            "methods_in_chain": [
                "btnProcessOrder_Click",
                "ValidateOrder",
                "CalculateTotal",
                "SaveOrder"
            ],
            "touches_database": True,
            "database_tables": ["Orders", "OrderItems", "Inventory"],
            "total_methods": 4,
            "is_async": True,
            "has_error_handling": True,
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_flow",
        }

        # Insert document
        collection.insert_one(call_chain)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            call_chain["_id"],
            expected_fields=["entry_point", "methods_in_chain", "touches_database"]
        )

        # Verify content
        assert stored_doc["chain_depth"] == 4
        assert len(stored_doc["methods_in_chain"]) == 4
        assert stored_doc["touches_database"] is True

    @pytest.mark.requires_mongodb
    def test_bulk_insert_operations(self, mongodb_database, pipeline_config):
        """Test bulk inserting database operations."""
        collection = mongodb_database["code_dboperations"]

        # Create multiple operations
        operations = []
        for i in range(10):
            op = {
                "_id": f"test_{generate_test_id()}",
                "operation_type": ["SELECT", "INSERT", "UPDATE", "DELETE"][i % 4],
                "method_name": f"Operation{i}",
                "class_name": "DataService",
                "project": "TestProject",
                "tables_accessed": [f"Table{i}"],
                "is_test": True,
                "test_marker": True,
                "test_run_id": pipeline_config.test_run_id,
                "created_at": datetime.utcnow(),
                "pipeline": "code_flow",
            }
            operations.append(op)

        # Bulk insert
        ids = insert_test_documents(
            collection,
            operations,
            test_run_id=pipeline_config.test_run_id
        )

        # Verify all inserted
        assert len(ids) == 10

        # Verify counts by operation type
        insert_count = collection.count_documents({
            "operation_type": "INSERT",
            "test_run_id": pipeline_config.test_run_id
        })
        assert insert_count >= 2

    @pytest.mark.requires_mongodb
    def test_query_operations_by_table(self, mongodb_database, pipeline_config):
        """Test querying operations by table accessed."""
        collection = mongodb_database["code_dboperations"]

        # Insert operations accessing different tables
        ops = [
            {
                "_id": f"test_{generate_test_id()}",
                "operation_type": "SELECT",
                "method_name": "GetUser",
                "tables_accessed": ["Users"],
                "project": "TestProject",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "operation_type": "UPDATE",
                "method_name": "UpdateUser",
                "tables_accessed": ["Users", "UserAudit"],
                "project": "TestProject",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "operation_type": "INSERT",
                "method_name": "CreateOrder",
                "tables_accessed": ["Orders"],
                "project": "TestProject",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
        ]

        insert_test_documents(collection, ops, pipeline_config.test_run_id)

        # Query operations on Users table
        users_ops = list(collection.find({
            "tables_accessed": "Users",
            "test_run_id": pipeline_config.test_run_id
        }))

        # Verify results
        assert len(users_ops) == 2
        assert all("Users" in op["tables_accessed"] for op in users_ops)

    @pytest.mark.requires_mongodb
    def test_aggregate_operations_by_type(self, mongodb_database, pipeline_config):
        """Test aggregating operations by type."""
        collection = mongodb_database["code_dboperations"]

        # Insert varied operations
        ops = []
        operation_types = ["SELECT", "INSERT", "UPDATE", "DELETE", "SELECT", "SELECT", "INSERT"]

        for i, op_type in enumerate(operation_types):
            op = {
                "_id": f"test_{generate_test_id()}",
                "operation_type": op_type,
                "method_name": f"Method{i}",
                "project": "TestProject",
                "tables_accessed": ["TestTable"],
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            }
            ops.append(op)

        insert_test_documents(collection, ops, pipeline_config.test_run_id)

        # Aggregate by operation type
        pipeline_agg = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": "$operation_type",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]

        results = list(collection.aggregate(pipeline_agg))

        # Verify aggregation
        assert len(results) == 4
        select_result = next((r for r in results if r["_id"] == "SELECT"), None)
        assert select_result is not None
        assert select_result["count"] == 3

    @pytest.mark.requires_mongodb
    def test_index_for_performance(self, mongodb_database, pipeline_config):
        """Test creating indexes for code flow queries."""
        collection = mongodb_database["code_dboperations"]

        # Create indexes
        collection.create_index([("operation_type", 1)])
        collection.create_index([("tables_accessed", 1)])
        collection.create_index([("project", 1), ("class_name", 1)])

        # Verify indexes exist
        indexes = list(collection.list_indexes())
        index_names = [idx["name"] for idx in indexes]

        assert any("operation_type" in name for name in index_names)
        assert any("tables_accessed" in name for name in index_names)
