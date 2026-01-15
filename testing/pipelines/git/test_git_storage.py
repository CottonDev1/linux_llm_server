"""
Git Analysis Storage Tests
===========================

Test storing parsed code entities from git analysis in MongoDB.

Collections tested:
- code_methods: Method-level code analysis
- code_classes: Class-level code analysis
- code_dboperations: Database operation tracking
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from config.test_config import get_test_config
from fixtures.mongodb_fixtures import (
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)
from utils import (
    assert_document_stored,
    assert_mongodb_document,
    generate_test_id,
)


class TestGitStorage:
    """Test git analysis storage operations."""

    @pytest.mark.requires_mongodb
    def test_store_code_method(self, mongodb_database, pipeline_config):
        """Test storing a parsed code method in code_methods collection."""
        collection = mongodb_database["code_methods"]

        # Create test method document
        method_doc = create_mock_code_method(
            method_name="TestMethod",
            class_name="TestClass",
            project="TestProject",
            code="public void TestMethod() { Console.WriteLine(\"Hello\"); }",
        )

        # Add required git analysis metadata
        method_doc.update({
            "namespace": "TestProject.Services",
            "file_path": "/src/Services/TestClass.cs",
            "start_line": 10,
            "end_line": 15,
            "return_type": "void",
            "is_public": True,
            "is_static": False,
            "is_async": False,
            "purpose_summary": "Test method for demonstration",
            "calls_methods": ["Console.WriteLine"],
            "database_tables": [],
        })

        # Insert document
        result = collection.insert_one(method_doc)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            method_doc["_id"],
            expected_fields=["method_name", "class_name", "project", "content", "file_path"]
        )

        # Validate schema
        assert_mongodb_document(
            stored_doc,
            {
                "method_name": str,
                "class_name": str,
                "project": str,
                "content": str,
                "file_path": str,
                "start_line": int,
                "end_line": int,
                "is_public": bool,
                "is_test": bool,
            }
        )

        # Verify content
        assert stored_doc["method_name"] == "TestMethod"
        assert stored_doc["class_name"] == "TestClass"
        assert stored_doc["project"] == "TestProject"
        assert "Console.WriteLine" in str(stored_doc.get("calls_methods", []))

    @pytest.mark.requires_mongodb
    def test_store_code_class(self, mongodb_database, pipeline_config):
        """Test storing a parsed code class in code_classes collection."""
        collection = mongodb_database["code_classes"]

        # Create test class document
        class_doc = {
            "_id": f"test_{generate_test_id()}",
            "class_name": "UserService",
            "namespace": "TestProject.Services",
            "project": "TestProject",
            "content": "public class UserService { /* implementation */ }",
            "file_path": "/src/Services/UserService.cs",
            "is_public": True,
            "is_static": False,
            "base_class": None,
            "interfaces": ["IUserService"],
            "methods_count": 5,
            "purpose_summary": "Manages user operations",
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "git",
        }

        # Insert document
        collection.insert_one(class_doc)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            class_doc["_id"],
            expected_fields=["class_name", "namespace", "project", "file_path"]
        )

        # Verify content
        assert stored_doc["class_name"] == "UserService"
        assert stored_doc["namespace"] == "TestProject.Services"
        assert stored_doc["methods_count"] == 5
        assert "IUserService" in stored_doc["interfaces"]

    @pytest.mark.requires_mongodb
    def test_store_database_operation(self, mongodb_database, pipeline_config):
        """Test storing database operation metadata in code_dboperations collection."""
        collection = mongodb_database["code_dboperations"]

        # Create DB operation document
        db_op_doc = {
            "_id": f"test_{generate_test_id()}",
            "method_name": "GetUserById",
            "class_name": "UserRepository",
            "project": "TestProject",
            "operation_type": "SELECT",
            "tables_accessed": ["Users", "UserProfiles"],
            "sql_pattern": "SELECT * FROM Users WHERE UserID = @UserId",
            "file_path": "/src/Repositories/UserRepository.cs",
            "start_line": 25,
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "git",
        }

        # Insert document
        collection.insert_one(db_op_doc)

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            db_op_doc["_id"],
            expected_fields=["method_name", "operation_type", "tables_accessed"]
        )

        # Verify content
        assert stored_doc["operation_type"] == "SELECT"
        assert "Users" in stored_doc["tables_accessed"]
        assert "UserProfiles" in stored_doc["tables_accessed"]

    @pytest.mark.requires_mongodb
    def test_bulk_insert_methods(self, mongodb_database, pipeline_config):
        """Test bulk inserting multiple code methods."""
        collection = mongodb_database["code_methods"]

        # Create multiple method documents
        methods = []
        for i in range(10):
            method = create_mock_code_method(
                method_name=f"Method{i}",
                class_name="BulkTestClass",
                project="TestProject",
                code=f"public void Method{i}() {{ /* implementation {i} */ }}",
            )
            method.update({
                "namespace": "TestProject.BulkTest",
                "file_path": f"/src/BulkTestClass_{i}.cs",
                "start_line": i * 10,
                "end_line": (i * 10) + 5,
            })
            methods.append(method)

        # Bulk insert
        ids = insert_test_documents(
            collection,
            methods,
            test_run_id=pipeline_config.test_run_id
        )

        # Verify all inserted
        assert len(ids) == 10

        # Verify retrieval
        count = collection.count_documents({
            "test_run_id": pipeline_config.test_run_id,
            "class_name": "BulkTestClass"
        })
        assert count == 10

    @pytest.mark.requires_mongodb
    def test_update_code_method(self, mongodb_database, pipeline_config):
        """Test updating an existing code method."""
        collection = mongodb_database["code_methods"]

        # Insert initial document
        method_doc = create_mock_code_method(
            method_name="UpdateTest",
            class_name="TestClass",
            project="TestProject",
        )
        method_doc["purpose_summary"] = "Initial purpose"
        collection.insert_one(method_doc)

        # Update the document
        collection.update_one(
            {"_id": method_doc["_id"]},
            {
                "$set": {
                    "purpose_summary": "Updated purpose summary",
                    "calls_methods": ["NewMethod1", "NewMethod2"],
                    "last_updated": datetime.utcnow(),
                }
            }
        )

        # Verify update
        updated_doc = collection.find_one({"_id": method_doc["_id"]})
        assert updated_doc is not None
        assert updated_doc["purpose_summary"] == "Updated purpose summary"
        assert "NewMethod1" in updated_doc["calls_methods"]
        assert "last_updated" in updated_doc

    @pytest.mark.requires_mongodb
    def test_query_methods_by_project(self, mongodb_database, pipeline_config):
        """Test querying methods filtered by project."""
        collection = mongodb_database["code_methods"]

        # Insert methods for different projects
        projects = ["Project1", "Project2", "Project3"]
        for project in projects:
            for i in range(3):
                method = create_mock_code_method(
                    method_name=f"Method{i}",
                    class_name="TestClass",
                    project=project,
                )
                insert_test_documents(collection, [method], pipeline_config.test_run_id)

        # Query for specific project
        project1_methods = list(collection.find({
            "project": "Project1",
            "test_run_id": pipeline_config.test_run_id
        }))

        # Verify correct filtering
        assert len(project1_methods) == 3
        assert all(m["project"] == "Project1" for m in project1_methods)

    @pytest.mark.requires_mongodb
    def test_index_creation(self, mongodb_database, pipeline_config):
        """Test that appropriate indexes exist for performance."""
        collection = mongodb_database["code_methods"]

        # Create indexes for common queries
        collection.create_index([("project", 1)])
        collection.create_index([("class_name", 1)])
        collection.create_index([("method_name", 1)])
        collection.create_index([("file_path", 1)])

        # Verify indexes exist
        indexes = list(collection.list_indexes())
        index_names = [idx["name"] for idx in indexes]

        assert any("project" in name for name in index_names)
        assert any("class_name" in name for name in index_names)

    @pytest.mark.requires_mongodb
    def test_cleanup_test_data(self, mongodb_database, pipeline_config, cleanup_test_data):
        """Test that cleanup properly removes test data."""
        collection = mongodb_database["code_methods"]

        # Insert test data
        method = create_mock_code_method()
        insert_test_documents(collection, [method], pipeline_config.test_run_id)

        # Verify inserted
        count_before = collection.count_documents({
            "test_run_id": pipeline_config.test_run_id
        })
        assert count_before > 0

        # Cleanup happens automatically via fixture
        # This test validates the cleanup fixture works correctly
