"""
Git Analysis End-to-End Tests
==============================

Test complete git analysis pipeline: parse → analyze → store → query

These tests simulate the full workflow from code parsing through
analysis to storage and retrieval.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from config.test_config import get_test_config
from fixtures.mongodb_fixtures import create_mock_code_method
from fixtures.llm_fixtures import LocalLLMClient
from utils import (
    assert_document_stored,
    assert_llm_response_valid,
    generate_test_id,
    measure_time,
)


class TestGitE2E:
    """End-to-end tests for git analysis pipeline."""

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    async def test_complete_method_analysis_workflow(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config
    ):
        """Test complete workflow: code → analysis → storage → retrieval."""
        collection = mongodb_database["code_methods"]

        # Step 1: Input - raw code method
        method_code = """
        public async Task<User> AuthenticateUserAsync(string username, string password)
        {
            // Hash password
            var hashedPassword = HashPassword(password);

            // Query database
            var sql = "SELECT * FROM Users WHERE Username = @Username AND PasswordHash = @PasswordHash";
            using (var conn = new SqlConnection(_connectionString))
            {
                var user = await conn.QuerySingleOrDefaultAsync<User>(sql, new { Username = username, PasswordHash = hashedPassword });

                if (user != null)
                {
                    // Update last login
                    await UpdateLastLoginAsync(user.UserID);
                }

                return user;
            }
        }
        """

        # Step 2: Generate analysis using LLM
        with measure_time("LLM Analysis") as timing:
            analysis_prompt = f"""Analyze this C# method and extract:
1. Purpose (1-2 sentences)
2. Method calls (comma-separated)
3. Database tables accessed (comma-separated)
4. Is this an async method? (yes/no)

Code:
{method_code}

Analysis:"""

            response = llm_client.generate(
                prompt=analysis_prompt,
                model_type="code",
                max_tokens=300,
                temperature=0.0,
            )

        # Verify LLM response
        assert_llm_response_valid(response, min_length=20)
        assert timing["elapsed_ms"] < 30000  # Should complete within 30s

        # Step 3: Create structured document from analysis
        method_doc = create_mock_code_method(
            method_name="AuthenticateUserAsync",
            class_name="UserService",
            project="TestProject",
            code=method_code,
        )

        # Enhance with analysis results
        method_doc.update({
            "namespace": "TestProject.Services",
            "file_path": "/src/Services/UserService.cs",
            "start_line": 45,
            "end_line": 62,
            "return_type": "Task<User>",
            "is_public": True,
            "is_static": False,
            "is_async": True,
            "purpose_summary": response.text[:200],  # Extract purpose from LLM
            "calls_methods": ["HashPassword", "QuerySingleOrDefaultAsync", "UpdateLastLoginAsync"],
            "database_tables": ["Users"],
            "llm_analysis": response.text,
            "test_run_id": pipeline_config.test_run_id,
        })

        # Step 4: Store in MongoDB
        with measure_time("MongoDB Insert") as timing:
            collection.insert_one(method_doc)

        assert timing["elapsed_ms"] < 1000  # Should be fast

        # Step 5: Query back and verify
        stored_doc = assert_document_stored(
            collection,
            method_doc["_id"],
            expected_fields=["method_name", "is_async", "database_tables", "purpose_summary"]
        )

        # Verify stored data
        assert stored_doc["method_name"] == "AuthenticateUserAsync"
        assert stored_doc["is_async"] is True
        assert "Users" in stored_doc["database_tables"]
        assert len(stored_doc["purpose_summary"]) > 0

        # Step 6: Query by different criteria
        # Query by table access
        by_table = collection.find_one({
            "database_tables": "Users",
            "test_run_id": pipeline_config.test_run_id
        })
        assert by_table is not None

        # Query by async flag
        by_async = collection.find_one({
            "is_async": True,
            "test_run_id": pipeline_config.test_run_id
        })
        assert by_async is not None

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    async def test_bulk_method_analysis(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config
    ):
        """Test analyzing multiple methods in batch."""
        collection = mongodb_database["code_methods"]

        # Define multiple methods
        methods_to_analyze = [
            {
                "name": "GetAllUsers",
                "code": "public List<User> GetAllUsers() { return _db.Users.ToList(); }",
                "expected_tables": ["Users"],
            },
            {
                "name": "CreateOrder",
                "code": "public void CreateOrder(Order order) { _db.Orders.Add(order); _db.SaveChanges(); }",
                "expected_tables": ["Orders"],
            },
            {
                "name": "CalculateDiscount",
                "code": "public decimal CalculateDiscount(decimal total) { return total * 0.1m; }",
                "expected_tables": [],
            },
        ]

        analyzed_methods = []

        # Process each method
        for method_info in methods_to_analyze:
            # Generate purpose summary
            prompt = f"Summarize this C# method in one sentence:\n{method_info['code']}\nSummary:"

            response = llm_client.generate(
                prompt=prompt,
                model_type="code",
                max_tokens=100,
                temperature=0.0,
            )

            # Create document
            doc = create_mock_code_method(
                method_name=method_info["name"],
                class_name="TestService",
                project="TestProject",
                code=method_info["code"],
            )
            doc.update({
                "purpose_summary": response.text if response.success else "No summary available",
                "database_tables": method_info["expected_tables"],
                "test_run_id": pipeline_config.test_run_id,
            })

            analyzed_methods.append(doc)

        # Bulk insert
        collection.insert_many(analyzed_methods)

        # Verify all stored
        count = collection.count_documents({
            "test_run_id": pipeline_config.test_run_id,
            "class_name": "TestService"
        })
        assert count == 3

        # Query methods by table access
        methods_with_db = list(collection.find({
            "database_tables": {"$ne": []},
            "test_run_id": pipeline_config.test_run_id
        }))
        assert len(methods_with_db) == 2

    @pytest.mark.requires_mongodb
    @pytest.mark.e2e
    def test_cross_reference_method_calls(
        self,
        mongodb_database,
        pipeline_config
    ):
        """Test building call graph across multiple methods."""
        collection = mongodb_database["code_methods"]

        # Create method call chain: A -> B -> C
        methods = [
            create_mock_code_method(
                method_name="MethodA",
                class_name="ServiceA",
                project="TestProject",
                calls_methods=["MethodB"],
            ),
            create_mock_code_method(
                method_name="MethodB",
                class_name="ServiceB",
                project="TestProject",
                calls_methods=["MethodC"],
            ),
            create_mock_code_method(
                method_name="MethodC",
                class_name="ServiceC",
                project="TestProject",
                calls_methods=[],
            ),
        ]

        # Add test metadata
        for method in methods:
            method["test_run_id"] = pipeline_config.test_run_id

        # Insert methods
        collection.insert_many(methods)

        # Build call graph: find all methods called by MethodA (direct and indirect)
        called_methods = set()

        # Direct calls from MethodA
        method_a = collection.find_one({"method_name": "MethodA", "test_run_id": pipeline_config.test_run_id})
        called_methods.update(method_a.get("calls_methods", []))

        # Indirect calls (MethodB's calls)
        for called in list(called_methods):
            method = collection.find_one({
                "method_name": called,
                "test_run_id": pipeline_config.test_run_id
            })
            if method:
                called_methods.update(method.get("calls_methods", []))

        # Verify call graph
        assert "MethodB" in called_methods
        assert "MethodC" in called_methods

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_full_repository_analysis_simulation(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config
    ):
        """Simulate analyzing an entire small repository."""
        collection = mongodb_database["code_methods"]

        # Simulate repository with multiple classes and methods
        repository_structure = {
            "UserService": ["GetUser", "CreateUser", "UpdateUser", "DeleteUser"],
            "OrderService": ["CreateOrder", "GetOrder", "CancelOrder"],
            "PaymentService": ["ProcessPayment", "RefundPayment"],
        }

        total_methods = sum(len(methods) for methods in repository_structure.values())
        stored_count = 0

        with measure_time("Full Repository Analysis") as timing:
            for class_name, method_names in repository_structure.items():
                for method_name in method_names:
                    # Simulate code
                    code = f"public void {method_name}() {{ /* implementation */ }}"

                    # Create document (skip LLM for speed in simulation)
                    doc = create_mock_code_method(
                        method_name=method_name,
                        class_name=class_name,
                        project="TestRepository",
                        code=code,
                    )
                    doc.update({
                        "purpose_summary": f"{method_name} in {class_name}",
                        "test_run_id": pipeline_config.test_run_id,
                    })

                    # Store
                    collection.insert_one(doc)
                    stored_count += 1

        # Verify all methods stored
        assert stored_count == total_methods

        # Query repository statistics
        stats_pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": "$class_name",
                "method_count": {"$sum": 1}
            }},
            {"$sort": {"method_count": -1}}
        ]

        stats = list(collection.aggregate(stats_pipeline))

        # Verify statistics
        assert len(stats) == 3
        user_service_stats = next(s for s in stats if s["_id"] == "UserService")
        assert user_service_stats["method_count"] == 4

    @pytest.mark.requires_mongodb
    @pytest.mark.e2e
    def test_update_analysis_workflow(
        self,
        mongodb_database,
        pipeline_config
    ):
        """Test updating method analysis when code changes."""
        collection = mongodb_database["code_methods"]

        # Initial version
        method_doc = create_mock_code_method(
            method_name="ProcessData",
            class_name="DataService",
            project="TestProject",
        )
        method_doc.update({
            "version": 1,
            "calls_methods": ["MethodA"],
            "test_run_id": pipeline_config.test_run_id,
        })

        collection.insert_one(method_doc)

        # Simulate code change - method now calls additional methods
        collection.update_one(
            {"_id": method_doc["_id"]},
            {
                "$set": {
                    "version": 2,
                    "calls_methods": ["MethodA", "MethodB", "MethodC"],
                    "updated_at": datetime.utcnow(),
                }
            }
        )

        # Verify update
        updated_doc = collection.find_one({"_id": method_doc["_id"]})
        assert updated_doc["version"] == 2
        assert len(updated_doc["calls_methods"]) == 3
        assert "MethodB" in updated_doc["calls_methods"]
