"""
Code Assistance Storage Tests
==============================

Test storing code assistance interactions in MongoDB.

Collections tested:
- code_interactions: User queries and responses
- code_feedback: User feedback on responses
"""

import pytest
from datetime import datetime
from utils import assert_document_stored, assert_mongodb_document, generate_test_id


class TestCodeAssistanceStorage:
    """Test code assistance storage operations."""

    @pytest.mark.requires_mongodb
    def test_store_code_interaction(self, mongodb_database, pipeline_config):
        """Test storing a code assistance interaction."""
        collection = mongodb_database["code_interactions"]

        interaction = {
            "_id": f"test_{generate_test_id()}",
            "response_id": f"resp_{generate_test_id()}",
            "query": "How does the bale saving process work?",
            "answer": "The bale saving process involves validation, database insertion, and audit logging.",
            "sources": ["BaleService.SaveBale", "ValidationService.ValidateBale"],
            "call_chain": ["btnSaveBale_Click", "ValidateBale", "SaveBale"],
            "project": "Gin",
            "model_used": "qwen2.5-coder",
            "retrieval_time_ms": 150,
            "generation_time_ms": 2500,
            "total_time_ms": 2650,
            "feedback_received": False,
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_assistance",
        }

        collection.insert_one(interaction)

        stored_doc = assert_document_stored(
            collection,
            interaction["_id"],
            expected_fields=["query", "answer", "sources", "model_used"]
        )

        assert_mongodb_document(
            stored_doc,
            {
                "query": str,
                "answer": str,
                "sources": list,
                "retrieval_time_ms": int,
                "generation_time_ms": int,
                "feedback_received": bool,
            }
        )

        assert stored_doc["query"] == interaction["query"]
        assert "BaleService.SaveBale" in stored_doc["sources"]

    @pytest.mark.requires_mongodb
    def test_store_code_feedback(self, mongodb_database, pipeline_config):
        """Test storing user feedback on code assistance."""
        collection = mongodb_database["code_feedback"]

        feedback = {
            "_id": f"test_{generate_test_id()}",
            "feedback_id": f"fb_{generate_test_id()}",
            "response_id": f"resp_{generate_test_id()}",
            "is_helpful": True,
            "rating": 5,
            "comment": "Very helpful explanation",
            "error_category": None,
            "expected_methods": [],
            "is_test": True,
            "test_marker": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_assistance",
        }

        collection.insert_one(feedback)

        stored_doc = assert_document_stored(
            collection,
            feedback["_id"],
            expected_fields=["response_id", "is_helpful", "rating"]
        )

        assert stored_doc["is_helpful"] is True
        assert stored_doc["rating"] == 5

    @pytest.mark.requires_mongodb
    def test_update_interaction_with_feedback(self, mongodb_database, pipeline_config):
        """Test updating interaction when feedback is received."""
        interactions_collection = mongodb_database["code_interactions"]
        feedback_collection = mongodb_database["code_feedback"]

        # Create interaction
        interaction_id = f"test_{generate_test_id()}"
        response_id = f"resp_{generate_test_id()}"

        interaction = {
            "_id": interaction_id,
            "response_id": response_id,
            "query": "Test query",
            "answer": "Test answer",
            "feedback_received": False,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        interactions_collection.insert_one(interaction)

        # Add feedback
        feedback = {
            "_id": f"test_{generate_test_id()}",
            "response_id": response_id,
            "is_helpful": True,
            "rating": 4,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        feedback_collection.insert_one(feedback)

        # Update interaction
        interactions_collection.update_one(
            {"response_id": response_id},
            {
                "$set": {
                    "feedback_received": True,
                    "feedback_rating": 4,
                    "feedback_timestamp": datetime.utcnow(),
                }
            }
        )

        # Verify update
        updated = interactions_collection.find_one({"_id": interaction_id})
        assert updated["feedback_received"] is True
        assert updated["feedback_rating"] == 4

    @pytest.mark.requires_mongodb
    def test_query_interactions_by_performance(self, mongodb_database, pipeline_config):
        """Test querying slow interactions for optimization."""
        collection = mongodb_database["code_interactions"]

        interactions = [
            {
                "_id": f"test_{generate_test_id()}",
                "query": f"Query {i}",
                "total_time_ms": 1000 * i,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            }
            for i in range(1, 6)
        ]

        collection.insert_many(interactions)

        # Find slow interactions (>3000ms)
        slow_queries = list(collection.find({
            "total_time_ms": {"$gt": 3000},
            "test_run_id": pipeline_config.test_run_id
        }).sort("total_time_ms", -1))

        assert len(slow_queries) == 2
        assert slow_queries[0]["total_time_ms"] > slow_queries[1]["total_time_ms"]

    @pytest.mark.requires_mongodb
    def test_aggregate_feedback_stats(self, mongodb_database, pipeline_config):
        """Test aggregating feedback statistics."""
        collection = mongodb_database["code_feedback"]

        feedback_items = [
            {"response_id": f"r1", "is_helpful": True, "rating": 5, "is_test": True, "test_run_id": pipeline_config.test_run_id},
            {"response_id": f"r2", "is_helpful": True, "rating": 4, "is_test": True, "test_run_id": pipeline_config.test_run_id},
            {"response_id": f"r3", "is_helpful": False, "rating": 2, "is_test": True, "test_run_id": pipeline_config.test_run_id},
            {"response_id": f"r4", "is_helpful": True, "rating": 5, "is_test": True, "test_run_id": pipeline_config.test_run_id},
        ]

        for fb in feedback_items:
            fb["_id"] = f"test_{generate_test_id()}"

        collection.insert_many(feedback_items)

        # Aggregate stats
        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": None,
                "total_feedback": {"$sum": 1},
                "helpful_count": {
                    "$sum": {"$cond": [{"$eq": ["$is_helpful", True]}, 1, 0]}
                },
                "avg_rating": {"$avg": "$rating"},
            }}
        ]

        stats = list(collection.aggregate(pipeline))

        assert len(stats) == 1
        assert stats[0]["total_feedback"] == 4
        assert stats[0]["helpful_count"] == 3
        assert stats[0]["avg_rating"] == 4.0
