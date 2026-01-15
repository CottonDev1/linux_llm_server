"""
Code Assistance Retrieval Tests
================================

Test retrieving code assistance history and feedback.
"""

import pytest
from datetime import datetime, timedelta
from utils import generate_test_id


class TestCodeAssistanceRetrieval:
    """Test code assistance retrieval operations."""

    @pytest.fixture(autouse=True)
    async def setup_test_data(self, mongodb_database, pipeline_config):
        """Set up test data."""
        self.interactions_collection = mongodb_database["code_interactions"]

        interactions = [
            {
                "_id": f"test_{generate_test_id()}",
                "query": "How to save a bale?",
                "project": "Gin",
                "model_used": "qwen2.5-coder",
                "total_time_ms": 2000,
                "created_at": datetime.utcnow() - timedelta(days=1),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "query": "Order processing flow?",
                "project": "Warehouse",
                "model_used": "qwen2.5-coder",
                "total_time_ms": 3500,
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
        ]

        self.interactions_collection.insert_many(interactions)

    @pytest.mark.requires_mongodb
    def test_retrieve_recent_interactions(self, mongodb_database, pipeline_config):
        """Test retrieving recent interactions."""
        collection = mongodb_database["code_interactions"]

        results = list(collection.find({
            "test_run_id": pipeline_config.test_run_id
        }).sort("created_at", -1).limit(10))

        assert len(results) == 2
        assert results[0]["created_at"] > results[1]["created_at"]

    @pytest.mark.requires_mongodb
    def test_retrieve_by_project(self, mongodb_database, pipeline_config):
        """Test retrieving interactions for a specific project."""
        collection = mongodb_database["code_interactions"]

        gin_results = list(collection.find({
            "project": "Gin",
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(gin_results) == 1
        assert gin_results[0]["query"] == "How to save a bale?"

    @pytest.mark.requires_mongodb
    def test_search_by_query_text(self, mongodb_database, pipeline_config):
        """Test searching interactions by query text."""
        collection = mongodb_database["code_interactions"]

        # Text search requires index
        try:
            collection.create_index([("query", "text")])
        except:
            pass

        results = list(collection.find({
            "$text": {"$search": "bale"},
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 1
        assert "bale" in results[0]["query"].lower()

    @pytest.mark.requires_mongodb
    def test_aggregate_performance_metrics(self, mongodb_database, pipeline_config):
        """Test aggregating performance metrics."""
        collection = mongodb_database["code_interactions"]

        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": "$project",
                "avg_time_ms": {"$avg": "$total_time_ms"},
                "max_time_ms": {"$max": "$total_time_ms"},
                "query_count": {"$sum": 1},
            }}
        ]

        stats = list(collection.aggregate(pipeline))

        assert len(stats) == 2
        gin_stats = next(s for s in stats if s["_id"] == "Gin")
        assert gin_stats["query_count"] == 1
        assert gin_stats["avg_time_ms"] == 2000
