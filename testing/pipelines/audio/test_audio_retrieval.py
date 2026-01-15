"""
Audio Retrieval Tests
=====================

Tests for retrieving and querying audio analysis results from MongoDB.
Tests include searching by filename, date, emotion, call metadata, and customer info.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict

from config.test_config import get_test_config
from utils import generate_test_id


class TestAudioRetrieval:
    """Test audio analysis retrieval operations."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, mongodb_database, pipeline_config):
        """Set up test audio analyses for retrieval tests."""
        self.collection = mongodb_database["audio_analyses"]
        self.test_doc_ids = []

        # Create varied test documents
        test_documents = [
            {
                "_id": f"test_{generate_test_id('audio_1')}",
                "original_filename": "20251104-133555_302_(252)792-8686_Outgoing_Auto_2243071124051.mp3",
                "transcription": "Customer calling about billing issue.",
                "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
                "duration_seconds": 45.0,
                "call_metadata": {
                    "parsed": True,
                    "call_date": "2025-11-04",
                    "extension": "302",
                    "phone_number": "(252)792-8686",
                    "direction": "Outgoing"
                },
                "call_content": {
                    "subject": "Billing inquiry",
                    "outcome": "Resolved",
                    "customer_name": "Alice Johnson"
                },
                "customer_lookup": {
                    "found": True,
                    "company_name": "ABC Corp"
                },
                "created_at": datetime.utcnow() - timedelta(hours=2)
            },
            {
                "_id": f"test_{generate_test_id('audio_2')}",
                "original_filename": "20251221-100000_(469)906-0558_Incoming_Auto_2254843027051.mp3",
                "transcription": "Angry customer complaining about service delay.",
                "emotions": {"primary": "ANGRY", "detected": ["ANGRY", "FRUSTRATED"], "timestamps": []},
                "duration_seconds": 120.5,
                "call_metadata": {
                    "parsed": True,
                    "call_date": "2025-12-21",
                    "extension": None,
                    "phone_number": "(469)906-0558",
                    "direction": "Incoming"
                },
                "call_content": {
                    "subject": "Service delay complaint",
                    "outcome": "Pending Follow-up",
                    "customer_name": "Bob Smith"
                },
                "customer_lookup": {
                    "found": True,
                    "company_name": "XYZ Industries"
                },
                "created_at": datetime.utcnow() - timedelta(hours=1)
            },
            {
                "_id": f"test_{generate_test_id('audio_3')}",
                "original_filename": "test_happy_call.mp3",
                "transcription": "Customer very satisfied with resolution.",
                "emotions": {"primary": "HAPPY", "detected": ["HAPPY"], "timestamps": []},
                "duration_seconds": 30.0,
                "call_metadata": {"parsed": False},
                "call_content": {
                    "subject": "Follow-up satisfaction check",
                    "outcome": "Resolved",
                    "customer_name": None
                },
                "customer_lookup": {"found": False},
                "created_at": datetime.utcnow()
            }
        ]

        # Add common fields and insert
        for doc in test_documents:
            doc.update({
                "file_path": f"/tmp/{doc['original_filename']}",
                "formatted_transcription": doc["transcription"],
                "transcription_summary": None,
                "raw_transcription": f"<|en|><|{doc['emotions']['primary']}|><|Speech|>{doc['transcription']}",
                "audio_events": {"detected": ["Speech"], "timestamps": []},
                "language": "en",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })
            self.test_doc_ids.append(doc["_id"])

        self.collection.insert_many(test_documents)

        yield

        # Cleanup
        self.collection.delete_many({"_id": {"$in": self.test_doc_ids}})

    @pytest.mark.requires_mongodb
    def test_retrieve_by_filename(self, mongodb_database, pipeline_config):
        """Test retrieving audio analysis by original filename."""
        result = self.collection.find_one({
            "original_filename": "test_happy_call.mp3",
            "is_test": True
        })

        assert result is not None
        assert result["transcription"] == "Customer very satisfied with resolution."
        assert result["emotions"]["primary"] == "HAPPY"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_phone_number(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by phone number in call metadata."""
        results = list(self.collection.find({
            "call_metadata.phone_number": "(252)792-8686",
            "is_test": True
        }))

        assert len(results) >= 1
        assert results[0]["call_metadata"]["phone_number"] == "(252)792-8686"
        assert results[0]["call_content"]["customer_name"] == "Alice Johnson"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_extension(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by extension (outgoing calls)."""
        results = list(self.collection.find({
            "call_metadata.extension": "302",
            "is_test": True
        }))

        assert len(results) >= 1
        assert results[0]["call_metadata"]["extension"] == "302"
        assert results[0]["call_metadata"]["direction"] == "Outgoing"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_emotion(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by detected emotion."""
        # Find angry calls
        angry_calls = list(self.collection.find({
            "emotions.primary": "ANGRY",
            "is_test": True
        }))

        assert len(angry_calls) >= 1
        assert "ANGRY" in angry_calls[0]["emotions"]["detected"]

        # Find happy calls
        happy_calls = list(self.collection.find({
            "emotions.primary": "HAPPY",
            "is_test": True
        }))

        assert len(happy_calls) >= 1
        assert happy_calls[0]["emotions"]["primary"] == "HAPPY"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_call_outcome(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by call outcome."""
        # Find resolved calls
        resolved = list(self.collection.find({
            "call_content.outcome": "Resolved",
            "is_test": True
        }))

        assert len(resolved) >= 2

        # Find pending follow-ups
        pending = list(self.collection.find({
            "call_content.outcome": "Pending Follow-up",
            "is_test": True
        }))

        assert len(pending) >= 1
        assert pending[0]["call_content"]["outcome"] == "Pending Follow-up"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_customer_name(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by customer name."""
        result = self.collection.find_one({
            "call_content.customer_name": "Bob Smith",
            "is_test": True
        })

        assert result is not None
        assert result["call_content"]["customer_name"] == "Bob Smith"
        assert result["customer_lookup"]["company_name"] == "XYZ Industries"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_company(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by company name."""
        results = list(self.collection.find({
            "customer_lookup.company_name": "ABC Corp",
            "is_test": True
        }))

        assert len(results) >= 1
        assert results[0]["customer_lookup"]["company_name"] == "ABC Corp"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_date_range(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses within a date range."""
        now = datetime.utcnow()
        three_hours_ago = now - timedelta(hours=3)

        results = list(self.collection.find({
            "created_at": {"$gte": three_hours_ago},
            "is_test": True
        }).sort("created_at", -1))

        assert len(results) >= 3
        # Verify sorted by date descending
        for i in range(len(results) - 1):
            assert results[i]["created_at"] >= results[i + 1]["created_at"]

    @pytest.mark.requires_mongodb
    def test_retrieve_by_duration(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses by duration range."""
        # Find calls longer than 1 minute
        long_calls = list(self.collection.find({
            "duration_seconds": {"$gt": 60},
            "is_test": True
        }))

        assert len(long_calls) >= 1
        assert long_calls[0]["duration_seconds"] > 60

        # Find short calls (under 1 minute)
        short_calls = list(self.collection.find({
            "duration_seconds": {"$lt": 60},
            "is_test": True
        }))

        assert len(short_calls) >= 2

    @pytest.mark.requires_mongodb
    def test_retrieve_incoming_vs_outgoing(self, mongodb_database, pipeline_config):
        """Test retrieving incoming vs outgoing calls."""
        # Find incoming calls
        incoming = list(self.collection.find({
            "call_metadata.direction": "Incoming",
            "is_test": True
        }))

        assert len(incoming) >= 1
        assert incoming[0]["call_metadata"]["direction"] == "Incoming"

        # Find outgoing calls
        outgoing = list(self.collection.find({
            "call_metadata.direction": "Outgoing",
            "is_test": True
        }))

        assert len(outgoing) >= 1
        assert outgoing[0]["call_metadata"]["direction"] == "Outgoing"

    @pytest.mark.requires_mongodb
    def test_retrieve_with_customer_lookup(self, mongodb_database, pipeline_config):
        """Test retrieving only calls with successful customer lookups."""
        results = list(self.collection.find({
            "customer_lookup.found": True,
            "is_test": True
        }))

        assert len(results) >= 2
        for doc in results:
            assert doc["customer_lookup"]["found"] is True
            assert doc["customer_lookup"]["company_name"] is not None

    @pytest.mark.requires_mongodb
    def test_full_text_search_transcription(self, mongodb_database, pipeline_config):
        """Test searching transcriptions by keyword."""
        # Search for "billing"
        results = list(self.collection.find({
            "transcription": {"$regex": "billing", "$options": "i"},
            "is_test": True
        }))

        assert len(results) >= 1
        assert "billing" in results[0]["transcription"].lower()

        # Search for "delay"
        delay_results = list(self.collection.find({
            "transcription": {"$regex": "delay", "$options": "i"},
            "is_test": True
        }))

        assert len(delay_results) >= 1
        assert "delay" in delay_results[0]["transcription"].lower()

    @pytest.mark.requires_mongodb
    def test_aggregate_by_emotion(self, mongodb_database, pipeline_config):
        """Test aggregating audio analyses by primary emotion."""
        pipeline = [
            {"$match": {"is_test": True, "test_run_id": pipeline_config.test_run_id}},
            {
                "$group": {
                    "_id": "$emotions.primary",
                    "count": {"$sum": 1},
                    "avg_duration": {"$avg": "$duration_seconds"}
                }
            },
            {"$sort": {"count": -1}}
        ]

        results = list(self.collection.aggregate(pipeline))

        assert len(results) >= 1
        # Verify aggregation structure
        for result in results:
            assert "_id" in result  # emotion name
            assert "count" in result
            assert "avg_duration" in result

    @pytest.mark.requires_mongodb
    def test_aggregate_by_outcome(self, mongodb_database, pipeline_config):
        """Test aggregating audio analyses by call outcome."""
        pipeline = [
            {"$match": {"is_test": True, "test_run_id": pipeline_config.test_run_id}},
            {
                "$group": {
                    "_id": "$call_content.outcome",
                    "count": {"$sum": 1}
                }
            }
        ]

        results = list(self.collection.aggregate(pipeline))

        assert len(results) >= 1
        outcomes = {r["_id"]: r["count"] for r in results}
        assert "Resolved" in outcomes
        assert outcomes["Resolved"] >= 2

    @pytest.mark.requires_mongodb
    def test_retrieve_sorted_by_duration(self, mongodb_database, pipeline_config):
        """Test retrieving audio analyses sorted by duration."""
        results = list(self.collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("duration_seconds", -1))

        assert len(results) >= 3
        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i]["duration_seconds"] >= results[i + 1]["duration_seconds"]

    @pytest.mark.requires_mongodb
    def test_count_by_call_direction(self, mongodb_database, pipeline_config):
        """Test counting calls by direction."""
        incoming_count = self.collection.count_documents({
            "call_metadata.direction": "Incoming",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        outgoing_count = self.collection.count_documents({
            "call_metadata.direction": "Outgoing",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        assert incoming_count >= 1
        assert outgoing_count >= 1
        assert incoming_count + outgoing_count >= 2

    @pytest.mark.requires_mongodb
    def test_retrieve_recent_analyses(self, mongodb_database, pipeline_config):
        """Test retrieving most recent audio analyses."""
        results = list(self.collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("created_at", -1).limit(5))

        assert len(results) >= 3
        # Verify most recent first
        assert results[0]["created_at"] >= results[-1]["created_at"]

    @pytest.mark.requires_mongodb
    def test_retrieve_with_projection(self, mongodb_database, pipeline_config):
        """Test retrieving specific fields only (projection)."""
        results = list(self.collection.find(
            {"is_test": True, "test_run_id": pipeline_config.test_run_id},
            {
                "original_filename": 1,
                "transcription": 1,
                "emotions.primary": 1,
                "duration_seconds": 1,
                "_id": 0
            }
        ))

        assert len(results) >= 3
        for doc in results:
            # Should have only projected fields
            assert "original_filename" in doc
            assert "transcription" in doc
            assert "emotions" in doc
            assert "duration_seconds" in doc
            # Should not have _id or other fields
            assert "_id" not in doc
            assert "file_path" not in doc
