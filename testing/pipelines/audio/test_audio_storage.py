"""
Audio Storage Tests
===================

Tests for storing audio analysis results in the audio_analyses MongoDB collection.

Collection: audio_analyses
Schema:
- _id: unique identifier
- original_filename: original audio file name
- file_path: path to audio file
- transcription: transcribed text
- formatted_transcription: LLM-formatted readable transcription
- transcription_summary: LLM summary (for long recordings)
- raw_transcription: raw transcription with SenseVoice tags
- emotions: detected emotions
- audio_events: detected audio events
- language: detected language
- duration_seconds: audio duration
- call_metadata: parsed RingCentral metadata
- call_content: LLM-analyzed content (subject, outcome, customer_name)
- customer_lookup: database lookup results
- created_at: timestamp
"""

import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

from config.test_config import get_test_config
from utils import generate_test_id, assert_document_stored, assert_mongodb_document


class TestAudioStorage:
    """Test audio analysis storage operations."""

    @pytest.mark.requires_mongodb
    def test_store_basic_audio_analysis(self, mongodb_database, pipeline_config):
        """Test storing a basic audio analysis result."""
        collection = mongodb_database["audio_analyses"]

        # Create test document
        doc_id = f"test_{generate_test_id('audio')}"
        test_doc = {
            "_id": doc_id,
            "original_filename": "test_call_20251222-120000.mp3",
            "file_path": "/tmp/test_audio.mp3",
            "transcription": "This is a test transcription.",
            "formatted_transcription": "This is a test transcription.",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>This is a test transcription.",
            "emotions": {
                "primary": "NEUTRAL",
                "detected": ["NEUTRAL"],
                "timestamps": []
            },
            "audio_events": {
                "detected": ["Speech"],
                "timestamps": []
            },
            "language": "en",
            "duration_seconds": 15.5,
            "call_metadata": {
                "parsed": False,
                "call_date": None,
                "call_time": None,
                "extension": None,
                "phone_number": None,
                "direction": None,
                "auto_flag": None,
                "recording_id": None
            },
            "call_content": {
                "subject": None,
                "outcome": None,
                "customer_name": None,
                "confidence": 0.0,
                "analysis_model": ""
            },
            "customer_lookup": {
                "found": False
            },
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        # Insert document
        result = collection.insert_one(test_doc)
        assert result.inserted_id == doc_id

        # Verify document was stored
        stored_doc = assert_document_stored(
            collection,
            doc_id,
            expected_fields=[
                "transcription",
                "emotions",
                "audio_events",
                "duration_seconds",
                "created_at"
            ]
        )

        assert stored_doc["transcription"] == "This is a test transcription."
        assert stored_doc["emotions"]["primary"] == "NEUTRAL"
        assert stored_doc["duration_seconds"] == 15.5

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_audio_with_call_metadata(self, mongodb_database, pipeline_config):
        """Test storing audio analysis with parsed RingCentral call metadata."""
        collection = mongodb_database["audio_analyses"]

        doc_id = f"test_{generate_test_id('audio_call')}"
        test_doc = {
            "_id": doc_id,
            "original_filename": "20251104-133555_302_(252)792-8686_Outgoing_Auto_2243071124051.mp3",
            "file_path": "/tmp/test_call.mp3",
            "transcription": "Customer support call regarding ticket issue.",
            "formatted_transcription": "Customer support call regarding ticket issue.",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Customer support call regarding ticket issue.",
            "emotions": {
                "primary": "NEUTRAL",
                "detected": ["NEUTRAL"],
                "timestamps": []
            },
            "audio_events": {
                "detected": ["Speech"],
                "timestamps": []
            },
            "language": "en",
            "duration_seconds": 45.2,
            "call_metadata": {
                "parsed": True,
                "call_date": "2025-11-04",
                "call_time": "13:35:55",
                "extension": "302",
                "phone_number": "(252)792-8686",
                "direction": "Outgoing",
                "auto_flag": "Auto",
                "recording_id": "2243071124051"
            },
            "call_content": {
                "subject": "Ticket issue discussion",
                "outcome": "Resolved",
                "customer_name": "John Doe",
                "confidence": 0.8,
                "analysis_model": "Llama-3.2-3B-Instruct"
            },
            "customer_lookup": {
                "found": True,
                "customer_name": "John Doe",
                "company_name": "ABC Company",
                "company_id": 123,
                "email": "contact@abccompany.com",
                "ticket_count": 5,
                "match_source": "ticket"
            },
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        # Insert and verify
        collection.insert_one(test_doc)
        stored_doc = assert_document_stored(collection, doc_id)

        # Verify call metadata
        assert stored_doc["call_metadata"]["parsed"] is True
        assert stored_doc["call_metadata"]["extension"] == "302"
        assert stored_doc["call_metadata"]["direction"] == "Outgoing"
        assert stored_doc["call_metadata"]["phone_number"] == "(252)792-8686"

        # Verify call content
        assert stored_doc["call_content"]["subject"] == "Ticket issue discussion"
        assert stored_doc["call_content"]["outcome"] == "Resolved"
        assert stored_doc["call_content"]["customer_name"] == "John Doe"

        # Verify customer lookup
        assert stored_doc["customer_lookup"]["found"] is True
        assert stored_doc["customer_lookup"]["company_name"] == "ABC Company"

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_audio_with_emotions(self, mongodb_database, pipeline_config):
        """Test storing audio analysis with multiple detected emotions."""
        collection = mongodb_database["audio_analyses"]

        doc_id = f"test_{generate_test_id('audio_emotions')}"
        test_doc = {
            "_id": doc_id,
            "original_filename": "test_emotional_call.mp3",
            "file_path": "/tmp/test_emotional.mp3",
            "transcription": "I am very frustrated with this issue!",
            "formatted_transcription": "I am very frustrated with this issue!",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|ANGRY|><|Speech|>I am very frustrated with this issue!",
            "emotions": {
                "primary": "ANGRY",
                "detected": ["ANGRY", "SAD"],
                "timestamps": []
            },
            "audio_events": {
                "detected": ["Speech", "Cry"],
                "timestamps": []
            },
            "language": "en",
            "duration_seconds": 20.0,
            "call_metadata": {"parsed": False},
            "call_content": {
                "subject": "Customer expressing frustration",
                "outcome": "Pending Follow-up",
                "customer_name": None,
                "confidence": 0.7,
                "analysis_model": "Llama-3.2-3B-Instruct"
            },
            "customer_lookup": {"found": False},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)
        stored_doc = assert_document_stored(collection, doc_id)

        # Verify emotions
        assert stored_doc["emotions"]["primary"] == "ANGRY"
        assert "ANGRY" in stored_doc["emotions"]["detected"]
        assert "SAD" in stored_doc["emotions"]["detected"]

        # Verify audio events
        assert "Speech" in stored_doc["audio_events"]["detected"]
        assert "Cry" in stored_doc["audio_events"]["detected"]

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_long_audio_with_summary(self, mongodb_database, pipeline_config):
        """Test storing long audio analysis with LLM-generated summary."""
        collection = mongodb_database["audio_analyses"]

        doc_id = f"test_{generate_test_id('audio_long')}"
        long_transcription = "This is a very long customer support call. " * 50
        test_doc = {
            "_id": doc_id,
            "original_filename": "test_long_call.mp3",
            "file_path": "/tmp/test_long.mp3",
            "transcription": long_transcription,
            "formatted_transcription": long_transcription,
            "transcription_summary": "Customer called about account issues. Support provided resolution steps. Issue marked as resolved.",
            "raw_transcription": f"<|en|><|NEUTRAL|><|Speech|>{long_transcription}",
            "emotions": {
                "primary": "NEUTRAL",
                "detected": ["NEUTRAL"],
                "timestamps": []
            },
            "audio_events": {
                "detected": ["Speech"],
                "timestamps": []
            },
            "language": "en",
            "duration_seconds": 185.5,  # Over 2 minutes, should have summary
            "call_metadata": {"parsed": False},
            "call_content": {
                "subject": "Account issues",
                "outcome": "Resolved",
                "customer_name": None,
                "confidence": 0.8,
                "analysis_model": "Llama-3.2-3B-Instruct"
            },
            "customer_lookup": {"found": False},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)
        stored_doc = assert_document_stored(collection, doc_id)

        # Verify summary exists for long audio
        assert stored_doc["transcription_summary"] is not None
        assert len(stored_doc["transcription_summary"]) > 0
        assert "Customer called about account issues" in stored_doc["transcription_summary"]
        assert stored_doc["duration_seconds"] > 120  # Over 2 minutes

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_multiple_audio_analyses(self, mongodb_database, pipeline_config):
        """Test storing multiple audio analyses in batch."""
        collection = mongodb_database["audio_analyses"]

        docs = []
        doc_ids = []

        for i in range(5):
            doc_id = f"test_{generate_test_id(f'audio_batch_{i}')}"
            doc_ids.append(doc_id)
            docs.append({
                "_id": doc_id,
                "original_filename": f"test_call_{i}.mp3",
                "file_path": f"/tmp/test_audio_{i}.mp3",
                "transcription": f"Test transcription {i}",
                "formatted_transcription": f"Test transcription {i}",
                "transcription_summary": None,
                "raw_transcription": f"<|en|><|NEUTRAL|><|Speech|>Test transcription {i}",
                "emotions": {
                    "primary": "NEUTRAL",
                    "detected": ["NEUTRAL"],
                    "timestamps": []
                },
                "audio_events": {
                    "detected": ["Speech"],
                    "timestamps": []
                },
                "language": "en",
                "duration_seconds": 10.0 + i,
                "call_metadata": {"parsed": False},
                "call_content": {
                    "subject": None,
                    "outcome": None,
                    "customer_name": None,
                    "confidence": 0.0,
                    "analysis_model": ""
                },
                "customer_lookup": {"found": False},
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })

        # Batch insert
        result = collection.insert_many(docs)
        assert len(result.inserted_ids) == 5

        # Verify all documents
        for doc_id in doc_ids:
            stored_doc = assert_document_stored(collection, doc_id)
            assert stored_doc["transcription"].startswith("Test transcription")

        # Cleanup
        collection.delete_many({"_id": {"$in": doc_ids}})

    @pytest.mark.requires_mongodb
    def test_query_audio_by_date_range(self, mongodb_database, pipeline_config):
        """Test querying audio analyses by date range."""
        collection = mongodb_database["audio_analyses"]

        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)

        # Create test documents with different timestamps
        doc_ids = []
        for i, created_at in enumerate([yesterday, now]):
            doc_id = f"test_{generate_test_id(f'audio_date_{i}')}"
            doc_ids.append(doc_id)
            collection.insert_one({
                "_id": doc_id,
                "original_filename": f"test_call_{i}.mp3",
                "file_path": f"/tmp/test_{i}.mp3",
                "transcription": f"Test {i}",
                "formatted_transcription": f"Test {i}",
                "transcription_summary": None,
                "raw_transcription": f"<|en|><|NEUTRAL|><|Speech|>Test {i}",
                "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
                "audio_events": {"detected": ["Speech"], "timestamps": []},
                "language": "en",
                "duration_seconds": 10.0,
                "call_metadata": {"parsed": False},
                "call_content": {"subject": None, "outcome": None, "customer_name": None, "confidence": 0.0, "analysis_model": ""},
                "customer_lookup": {"found": False},
                "created_at": created_at,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })

        # Query recent documents (last hour)
        one_hour_ago = now - timedelta(hours=1)
        recent_docs = list(collection.find({
            "created_at": {"$gte": one_hour_ago},
            "is_test": True
        }))

        # Should find only the document created "now"
        assert len(recent_docs) >= 1

        # Query all test documents
        all_test_docs = list(collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(all_test_docs) >= 2

        # Cleanup
        collection.delete_many({"_id": {"$in": doc_ids}})

    @pytest.mark.requires_mongodb
    def test_update_audio_analysis(self, mongodb_database, pipeline_config):
        """Test updating an existing audio analysis document."""
        collection = mongodb_database["audio_analyses"]

        doc_id = f"test_{generate_test_id('audio_update')}"

        # Insert initial document
        collection.insert_one({
            "_id": doc_id,
            "original_filename": "test_update.mp3",
            "file_path": "/tmp/test.mp3",
            "transcription": "Original transcription",
            "formatted_transcription": "Original transcription",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Original transcription",
            "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
            "audio_events": {"detected": ["Speech"], "timestamps": []},
            "language": "en",
            "duration_seconds": 10.0,
            "call_metadata": {"parsed": False},
            "call_content": {"subject": None, "outcome": None, "customer_name": None, "confidence": 0.0, "analysis_model": ""},
            "customer_lookup": {"found": False},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Update with call content analysis
        update_result = collection.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "call_content": {
                        "subject": "Updated subject",
                        "outcome": "Resolved",
                        "customer_name": "Jane Smith",
                        "confidence": 0.85,
                        "analysis_model": "Llama-3.2-3B-Instruct"
                    },
                    "updated_at": datetime.utcnow()
                }
            }
        )

        assert update_result.modified_count == 1

        # Verify update
        updated_doc = assert_document_stored(collection, doc_id)
        assert updated_doc["call_content"]["subject"] == "Updated subject"
        assert updated_doc["call_content"]["outcome"] == "Resolved"
        assert updated_doc["call_content"]["customer_name"] == "Jane Smith"
        assert "updated_at" in updated_doc

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_audio_document_schema_validation(self, mongodb_database, pipeline_config):
        """Test that audio analysis documents match expected schema."""
        collection = mongodb_database["audio_analyses"]

        doc_id = f"test_{generate_test_id('audio_schema')}"
        test_doc = {
            "_id": doc_id,
            "original_filename": "test_schema.mp3",
            "file_path": "/tmp/test_schema.mp3",
            "transcription": "Test transcription",
            "formatted_transcription": "Test transcription",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Test transcription",
            "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
            "audio_events": {"detected": ["Speech"], "timestamps": []},
            "language": "en",
            "duration_seconds": 15.0,
            "call_metadata": {"parsed": False},
            "call_content": {"subject": None, "outcome": None, "customer_name": None, "confidence": 0.0, "analysis_model": ""},
            "customer_lookup": {"found": False},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)
        stored_doc = assert_document_stored(collection, doc_id)

        # Validate schema
        expected_schema = {
            "original_filename": str,
            "file_path": str,
            "transcription": str,
            "formatted_transcription": str,
            "raw_transcription": str,
            "emotions": dict,
            "audio_events": dict,
            "language": str,
            "duration_seconds": (int, float),
            "call_metadata": dict,
            "call_content": dict,
            "customer_lookup": dict,
            "created_at": datetime,
        }

        for field, expected_type in expected_schema.items():
            assert field in stored_doc, f"Missing field: {field}"
            if isinstance(expected_type, tuple):
                assert isinstance(stored_doc[field], expected_type), f"Field {field} has wrong type"
            else:
                assert isinstance(stored_doc[field], expected_type), f"Field {field} has wrong type"

        # Cleanup
        collection.delete_one({"_id": doc_id})
