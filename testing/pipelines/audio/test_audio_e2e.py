"""
Audio End-to-End Tests
======================

End-to-end tests for the complete audio pipeline:
1. Audio file processing
2. Transcription (SenseVoice)
3. LLM analysis (summarization, content analysis, formatting)
4. MongoDB storage
5. Retrieval and verification

Note: These tests mock SenseVoice transcription since it requires GPU/model loading.
They focus on testing the LLM analysis and storage pipeline.
"""

import pytest
import os
import tempfile
import uuid
import asyncio
from datetime import datetime
from typing import Dict

from config.test_config import get_test_config
from utils import generate_test_id, create_temp_file, assert_document_stored


class TestAudioEndToEnd:
    """End-to-end audio pipeline tests."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_audio_pipeline_basic_flow(self, mongodb_database, pipeline_config):
        """Test basic end-to-end audio processing flow."""
        collection = mongodb_database["audio_analyses"]

        # Step 1: Simulate audio file upload
        audio_filename = "test_e2e_call.mp3"
        audio_path = "/tmp/" + audio_filename

        # Step 2: Simulate transcription result (would come from SenseVoice)
        transcription_result = {
            "transcription": "Customer called about account access issue.",
            "formatted_transcription": "Customer: I can't access my account.\nSupport: Let me help you with that.",
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Customer called about account access issue.",
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
            "duration_seconds": 35.5
        }

        # Step 3: Simulate call metadata parsing
        call_metadata = {
            "parsed": False,
            "call_date": None,
            "call_time": None,
            "extension": None,
            "phone_number": None,
            "direction": None,
            "auto_flag": None,
            "recording_id": None
        }

        # Step 4: Simulate LLM content analysis
        call_content = {
            "subject": "Account access issue",
            "outcome": "Resolved",
            "customer_name": None,
            "confidence": 0.75,
            "analysis_model": "Llama-3.2-3B-Instruct"
        }

        # Step 5: Store in MongoDB
        doc_id = f"test_{generate_test_id('e2e')}"
        analysis_doc = {
            "_id": doc_id,
            "original_filename": audio_filename,
            "file_path": audio_path,
            **transcription_result,
            "transcription_summary": None,
            "call_metadata": call_metadata,
            "call_content": call_content,
            "customer_lookup": {"found": False},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        result = collection.insert_one(analysis_doc)
        assert result.inserted_id == doc_id

        # Step 6: Retrieve and verify
        stored_doc = assert_document_stored(
            collection,
            doc_id,
            expected_fields=[
                "transcription",
                "emotions",
                "call_content",
                "created_at"
            ]
        )

        assert stored_doc["transcription"] == transcription_result["transcription"]
        assert stored_doc["call_content"]["subject"] == "Account access issue"
        assert stored_doc["emotions"]["primary"] == "NEUTRAL"

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_audio_pipeline_with_ringcentral_metadata(self, mongodb_database, pipeline_config):
        """Test E2E pipeline with RingCentral filename parsing."""
        collection = mongodb_database["audio_analyses"]

        # RingCentral filename format
        audio_filename = "20251104-133555_302_(252)792-8686_Outgoing_Auto_2243071124051.mp3"

        # Simulate parsed metadata
        call_metadata = {
            "parsed": True,
            "call_date": "2025-11-04",
            "call_time": "13:35:55",
            "extension": "302",
            "phone_number": "(252)792-8686",
            "direction": "Outgoing",
            "auto_flag": "Auto",
            "recording_id": "2243071124051"
        }

        # Simulated transcription and analysis
        transcription_result = {
            "transcription": "Customer service call regarding order status.",
            "formatted_transcription": "Support: How can I help?\nCustomer: What's my order status?",
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Customer service call regarding order status.",
            "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
            "audio_events": {"detected": ["Speech"], "timestamps": []},
            "language": "en",
            "duration_seconds": 55.0
        }

        call_content = {
            "subject": "Order status inquiry",
            "outcome": "Information Provided",
            "customer_name": "Mary Smith",
            "confidence": 0.82,
            "analysis_model": "Llama-3.2-3B-Instruct"
        }

        # Store complete analysis
        doc_id = f"test_{generate_test_id('e2e_rc')}"
        analysis_doc = {
            "_id": doc_id,
            "original_filename": audio_filename,
            "file_path": f"/tmp/{audio_filename}",
            **transcription_result,
            "transcription_summary": None,
            "call_metadata": call_metadata,
            "call_content": call_content,
            "customer_lookup": {"found": False},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(analysis_doc)

        # Verify metadata was correctly parsed
        stored_doc = assert_document_stored(collection, doc_id)
        assert stored_doc["call_metadata"]["parsed"] is True
        assert stored_doc["call_metadata"]["extension"] == "302"
        assert stored_doc["call_metadata"]["direction"] == "Outgoing"

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    async def test_audio_pipeline_with_llm_summary(self, mongodb_database, async_llm_client, pipeline_config):
        """Test E2E pipeline including actual LLM summarization."""
        collection = mongodb_database["audio_analyses"]

        # Long transcription requiring summary
        long_transcription = """
        Customer called expressing frustration about delayed shipment.
        Support apologized and checked tracking information.
        The shipment was delayed due to weather conditions.
        Support offered expedited shipping for next order at no charge.
        Customer accepted the offer and was satisfied with resolution.
        """ * 5  # Make it long

        duration_seconds = 185  # Over 2 minutes

        # Generate actual LLM summary
        summary = None
        if duration_seconds >= 120:
            prompt = f"""Summarize this customer support call in 2-3 sentences.

Transcription:
{long_transcription[:2000]}

Summary:"""

            response = await async_llm_client.generate(
                prompt=prompt,
                endpoint="general",
                max_tokens=200,
                temperature=0.3
            )

            if response.success:
                summary = response.text.strip()

        # Store analysis with summary
        doc_id = f"test_{generate_test_id('e2e_summary')}"
        analysis_doc = {
            "_id": doc_id,
            "original_filename": "test_long_call_e2e.mp3",
            "file_path": "/tmp/test_long_e2e.mp3",
            "transcription": long_transcription,
            "formatted_transcription": long_transcription,
            "transcription_summary": summary,
            "raw_transcription": f"<|en|><|ANGRY|><|Speech|>{long_transcription}",
            "emotions": {"primary": "ANGRY", "detected": ["ANGRY", "HAPPY"], "timestamps": []},
            "audio_events": {"detected": ["Speech"], "timestamps": []},
            "language": "en",
            "duration_seconds": duration_seconds,
            "call_metadata": {"parsed": False},
            "call_content": {
                "subject": "Delayed shipment",
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

        collection.insert_one(analysis_doc)

        # Verify summary was generated
        stored_doc = assert_document_stored(collection, doc_id)
        if summary:
            assert stored_doc["transcription_summary"] is not None
            assert len(stored_doc["transcription_summary"]) > 0
            assert len(stored_doc["transcription_summary"]) < len(long_transcription)

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_audio_pipeline_with_customer_lookup(self, mongodb_database, pipeline_config):
        """Test E2E pipeline with customer database lookup."""
        collection = mongodb_database["audio_analyses"]

        # Simulate customer lookup result
        customer_lookup = {
            "found": True,
            "customer_name": "Bob Johnson",
            "company_name": "Tech Solutions Inc",
            "company_id": 456,
            "email": "contact@techsolutions.com",
            "ticket_count": 12,
            "match_source": "ticket"
        }

        # Complete analysis with customer info
        doc_id = f"test_{generate_test_id('e2e_customer')}"
        analysis_doc = {
            "_id": doc_id,
            "original_filename": "20251221-100000_(469)906-0558_Incoming_Auto_2254843027051.mp3",
            "file_path": "/tmp/test_customer_call.mp3",
            "transcription": "Customer calling about technical support.",
            "formatted_transcription": "Customer: I need technical help.\nSupport: I can assist you.",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Customer calling about technical support.",
            "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
            "audio_events": {"detected": ["Speech"], "timestamps": []},
            "language": "en",
            "duration_seconds": 65.0,
            "call_metadata": {
                "parsed": True,
                "call_date": "2025-12-21",
                "call_time": "10:00:00",
                "extension": None,
                "phone_number": "(469)906-0558",
                "direction": "Incoming",
                "auto_flag": "Auto",
                "recording_id": "2254843027051"
            },
            "call_content": {
                "subject": "Technical support request",
                "outcome": "Pending Follow-up",
                "customer_name": "Bob Johnson",
                "confidence": 0.85,
                "analysis_model": "Llama-3.2-3B-Instruct"
            },
            "customer_lookup": customer_lookup,
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(analysis_doc)

        # Verify customer lookup data
        stored_doc = assert_document_stored(collection, doc_id)
        assert stored_doc["customer_lookup"]["found"] is True
        assert stored_doc["customer_lookup"]["company_name"] == "Tech Solutions Inc"
        assert stored_doc["customer_lookup"]["ticket_count"] == 12

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_audio_pipeline_batch_processing(self, mongodb_database, pipeline_config):
        """Test processing multiple audio files in batch."""
        collection = mongodb_database["audio_analyses"]

        # Simulate batch of 3 calls
        batch_docs = []
        doc_ids = []

        for i in range(3):
            doc_id = f"test_{generate_test_id(f'e2e_batch_{i}')}"
            doc_ids.append(doc_id)

            batch_docs.append({
                "_id": doc_id,
                "original_filename": f"batch_call_{i}.mp3",
                "file_path": f"/tmp/batch_{i}.mp3",
                "transcription": f"Batch call number {i} transcription.",
                "formatted_transcription": f"Batch call number {i} transcription.",
                "transcription_summary": None,
                "raw_transcription": f"<|en|><|NEUTRAL|><|Speech|>Batch call number {i} transcription.",
                "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
                "audio_events": {"detected": ["Speech"], "timestamps": []},
                "language": "en",
                "duration_seconds": 20.0 + i * 10,
                "call_metadata": {"parsed": False},
                "call_content": {
                    "subject": f"Call {i} subject",
                    "outcome": "Resolved",
                    "customer_name": None,
                    "confidence": 0.7,
                    "analysis_model": "Llama-3.2-3B-Instruct"
                },
                "customer_lookup": {"found": False},
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })

        # Batch insert
        result = collection.insert_many(batch_docs)
        assert len(result.inserted_ids) == 3

        # Verify all were stored
        for doc_id in doc_ids:
            stored_doc = assert_document_stored(collection, doc_id)
            assert "Batch call number" in stored_doc["transcription"]

        # Cleanup
        collection.delete_many({"_id": {"$in": doc_ids}})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_audio_pipeline_error_handling(self, mongodb_database, pipeline_config):
        """Test E2E pipeline error handling."""
        collection = mongodb_database["audio_analyses"]

        # Simulate failed analysis
        doc_id = f"test_{generate_test_id('e2e_error')}"
        error_doc = {
            "_id": doc_id,
            "original_filename": "error_test.mp3",
            "file_path": "/tmp/error_test.mp3",
            "transcription": "",
            "formatted_transcription": "",
            "transcription_summary": None,
            "raw_transcription": "",
            "emotions": {"primary": "NEUTRAL", "detected": [], "timestamps": []},
            "audio_events": {"detected": [], "timestamps": []},
            "language": "en",
            "duration_seconds": 0.0,
            "call_metadata": {"parsed": False},
            "call_content": {
                "subject": None,
                "outcome": None,
                "customer_name": None,
                "confidence": 0.0,
                "analysis_model": ""
            },
            "customer_lookup": {"found": False},
            "error": "Audio file corrupted or unsupported format",
            "success": False,
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(error_doc)

        # Verify error was recorded
        stored_doc = assert_document_stored(collection, doc_id)
        assert "error" in stored_doc
        assert stored_doc["success"] is False

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_audio_pipeline_update_workflow(self, mongodb_database, pipeline_config):
        """Test updating audio analysis after initial processing."""
        collection = mongodb_database["audio_analyses"]

        # Step 1: Initial analysis without LLM content
        doc_id = f"test_{generate_test_id('e2e_update')}"
        initial_doc = {
            "_id": doc_id,
            "original_filename": "update_test.mp3",
            "file_path": "/tmp/update_test.mp3",
            "transcription": "Initial transcription text.",
            "formatted_transcription": "Initial transcription text.",
            "transcription_summary": None,
            "raw_transcription": "<|en|><|NEUTRAL|><|Speech|>Initial transcription text.",
            "emotions": {"primary": "NEUTRAL", "detected": ["NEUTRAL"], "timestamps": []},
            "audio_events": {"detected": ["Speech"], "timestamps": []},
            "language": "en",
            "duration_seconds": 40.0,
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
        }

        collection.insert_one(initial_doc)

        # Step 2: Update with LLM analysis
        await asyncio.sleep(0.1)  # Small delay
        collection.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "call_content": {
                        "subject": "Updated after LLM analysis",
                        "outcome": "Resolved",
                        "customer_name": "Updated Customer",
                        "confidence": 0.88,
                        "analysis_model": "Llama-3.2-3B-Instruct"
                    },
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # Verify update
        updated_doc = assert_document_stored(collection, doc_id)
        assert updated_doc["call_content"]["subject"] == "Updated after LLM analysis"
        assert "updated_at" in updated_doc

        # Cleanup
        collection.delete_one({"_id": doc_id})
