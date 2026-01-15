"""
MongoDB Service - Audio Analysis Mixin

Handles audio transcription storage and retrieval operations.
"""
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, TYPE_CHECKING

from config import COLLECTION_AUDIO_ANALYSIS

if TYPE_CHECKING:
    from .base import MongoDBBase


class AudioMixin:
    """
    Mixin providing audio analysis operations.

    Methods:
        store_audio_analysis: Store audio transcription with embeddings
        get_audio_analysis: Get audio analysis by ID
        search_audio_analyses: Search with filters and semantic search
        update_audio_analysis: Update audio analysis metadata
        delete_audio_analysis: Delete audio analysis by ID
        get_audio_stats: Get audio analysis statistics
    """

    def _parse_filename_datetime(self, filename: str) -> tuple:
        """
        Parse date and time from filename pattern: YYYYMMDD-HHMMSS_...
        Returns (call_date, call_time, call_datetime)
        """
        match = re.match(r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_', filename)
        if not match:
            return None, None, None

        year, month, day, hour, minute, second = match.groups()

        call_date = f"{month}/{day}/{year[2:]}"

        hour_int = int(hour)
        period = "am" if hour_int < 12 else "pm"
        hour_12 = hour_int % 12
        if hour_12 == 0:
            hour_12 = 12
        call_time = f"{hour_12}:{minute} {period}"

        try:
            call_datetime = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        except ValueError:
            call_datetime = None

        return call_date, call_time, call_datetime

    async def store_audio_analysis(
        self: 'MongoDBBase',
        customer_support_staff: str,
        ewr_customer: str,
        mood: str,
        outcome: str,
        filename: str,
        transcription: str,
        raw_transcription: str,
        emotions: Dict,
        audio_events: Dict,
        language: str,
        audio_metadata: Dict,
        transcription_summary: Optional[str] = None,
        analyzed_by: str = "SenseVoice",
        analysis_version: str = "1.0",
        call_metadata: Optional[Dict] = None,
        call_content: Optional[Dict] = None,
        related_ticket_ids: Optional[List[int]] = None,
        linked_ticket_id: Optional[int] = None,
        speaker_diarization: Optional[Dict] = None
    ) -> Dict:
        """
        Store audio analysis with embeddings.

        Args:
            customer_support_staff: Staff member who handled the call
            ewr_customer: Customer name or ID
            mood: Overall mood classification
            outcome: Call outcome
            filename: Original audio filename
            transcription: Clean transcription text
            raw_transcription: Transcription with emotion/event tags
            emotions: Emotion detection results
            audio_events: Audio event detection results
            language: Detected language
            audio_metadata: Audio file metadata (duration, sample_rate, etc.)
            transcription_summary: LLM-generated summary
            analyzed_by: Analysis engine name
            analysis_version: Version of the analysis pipeline
            call_metadata: Parsed call metadata from filename
            call_content: LLM-analyzed call content
            related_ticket_ids: Related ticket IDs from EWRCentral
            linked_ticket_id: Primary linked ticket ID
            speaker_diarization: Speaker diarization results (segments, statistics)
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]

        analysis_id = str(uuid.uuid4())
        now = datetime.utcnow()

        call_date, call_time, call_datetime = self._parse_filename_datetime(filename)

        if call_metadata and call_metadata.get("parsed"):
            call_date = call_metadata.get("call_date") or call_date
            call_time = call_metadata.get("call_time") or call_time
            if call_metadata.get("call_date") and call_metadata.get("call_time"):
                try:
                    call_datetime = datetime.strptime(
                        f"{call_metadata['call_date']} {call_metadata['call_time']}",
                        "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    pass

        emotions_str = ", ".join(emotions.get("detected", []))

        embedding_text_parts = [
            f"Customer Support Staff: {customer_support_staff}. EWR Customer: {ewr_customer}.",
            f"Mood: {mood}. Outcome: {outcome}.",
            f"Emotions detected: {emotions_str}."
        ]

        if call_content and call_content.get("subject"):
            embedding_text_parts.append(f"Call Subject: {call_content['subject']}.")
        if call_content and call_content.get("customer_name"):
            embedding_text_parts.append(f"Customer Name Mentioned: {call_content['customer_name']}.")

        embedding_text_parts.append(f"Transcription: {transcription}")
        embedding_text = "\n".join(embedding_text_parts)

        embedding = await self.embedding_service.generate_embedding(embedding_text)

        document = {
            "id": analysis_id,
            "customer_support_staff": customer_support_staff,
            "ewr_customer": ewr_customer,
            "mood": mood,
            "outcome": outcome,
            "filename": filename,
            "call_date": call_date,
            "call_time": call_time,
            "call_datetime": call_datetime,
            "transcription": transcription,
            "transcription_summary": transcription_summary,
            "raw_transcription": raw_transcription,
            "emotions": emotions,
            "audio_events": audio_events,
            "language": language,
            "audio_metadata": audio_metadata,
            "embedding_text": embedding_text,
            "vector": embedding,
            "created_at": now,
            "updated_at": now,
            "analyzed_by": analyzed_by,
            "analysis_version": analysis_version,
            "call_metadata": call_metadata or {
                "call_date": None,
                "call_time": None,
                "extension": None,
                "phone_number": None,
                "direction": None,
                "auto_flag": None,
                "recording_id": None,
                "parsed": False
            },
            "call_content": call_content or {
                "subject": None,
                "outcome": None,
                "customer_name": None,
                "confidence": 0.0,
                "analysis_model": ""
            },
            "related_ticket_ids": related_ticket_ids or [],
            "linked_ticket_id": linked_ticket_id,
            "speaker_diarization": speaker_diarization or {
                "enabled": False,
                "segments": [],
                "statistics": {},
                "num_speakers": 0
            }
        }

        await collection.insert_one(document)

        print(f"Stored audio analysis: {analysis_id}")

        return {
            "success": True,
            "analysis_id": analysis_id,
            "message": "Audio analysis stored successfully"
        }

    async def get_audio_analysis(self: 'MongoDBBase', analysis_id: str) -> Optional[Dict]:
        """Get audio analysis by ID"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]
        doc = await collection.find_one({"id": analysis_id})

        if not doc:
            return None

        doc.pop("_id", None)
        doc.pop("vector", None)

        return doc

    async def search_audio_analyses(
        self: 'MongoDBBase',
        query: Optional[str] = None,
        limit: int = 10,
        customer_support_staff: Optional[str] = None,
        ewr_customer: Optional[str] = None,
        mood: Optional[str] = None,
        outcome: Optional[str] = None,
        emotion: Optional[str] = None,
        language: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict]:
        """Search audio analyses with filters and optional semantic search"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]

        filter_query = {}
        if customer_support_staff:
            filter_query["customer_support_staff"] = {"$regex": customer_support_staff, "$options": "i"}
        if ewr_customer:
            filter_query["ewr_customer"] = {"$regex": ewr_customer, "$options": "i"}
        if mood:
            filter_query["mood"] = mood
        if outcome:
            filter_query["outcome"] = outcome
        if emotion:
            filter_query["$or"] = [
                {"emotions.primary": emotion},
                {"emotions.detected": emotion}
            ]
        if language:
            filter_query["language"] = language

        if date_from or date_to:
            date_filter = {}
            if date_from:
                try:
                    from_date = datetime.strptime(date_from, "%Y-%m-%d")
                    date_filter["$gte"] = from_date
                except ValueError:
                    pass
            if date_to:
                try:
                    to_date = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                    date_filter["$lt"] = to_date
                except ValueError:
                    pass
            if date_filter:
                filter_query["call_datetime"] = date_filter

        if not query or query.strip() == "" or query == "*":
            cursor = collection.find(filter_query).sort("call_datetime", -1).limit(limit)
            results = await cursor.to_list(length=limit)
        else:
            query_vector = await self.embedding_service.generate_embedding(query)
            results = await self._vector_search(
                COLLECTION_AUDIO_ANALYSIS,
                query_vector,
                limit=limit,
                filter_query=filter_query if filter_query else None
            )

        formatted = []
        for doc in results:
            doc.pop("_id", None)
            doc.pop("vector", None)
            doc.pop("embedding_text", None)
            doc["relevance_score"] = doc.pop("_similarity", 0.0)
            doc.pop("_distance", None)
            if doc.get("call_datetime"):
                doc["call_datetime"] = doc["call_datetime"].isoformat()
            if doc.get("created_at"):
                doc["created_at"] = doc["created_at"].isoformat()
            if doc.get("updated_at"):
                doc["updated_at"] = doc["updated_at"].isoformat()
            formatted.append(doc)

        return formatted

    async def update_audio_analysis(
        self: 'MongoDBBase',
        analysis_id: str,
        update_fields: Dict
    ) -> bool:
        """Update audio analysis metadata by ID"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]

        result = await collection.update_one(
            {"id": analysis_id},
            {"$set": update_fields}
        )

        return result.modified_count > 0 or result.matched_count > 0

    async def delete_audio_analysis(self: 'MongoDBBase', analysis_id: str) -> Dict:
        """Delete audio analysis by ID"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]
        result = await collection.delete_one({"id": analysis_id})

        return {
            "success": result.deleted_count > 0,
            "message": f"Deleted {result.deleted_count} audio analysis" if result.deleted_count > 0 else "Audio analysis not found"
        }

    async def get_audio_stats(self: 'MongoDBBase') -> Dict:
        """Get audio analysis statistics"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]

        pipeline = [
            {
                "$facet": {
                    "total": [{"$count": "count"}],
                    "by_mood": [{"$group": {"_id": "$mood", "count": {"$sum": 1}}}],
                    "by_outcome": [{"$group": {"_id": "$outcome", "count": {"$sum": 1}}}],
                    "by_language": [{"$group": {"_id": "$language", "count": {"$sum": 1}}}],
                    "by_emotion": [{"$group": {"_id": "$emotions.primary", "count": {"$sum": 1}}}],
                    "duration_stats": [
                        {
                            "$group": {
                                "_id": None,
                                "total_duration": {"$sum": "$audio_metadata.duration_seconds"},
                                "avg_duration": {"$avg": "$audio_metadata.duration_seconds"}
                            }
                        }
                    ]
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)

        if not results:
            return {
                "total_analyses": 0,
                "by_mood": {},
                "by_outcome": {},
                "by_language": {},
                "by_emotion": {},
                "total_duration_hours": 0.0,
                "average_duration_seconds": 0.0
            }

        data = results[0]

        total = data["total"][0]["count"] if data["total"] else 0
        by_mood = {item["_id"]: item["count"] for item in data["by_mood"] if item["_id"]}
        by_outcome = {item["_id"]: item["count"] for item in data["by_outcome"] if item["_id"]}
        by_language = {item["_id"]: item["count"] for item in data["by_language"] if item["_id"]}
        by_emotion = {item["_id"]: item["count"] for item in data["by_emotion"] if item["_id"]}

        duration_data = data["duration_stats"][0] if data["duration_stats"] else {}
        total_duration_seconds = duration_data.get("total_duration", 0.0)
        avg_duration = duration_data.get("avg_duration", 0.0)

        return {
            "total_analyses": total,
            "by_mood": by_mood,
            "by_outcome": by_outcome,
            "by_language": by_language,
            "by_emotion": by_emotion,
            "total_duration_hours": total_duration_seconds / 3600.0,
            "average_duration_seconds": avg_duration
        }
