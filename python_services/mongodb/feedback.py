"""
MongoDB Service - Feedback Mixin

Handles feedback storage and quality scoring operations.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, TYPE_CHECKING

from pymongo import ASCENDING

from config import (
    COLLECTION_FEEDBACK,
    COLLECTION_SQL_SCHEMA_CONTEXT,
    COLLECTION_SQL_STORED_PROCEDURES,
    COLLECTION_SQL_EXAMPLES,
    COLLECTION_DOCUMENTS
)
from database_name_parser import normalize_database_name

if TYPE_CHECKING:
    from .base import MongoDBBase


class FeedbackMixin:
    """
    Mixin providing feedback and quality scoring operations.

    Methods:
        store_feedback: Store user feedback on RAG responses
        get_feedback_stats: Get aggregated feedback statistics
        get_low_performing_documents: Find documents with low quality scores
    """

    async def store_feedback(
        self: 'MongoDBBase',
        feedback_type: str,
        query: str,
        response: str,
        query_id: Optional[str] = None,
        database: Optional[str] = None,
        rating: Optional[Dict] = None,
        correction: Optional[Dict] = None,
        refinement: Optional[Dict] = None,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Store user feedback on a RAG system response.

        Args:
            feedback_type: Type of feedback (rating, correction, refinement, resolution)
            query: The original query text
            response: The response that feedback is about
            query_id: Optional ID linking to the original query session
            database: Database that was queried
            rating: Rating feedback details (is_helpful, rating, comment)
            correction: Correction feedback details
            refinement: Refinement feedback details
            document_ids: IDs of documents used in generating the response
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            Dict with success status, feedback_id, and quality_scores_updated count
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_FEEDBACK]

        feedback_id = f"feedback_{uuid.uuid4().hex[:16]}"
        now = datetime.utcnow()

        normalized_db = normalize_database_name(database) if database else None

        document = {
            "id": feedback_id,
            "feedback_type": feedback_type,
            "query": query,
            "response": response,
            "query_id": query_id,
            "database": normalized_db,
            "document_ids": document_ids or [],
            "user_id": user_id,
            "session_id": session_id,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now
        }

        if rating:
            document["rating"] = rating
        if correction:
            document["correction"] = correction
        if refinement:
            document["refinement"] = refinement

        await collection.insert_one(document)

        # Update document quality scores
        quality_scores_updated = 0
        if document_ids:
            quality_scores_updated = await self._update_document_quality_scores(
                document_ids=document_ids,
                feedback_type=feedback_type,
                is_positive=rating.get("is_helpful", True) if rating else False
            )

        # Store as failed query if correction
        if feedback_type == "correction" and correction:
            await self.store_failed_query(
                database=normalized_db or "unknown",
                prompt=query,
                sql=correction.get("original_response", ""),
                error=correction.get("error_type", "user_correction"),
                tables_involved=correction.get("affected_tables"),
                correction_id=feedback_id,
                corrected_sql=correction.get("corrected_response")
            )

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback stored successfully",
            "quality_scores_updated": quality_scores_updated
        }

    async def _update_document_quality_scores(
        self: 'MongoDBBase',
        document_ids: List[str],
        feedback_type: str,
        is_positive: bool
    ) -> int:
        """
        Update quality scores for documents based on feedback.

        The scoring algorithm:
        - Positive feedback increases score by a small amount
        - Negative feedback decreases score more significantly
        - Corrections have the largest negative impact
        - Scores are bounded between 0.0 and 1.0
        """
        if not document_ids:
            return 0

        if feedback_type == "correction":
            score_delta = -0.1
        elif feedback_type == "rating":
            score_delta = 0.02 if is_positive else -0.05
        else:
            score_delta = 0.01 if is_positive else -0.02

        updated_count = 0

        collections_to_update = [
            COLLECTION_SQL_SCHEMA_CONTEXT,
            COLLECTION_SQL_STORED_PROCEDURES,
            COLLECTION_SQL_EXAMPLES,
            COLLECTION_DOCUMENTS
        ]

        for coll_name in collections_to_update:
            collection = self.db[coll_name]

            for doc_id in document_ids:
                result = await collection.update_one(
                    {"id": doc_id},
                    [
                        {
                            "$set": {
                                "quality_score": {
                                    "$max": [
                                        0.0,
                                        {
                                            "$min": [
                                                1.0,
                                                {
                                                    "$add": [
                                                        {"$ifNull": ["$quality_score", 0.7]},
                                                        score_delta
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                },
                                "feedback_count": {
                                    "$add": [{"$ifNull": ["$feedback_count", 0]}, 1]
                                },
                                "last_feedback_date": datetime.utcnow()
                            }
                        }
                    ]
                )

                if result.modified_count > 0:
                    updated_count += 1

        return updated_count

    async def get_feedback_stats(
        self: 'MongoDBBase',
        database: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get aggregated feedback statistics.

        Args:
            database: Optional database filter
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            Dict with comprehensive feedback statistics
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_FEEDBACK]

        match_filter = {}
        if database:
            match_filter["database"] = normalize_database_name(database)
        if start_date or end_date:
            match_filter["created_at"] = {}
            if start_date:
                match_filter["created_at"]["$gte"] = start_date
            if end_date:
                match_filter["created_at"]["$lte"] = end_date

        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        pipeline = [
            {"$match": match_filter} if match_filter else {"$match": {}},
            {
                "$facet": {
                    "total": [{"$count": "count"}],
                    "by_type": [
                        {"$group": {"_id": "$feedback_type", "count": {"$sum": 1}}}
                    ],
                    "rating_stats": [
                        {"$match": {"feedback_type": "rating"}},
                        {
                            "$group": {
                                "_id": None,
                                "total": {"$sum": 1},
                                "helpful": {
                                    "$sum": {"$cond": [{"$eq": ["$rating.is_helpful", True]}, 1, 0]}
                                },
                                "not_helpful": {
                                    "$sum": {"$cond": [{"$eq": ["$rating.is_helpful", False]}, 1, 0]}
                                },
                                "ratings": {"$push": "$rating.rating"}
                            }
                        }
                    ],
                    "correction_stats": [
                        {"$match": {"feedback_type": "correction"}},
                        {
                            "$group": {
                                "_id": "$correction.error_type",
                                "count": {"$sum": 1}
                            }
                        }
                    ],
                    "by_database": [
                        {"$match": {"database": {"$ne": None}}},
                        {
                            "$group": {
                                "_id": "$database",
                                "total": {"$sum": 1},
                                "helpful": {
                                    "$sum": {
                                        "$cond": [
                                            {"$and": [
                                                {"$eq": ["$feedback_type", "rating"]},
                                                {"$eq": ["$rating.is_helpful", True]}
                                            ]},
                                            1, 0
                                        ]
                                    }
                                }
                            }
                        }
                    ],
                    "last_24h": [
                        {"$match": {"created_at": {"$gte": last_24h}}},
                        {"$count": "count"}
                    ],
                    "last_7d": [
                        {"$match": {"created_at": {"$gte": last_7d}}},
                        {"$count": "count"}
                    ]
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)

        if not results:
            return {
                "total_feedback": 0,
                "by_type": {},
                "total_ratings": 0,
                "helpful_count": 0,
                "not_helpful_count": 0,
                "helpfulness_rate": 0.0,
                "average_rating": None,
                "total_corrections": 0,
                "corrections_by_error_type": {},
                "by_database": {},
                "feedback_last_24h": 0,
                "feedback_last_7d": 0,
                "period_start": start_date,
                "period_end": end_date
            }

        data = results[0]

        total = data["total"][0]["count"] if data["total"] else 0
        by_type = {item["_id"]: item["count"] for item in data["by_type"] if item["_id"]}

        rating_data = data["rating_stats"][0] if data["rating_stats"] else {}
        total_ratings = rating_data.get("total", 0)
        helpful_count = rating_data.get("helpful", 0)
        not_helpful_count = rating_data.get("not_helpful", 0)
        helpfulness_rate = helpful_count / total_ratings if total_ratings > 0 else 0.0

        ratings = rating_data.get("ratings", [])
        valid_ratings = [r for r in ratings if r is not None]
        average_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None

        corrections_by_error_type = {
            item["_id"]: item["count"]
            for item in data["correction_stats"]
            if item["_id"]
        }
        total_corrections = sum(corrections_by_error_type.values())

        by_database = {
            item["_id"]: {"total": item["total"], "helpful": item["helpful"]}
            for item in data["by_database"]
            if item["_id"]
        }

        feedback_last_24h = data["last_24h"][0]["count"] if data["last_24h"] else 0
        feedback_last_7d = data["last_7d"][0]["count"] if data["last_7d"] else 0

        return {
            "total_feedback": total,
            "by_type": by_type,
            "total_ratings": total_ratings,
            "helpful_count": helpful_count,
            "not_helpful_count": not_helpful_count,
            "helpfulness_rate": helpfulness_rate,
            "average_rating": average_rating,
            "total_corrections": total_corrections,
            "corrections_by_error_type": corrections_by_error_type,
            "by_database": by_database,
            "feedback_last_24h": feedback_last_24h,
            "feedback_last_7d": feedback_last_7d,
            "period_start": start_date,
            "period_end": end_date
        }

    async def get_low_performing_documents(
        self: 'MongoDBBase',
        threshold: float = 0.5,
        min_feedback_count: int = 3,
        database: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Find documents with low quality scores based on feedback.

        Args:
            threshold: Quality score threshold (0.0-1.0)
            min_feedback_count: Minimum feedback count to be included
            database: Optional database filter
            limit: Maximum number of documents to return

        Returns:
            List of low-performing documents with their quality metrics
        """
        if not self.is_initialized:
            await self.initialize()

        low_performing = []

        collections_to_check = [
            (COLLECTION_SQL_SCHEMA_CONTEXT, "schema"),
            (COLLECTION_SQL_STORED_PROCEDURES, "procedure"),
            (COLLECTION_SQL_EXAMPLES, "example"),
            (COLLECTION_DOCUMENTS, "document")
        ]

        for coll_name, doc_type in collections_to_check:
            collection = self.db[coll_name]

            filter_query = {
                "quality_score": {"$lt": threshold, "$exists": True},
                "feedback_count": {"$gte": min_feedback_count}
            }

            if database:
                normalized_db = normalize_database_name(database)
                filter_query["database"] = normalized_db

            cursor = collection.find(
                filter_query,
                {
                    "id": 1,
                    "title": 1,
                    "table_name": 1,
                    "procedure_name": 1,
                    "database": 1,
                    "quality_score": 1,
                    "feedback_count": 1,
                    "last_feedback_date": 1
                }
            ).sort("quality_score", ASCENDING).limit(limit)

            docs = await cursor.to_list(length=limit)

            for doc in docs:
                title = doc.get("title") or doc.get("table_name") or doc.get("procedure_name") or "Unknown"

                low_performing.append({
                    "document_id": doc.get("id"),
                    "document_type": doc_type,
                    "title": title,
                    "database": doc.get("database"),
                    "quality_score": doc.get("quality_score", 0.0),
                    "total_feedback": doc.get("feedback_count", 0),
                    "negative_feedback": 0,
                    "correction_count": 0,
                    "last_feedback_date": doc.get("last_feedback_date"),
                    "recent_issues": []
                })

        low_performing.sort(key=lambda x: x["quality_score"])
        return low_performing[:limit]
