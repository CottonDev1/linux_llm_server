"""
Feedback Service for SQL Generation Learning.

Extracted from ewr_learning_agent implementation.
Provides feedback recording, preference learning, and pattern extraction.

Usage:
    service = FeedbackService()  # Uses MONGODB_URI from config
    await service.initialize()

    # Record feedback
    await service.record_feedback(
        query="Show tickets for today",
        answer="SELECT * FROM CentralTickets...",
        feedback_type=FeedbackType.THUMBS_UP
    )

    # Get cached correction
    correction = await service.get_cached_correction("Show tickets for today")
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

# Import config for default values
from config import MONGODB_URI

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    RATING = "rating"
    COMMENT = "comment"


@dataclass
class FeedbackRecord:
    """Record of user feedback."""
    feedback_id: str
    feedback_type: FeedbackType
    query: str
    answer: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    correction: Optional[str] = None
    rating: Optional[int] = None
    comment: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "query": self.query,
            "answer": self.answer,
            "created_at": self.created_at.isoformat(),
            "correction": self.correction,
            "rating": self.rating,
            "comment": self.comment,
            "sources": self.sources,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "processed": self.processed
        }


@dataclass
class PreferencePair:
    """Preference pair for training."""
    pair_id: str
    query: str
    chosen_response: str
    rejected_response: str
    preference_strength: float = 1.0
    source: str = "user_feedback"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair_id": self.pair_id,
            "query": self.query,
            "chosen_response": self.chosen_response,
            "rejected_response": self.rejected_response,
            "preference_strength": self.preference_strength,
            "source": self.source,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PatternMatch:
    """Pattern extracted from feedback."""
    pattern_id: str
    query_pattern: str
    expected_behavior: str
    priority: int = 0
    match_count: int = 0
    success_rate: float = 0.0
    last_matched: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "query_pattern": self.query_pattern,
            "expected_behavior": self.expected_behavior,
            "priority": self.priority,
            "match_count": self.match_count,
            "success_rate": self.success_rate,
            "last_matched": self.last_matched.isoformat() if self.last_matched else None
        }


@dataclass
class FeedbackStats:
    """Statistics about feedback."""
    total_feedback: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    correction_count: int = 0
    thumbs_up_rate: float = 0.0
    avg_rating: float = 0.0
    pending_processing: int = 0
    patterns_count: int = 0
    preference_pairs: int = 0
    cached_corrections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_feedback": self.total_feedback,
            "thumbs_up_count": self.thumbs_up_count,
            "thumbs_down_count": self.thumbs_down_count,
            "correction_count": self.correction_count,
            "thumbs_up_rate": self.thumbs_up_rate,
            "avg_rating": self.avg_rating,
            "pending_processing": self.pending_processing,
            "patterns_count": self.patterns_count,
            "preference_pairs": self.preference_pairs,
            "cached_corrections": self.cached_corrections
        }


@dataclass
class LearningConfig:
    """Configuration for learning behavior."""
    positive_boost: float = 0.1
    negative_penalty: float = 0.05
    cache_verified_corrections: bool = True
    pattern_min_occurrences: int = 3
    min_feedback_for_update: int = 5
    auto_learning_enabled: bool = False
    learning_cycle_interval_hours: int = 24


class FeedbackService:
    """
    Service for recording and processing SQL generation feedback.

    Implements feedback-driven improvement:
    - Records user feedback with full context
    - Caches verified corrections for instant replay
    - Builds preference pairs for training
    - Extracts patterns from successful/failed queries
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        config: Optional[LearningConfig] = None
    ):
        """
        Initialize the feedback service.

        Args:
            mongo_uri: MongoDB connection string
            config: Learning configuration
        """
        self._mongo_uri = mongo_uri or MONGODB_URI
        self._config = config or LearningConfig()

        # In-memory storage
        self._feedback_records: List[FeedbackRecord] = []
        self._preference_pairs: List[PreferencePair] = []
        self._patterns: Dict[str, PatternMatch] = {}
        self._verified_corrections: Dict[str, str] = {}  # query -> correction

        # MongoDB (lazy initialized)
        self._mongo_client = None
        self._feedback_collection = None
        self._preferences_collection = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize MongoDB connection.

        Returns:
            True if connected, False if using in-memory only
        """
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._mongo_client = AsyncIOMotorClient(self._mongo_uri)
            db = self._mongo_client["EWRAI"]
            self._feedback_collection = db["sql_feedback"]
            self._preferences_collection = db["sql_preferences"]

            # Test connection
            await self._mongo_client.admin.command('ping')

            # Create indexes
            await self._feedback_collection.create_index([("created_at", -1)])
            await self._feedback_collection.create_index([("processed", 1)])
            await self._feedback_collection.create_index([("query", "text")])

            self._initialized = True
            logger.info("FeedbackService connected to MongoDB")
            return True

        except Exception as e:
            logger.warning(f"MongoDB connection failed, using in-memory: {e}")
            self._initialized = True  # Mark as ready with in-memory
            return False

    async def record_feedback(
        self,
        query: str,
        answer: str,
        feedback_type: FeedbackType,
        sources: Optional[List[Dict[str, Any]]] = None,
        correction: Optional[str] = None,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackRecord:
        """
        Record user feedback with full context.

        Args:
            query: The original user query
            answer: The generated answer/SQL
            feedback_type: Type of feedback
            sources: Retrieved sources used
            correction: User-provided correction
            rating: Numeric rating 1-5
            comment: User comment
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            The recorded FeedbackRecord
        """
        feedback_id = str(uuid.uuid4())[:8]

        record = FeedbackRecord(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            query=query,
            answer=answer,
            correction=correction,
            rating=rating,
            comment=comment,
            sources=sources or [],
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )

        logger.info(
            f"Recording feedback [{feedback_id}]: type={feedback_type.value}, "
            f"query={query[:50]}..."
        )

        # Store in memory
        self._feedback_records.append(record)

        # Persist to MongoDB
        if self._feedback_collection is not None:
            try:
                await self._feedback_collection.insert_one(record.to_dict())
            except Exception as e:
                logger.warning(f"Failed to persist feedback: {e}")

        # Apply immediate actions
        await self._apply_immediate_actions(record)

        return record

    async def _apply_immediate_actions(self, record: FeedbackRecord):
        """Apply immediate actions based on feedback type."""

        if record.feedback_type == FeedbackType.CORRECTION and record.correction:
            # Cache the correction for instant replay
            if self._config.cache_verified_corrections:
                cache_key = record.query.lower().strip()
                self._verified_corrections[cache_key] = record.correction
                logger.info(f"Correction cached for: {record.query[:50]}...")

            # Create preference pair
            pair = PreferencePair(
                pair_id=str(uuid.uuid4())[:8],
                query=record.query,
                chosen_response=record.correction,
                rejected_response=record.answer,
                preference_strength=1.0,  # User corrections are strong
                source="user_correction"
            )
            self._preference_pairs.append(pair)

    async def get_cached_correction(self, query: str) -> Optional[str]:
        """
        Get cached correction for a query if available.

        Args:
            query: The user query

        Returns:
            Cached correction or None
        """
        cache_key = query.lower().strip()
        return self._verified_corrections.get(cache_key)

    async def get_learned_patterns(self, query: str) -> List[PatternMatch]:
        """
        Get patterns that match the query.

        Args:
            query: The user query

        Returns:
            List of matching patterns sorted by priority
        """
        query_lower = query.lower().strip()
        matches = []

        for pattern_query, pattern in self._patterns.items():
            if pattern_query in query_lower or query_lower in pattern_query:
                pattern.last_matched = datetime.utcnow()
                matches.append(pattern)

        matches.sort(key=lambda p: p.priority, reverse=True)
        return matches

    async def build_preference_dataset(self) -> List[PreferencePair]:
        """
        Build preference dataset from accumulated feedback.

        Creates pairs from:
        - User corrections (chosen=correction, rejected=original)
        - Thumbs up vs thumbs down comparisons

        Returns:
            List of preference pairs
        """
        pairs = list(self._preference_pairs)

        # Group feedback by query
        feedback_by_query: Dict[str, List[FeedbackRecord]] = defaultdict(list)
        for record in self._feedback_records:
            key = record.query.lower().strip()
            feedback_by_query[key].append(record)

        # Create pairs from contrasting feedback
        for query, records in feedback_by_query.items():
            positive = [r for r in records if r.feedback_type == FeedbackType.THUMBS_UP]
            negative = [r for r in records if r.feedback_type == FeedbackType.THUMBS_DOWN]

            for pos in positive:
                for neg in negative:
                    if pos.answer != neg.answer:
                        pair = PreferencePair(
                            pair_id=str(uuid.uuid4())[:8],
                            query=query,
                            chosen_response=pos.answer,
                            rejected_response=neg.answer,
                            preference_strength=0.8,
                            source="thumbs_comparison"
                        )
                        pairs.append(pair)

        logger.info(f"Built preference dataset with {len(pairs)} pairs")
        return pairs

    async def extract_patterns(self) -> List[PatternMatch]:
        """
        Extract patterns from feedback for future matching.

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Group by query
        query_counts: Dict[str, int] = defaultdict(int)
        query_success: Dict[str, int] = defaultdict(int)

        for record in self._feedback_records:
            query_key = record.query.lower().strip()
            query_counts[query_key] += 1
            if record.feedback_type == FeedbackType.THUMBS_UP:
                query_success[query_key] += 1

        # Create patterns for frequently occurring queries
        for query, count in query_counts.items():
            if count >= self._config.pattern_min_occurrences:
                success_rate = query_success.get(query, 0) / count

                pattern = PatternMatch(
                    pattern_id=str(uuid.uuid4())[:8],
                    query_pattern=query,
                    expected_behavior="handle_as_pattern",
                    priority=count,
                    match_count=count,
                    success_rate=success_rate
                )
                patterns.append(pattern)
                self._patterns[query] = pattern

        return patterns

    async def get_stats(self) -> FeedbackStats:
        """
        Get feedback and learning statistics.

        Returns:
            FeedbackStats with counts and rates
        """
        total = len(self._feedback_records)
        thumbs_up = sum(1 for r in self._feedback_records if r.feedback_type == FeedbackType.THUMBS_UP)
        thumbs_down = sum(1 for r in self._feedback_records if r.feedback_type == FeedbackType.THUMBS_DOWN)
        corrections = sum(1 for r in self._feedback_records if r.feedback_type == FeedbackType.CORRECTION)
        pending = sum(1 for r in self._feedback_records if not r.processed)

        ratings = [r.rating for r in self._feedback_records if r.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        return FeedbackStats(
            total_feedback=total,
            thumbs_up_count=thumbs_up,
            thumbs_down_count=thumbs_down,
            correction_count=corrections,
            thumbs_up_rate=thumbs_up / total if total > 0 else 0.0,
            avg_rating=avg_rating,
            pending_processing=pending,
            patterns_count=len(self._patterns),
            preference_pairs=len(self._preference_pairs),
            cached_corrections=len(self._verified_corrections)
        )

    async def get_mixed_feedback_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get queries with mixed feedback for active learning.

        These are queries that received both positive and negative feedback,
        indicating uncertainty that could benefit from review.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of query info dicts with feedback breakdown
        """
        query_feedback: Dict[str, List[FeedbackRecord]] = defaultdict(list)
        for record in self._feedback_records:
            key = record.query.lower().strip()
            query_feedback[key].append(record)

        mixed = []
        for query, records in query_feedback.items():
            types = [r.feedback_type for r in records]
            has_positive = FeedbackType.THUMBS_UP in types
            has_negative = FeedbackType.THUMBS_DOWN in types

            if has_positive and has_negative:
                mixed.append({
                    "query": query,
                    "total_feedback": len(records),
                    "positive_count": types.count(FeedbackType.THUMBS_UP),
                    "negative_count": types.count(FeedbackType.THUMBS_DOWN),
                    "reason": "Mixed user feedback"
                })

        mixed.sort(key=lambda q: q["total_feedback"], reverse=True)
        return mixed[:limit]

    async def close(self):
        """Close MongoDB connection."""
        if self._mongo_client:
            self._mongo_client.close()


# Convenience function for quick feedback recording
async def record_sql_feedback(
    query: str,
    sql: str,
    is_positive: bool,
    correction: Optional[str] = None,
    mongo_uri: Optional[str] = None
) -> FeedbackRecord:
    """
    Quick feedback recording for SQL generation.

    Args:
        query: The natural language query
        sql: The generated SQL
        is_positive: True for thumbs up, False for thumbs down
        correction: Optional user correction
        mongo_uri: Optional MongoDB URI

    Returns:
        The recorded FeedbackRecord
    """
    service = FeedbackService(mongo_uri=mongo_uri)
    await service.initialize()

    if correction:
        feedback_type = FeedbackType.CORRECTION
    elif is_positive:
        feedback_type = FeedbackType.THUMBS_UP
    else:
        feedback_type = FeedbackType.THUMBS_DOWN

    return await service.record_feedback(
        query=query,
        answer=sql,
        feedback_type=feedback_type,
        correction=correction
    )
