"""
Learning Feedback Step - Records and applies feedback for continuous improvement

This step implements the learning loop that enables the RAG pipeline to
improve over time based on user feedback. It handles:

1. **Recording Successful Retrievals**: Stores query-document pairs that
   received positive feedback for future reference.

2. **Storing User Feedback**: Captures thumbs up/down and detailed feedback
   with full context for analysis.

3. **Updating Document Relevance**: Adjusts document scores based on
   aggregated feedback to improve future retrievals.

4. **Preference Learning**: Builds a dataset of preferences for training
   reward models and optimizing retrieval.

Why Feedback Matters:
--------------------
RAG systems often fail in subtle ways that are hard to detect programmatically:
- Retrieved documents seem relevant but don't answer the question
- Answers are technically correct but not helpful
- Critical context is missing from retrieval

User feedback provides ground truth signal that enables:
- 10-15% continuous improvement in retrieval quality
- Identification of edge cases and failure patterns
- Data for fine-tuning and RLHF

Implementation Notes:
--------------------
This step is typically run AFTER the generation step when user feedback
is available. It can also run asynchronously in the background to avoid
adding latency to the response path.

Feedback is stored in MongoDB with:
- Full query context (original, rewritten, expanded)
- Retrieved documents and their grading scores
- User feedback type and optional text
- Timestamp and session information
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import logging
import uuid

from .base import (
    PipelineStep,
    PipelineContext,
    StepResult,
    RetrievedDocument,
)


class FeedbackType:
    """Types of feedback that can be recorded."""
    POSITIVE = "positive"  # Thumbs up
    NEGATIVE = "negative"  # Thumbs down
    CORRECTION = "correction"  # User provided correction
    IMPLICIT_POSITIVE = "implicit_positive"  # User didn't complain / continued session
    IMPLICIT_NEGATIVE = "implicit_negative"  # User retried query


class LearningFeedbackStep(PipelineStep):
    """
    Records feedback and updates relevance scores for learning.

    This step can be used in two modes:
    1. **Recording mode**: After user provides feedback, stores the
       complete context for learning.
    2. **Application mode**: Before retrieval, boosts documents that
       performed well for similar queries.

    The step maintains feedback in a MongoDB collection with indexes
    for efficient similarity lookup.
    """

    def __init__(
        self,
        mongodb_service: Any,
        embedding_service: Any,
        feedback_collection: str = "document_feedback",
        successful_retrievals_collection: str = "successful_retrievals",
        relevance_boost_weight: float = 0.1,
        min_feedback_count: int = 3,
        feedback_decay_days: int = 90,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the learning feedback step.

        Args:
            mongodb_service: MongoDBService instance for database operations
            embedding_service: EmbeddingService for query similarity
            feedback_collection: Collection name for storing feedback
            successful_retrievals_collection: Collection for positive examples
            relevance_boost_weight: How much to boost based on feedback (0-1)
            min_feedback_count: Minimum feedback count before applying boosts
            feedback_decay_days: Days before old feedback is discounted
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.mongodb_service = mongodb_service
        self.embedding_service = embedding_service
        self.feedback_collection = feedback_collection
        self.successful_retrievals_collection = successful_retrievals_collection
        self.relevance_boost_weight = relevance_boost_weight
        self.min_feedback_count = min_feedback_count
        self.feedback_decay_days = feedback_decay_days

    @property
    def name(self) -> str:
        return "LearningFeedback"

    @property
    def requires(self) -> Set[str]:
        return {"query_id", "original_query"}

    @property
    def produces(self) -> Set[str]:
        return set()  # This step primarily has side effects (database writes)

    async def execute(self, context: PipelineContext) -> StepResult:
        """
        Execute feedback recording or application based on context.

        This is a pass-through step that doesn't modify the pipeline
        results. It's mainly used for recording successful retrievals
        after generation.
        """
        # This step is typically called for side effects (recording)
        # The actual recording happens through explicit method calls
        return StepResult(
            success=True,
            data={},
            metadata={"action": "passthrough"},
        )

    async def record_feedback(
        self,
        context: PipelineContext,
        feedback_type: str,
        feedback_text: Optional[str] = None,
        corrected_answer: Optional[str] = None,
        user_id: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> StepResult:
        """
        Record user feedback for a query-response pair.

        This method stores the complete context along with feedback:
        - Original and rewritten queries
        - Retrieved documents with scores
        - User feedback type and text
        - Session information

        Args:
            context: The pipeline context from the query
            feedback_type: Type of feedback (positive, negative, correction)
            feedback_text: Optional detailed feedback text
            corrected_answer: If correction type, the user's correction
            user_id: Optional user identifier
            answer: The answer that was generated

        Returns:
            StepResult indicating success/failure of recording
        """
        try:
            collection = self.mongodb_service.db[self.feedback_collection]

            # Generate embedding for the query (for similarity search later)
            query_embedding = await self.embedding_service.generate_embedding(
                context.original_query
            )

            # Prepare document summaries (don't store full content)
            doc_summaries = []
            for doc in context.graded_documents:
                doc_summaries.append({
                    "id": doc.id,
                    "parent_id": doc.parent_id,
                    "title": doc.title,
                    "score": doc.score,
                    "grading_score": doc.grading_score,
                    "source": doc.source,
                    "content_preview": doc.content[:200] if doc.content else "",
                })

            # Build feedback document
            feedback_doc = {
                "id": str(uuid.uuid4()),
                "query_id": context.query_id,
                "created_at": datetime.utcnow().isoformat(),

                # Query information
                "original_query": context.original_query,
                "rewritten_query": context.rewritten_query,
                "expanded_queries": context.expanded_queries,
                "query_type": context.query_type.value,
                "query_embedding": query_embedding,

                # Retrieval information
                "documents": doc_summaries,
                "retrieval_method": context.retrieval_method,
                "average_relevance": context.average_relevance,

                # Feedback
                "feedback_type": feedback_type,
                "feedback_text": feedback_text,
                "corrected_answer": corrected_answer,
                "answer": answer,

                # User context
                "user_id": user_id,
                "filters": context.filters,
            }

            await collection.insert_one(feedback_doc)

            # If positive feedback, also store in successful retrievals
            if feedback_type == FeedbackType.POSITIVE:
                await self._record_successful_retrieval(context, query_embedding)

            # If negative feedback, update document relevance scores
            if feedback_type == FeedbackType.NEGATIVE:
                await self._update_relevance_scores(
                    context.graded_documents,
                    boost=-0.05  # Small negative adjustment
                )

            self.logger.info(
                f"Recorded {feedback_type} feedback for query '{context.original_query[:50]}'"
            )

            return StepResult(
                success=True,
                metadata={
                    "feedback_id": feedback_doc["id"],
                    "feedback_type": feedback_type,
                    "documents_recorded": len(doc_summaries),
                }
            )

        except Exception as e:
            self.logger.exception("Failed to record feedback")
            return StepResult(
                success=False,
                errors=[f"Failed to record feedback: {str(e)}"],
            )

    async def _record_successful_retrieval(
        self,
        context: PipelineContext,
        query_embedding: List[float]
    ):
        """
        Store a successful query-retrieval pair for few-shot learning.

        Successful retrievals are stored separately for efficient lookup
        when boosting future queries.
        """
        collection = self.mongodb_service.db[self.successful_retrievals_collection]

        # Store document IDs that were useful
        useful_doc_ids = [
            doc.id for doc in context.graded_documents
            if doc.grading_score and doc.grading_score >= 0.7
        ]

        if not useful_doc_ids:
            return

        retrieval_doc = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "query": context.original_query,
            "query_embedding": query_embedding,
            "useful_document_ids": useful_doc_ids,
            "document_count": len(useful_doc_ids),
            "average_relevance": context.average_relevance,
        }

        await collection.insert_one(retrieval_doc)

    async def _update_relevance_scores(
        self,
        documents: List[RetrievedDocument],
        boost: float
    ):
        """
        Update document relevance feedback scores in the database.

        This method incrementally adjusts the stored relevance scores
        based on feedback. Documents that receive positive feedback
        get boosted; negative feedback results in a penalty.

        Args:
            documents: Documents to update
            boost: Score adjustment (positive or negative)
        """
        if not documents:
            return

        docs_collection = self.mongodb_service.db["documents"]

        for doc in documents:
            try:
                # Update relevance_feedback_score with exponential moving average
                await docs_collection.update_one(
                    {"_id": doc.id},
                    {
                        "$inc": {"feedback_count": 1},
                        # Use $max/$min to clamp the score
                        "$set": {
                            "last_feedback_at": datetime.utcnow().isoformat(),
                        }
                    }
                )

                # Calculate new score with clamping
                existing = await docs_collection.find_one(
                    {"_id": doc.id},
                    {"relevance_feedback_score": 1, "feedback_count": 1}
                )

                if existing:
                    current_score = existing.get("relevance_feedback_score", 0.5)
                    feedback_count = existing.get("feedback_count", 1)

                    # Exponential moving average with decay
                    alpha = 1.0 / (1 + feedback_count * 0.1)  # Decay with count
                    new_score = current_score * (1 - alpha) + (current_score + boost) * alpha
                    new_score = max(0.0, min(1.0, new_score))

                    await docs_collection.update_one(
                        {"_id": doc.id},
                        {"$set": {"relevance_feedback_score": new_score}}
                    )

            except Exception as e:
                self.logger.warning(f"Failed to update relevance for doc {doc.id}: {e}")

    async def get_similar_successful_queries(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find successful retrievals for queries similar to the input.

        This method is used to identify documents that worked well
        for similar queries, enabling relevance boosting.

        Args:
            query: The current query
            limit: Maximum similar queries to return

        Returns:
            List of successful retrieval records
        """
        try:
            query_embedding = await self.embedding_service.generate_embedding(query)

            collection = self.mongodb_service.db[self.successful_retrievals_collection]

            # Use vector search to find similar queries
            results = await self.mongodb_service._vector_search(
                collection_name=self.successful_retrievals_collection,
                query_vector=query_embedding,
                limit=limit,
                threshold=0.7  # Only high similarity
            )

            return results

        except Exception as e:
            self.logger.warning(f"Failed to find similar successful queries: {e}")
            return []

    async def get_feedback_statistics(
        self,
        days: int = 30,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.

        Useful for monitoring system health and identifying
        patterns in user satisfaction.

        Args:
            days: Number of days to analyze
            user_id: Optional user filter

        Returns:
            Dictionary with feedback statistics
        """
        try:
            collection = self.mongodb_service.db[self.feedback_collection]

            # Calculate date threshold
            threshold_date = datetime.utcnow()
            from datetime import timedelta
            threshold_date = threshold_date - timedelta(days=days)

            # Build query
            query = {
                "created_at": {"$gte": threshold_date.isoformat()}
            }
            if user_id:
                query["user_id"] = user_id

            # Aggregate statistics
            pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$feedback_type",
                    "count": {"$sum": 1},
                    "avg_relevance": {"$avg": "$average_relevance"},
                }},
            ]

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=100)

            # Build statistics dictionary
            stats = {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "correction": 0,
                "positive_rate": 0.0,
                "avg_relevance_positive": 0.0,
                "avg_relevance_negative": 0.0,
                "period_days": days,
            }

            for r in results:
                feedback_type = r["_id"]
                count = r["count"]
                stats["total"] += count

                if feedback_type == FeedbackType.POSITIVE:
                    stats["positive"] = count
                    stats["avg_relevance_positive"] = r["avg_relevance"] or 0
                elif feedback_type == FeedbackType.NEGATIVE:
                    stats["negative"] = count
                    stats["avg_relevance_negative"] = r["avg_relevance"] or 0
                elif feedback_type == FeedbackType.CORRECTION:
                    stats["correction"] = count

            if stats["total"] > 0:
                stats["positive_rate"] = stats["positive"] / stats["total"]

            return stats

        except Exception as e:
            self.logger.exception("Failed to get feedback statistics")
            return {"error": str(e)}

    async def get_document_feedback_summary(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Get feedback summary for a specific document.

        Useful for understanding how well a document performs
        across different queries.

        Args:
            document_id: The document ID to analyze

        Returns:
            Dictionary with document feedback summary
        """
        try:
            collection = self.mongodb_service.db[self.feedback_collection]

            # Find all feedback that includes this document
            pipeline = [
                {"$match": {"documents.id": document_id}},
                {"$group": {
                    "_id": "$feedback_type",
                    "count": {"$sum": 1},
                    "avg_grading_score": {
                        "$avg": {
                            "$arrayElemAt": [
                                {"$filter": {
                                    "input": "$documents",
                                    "cond": {"$eq": ["$$this.id", document_id]}
                                }},
                                0
                            ]
                        }
                    }
                }},
            ]

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=10)

            summary = {
                "document_id": document_id,
                "total_appearances": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
            }

            for r in results:
                count = r["count"]
                summary["total_appearances"] += count

                if r["_id"] == FeedbackType.POSITIVE:
                    summary["positive_feedback"] = count
                elif r["_id"] == FeedbackType.NEGATIVE:
                    summary["negative_feedback"] = count

            if summary["total_appearances"] > 0:
                summary["positive_rate"] = (
                    summary["positive_feedback"] / summary["total_appearances"]
                )
            else:
                summary["positive_rate"] = 0.0

            return summary

        except Exception as e:
            self.logger.exception(f"Failed to get document feedback: {document_id}")
            return {"error": str(e), "document_id": document_id}

    async def build_preference_dataset(
        self,
        min_samples: int = 50,
        output_format: str = "dpo"
    ) -> List[Dict]:
        """
        Build a preference dataset for RLHF training.

        Creates pairs of (query, chosen_response, rejected_response)
        from feedback data for training reward models.

        Args:
            min_samples: Minimum samples required
            output_format: Format for output ("dpo" or "rm")

        Returns:
            List of preference pairs
        """
        try:
            collection = self.mongodb_service.db[self.feedback_collection]

            # Find queries with both positive and negative feedback
            pipeline = [
                {"$group": {
                    "_id": "$original_query",
                    "feedback": {"$push": {
                        "type": "$feedback_type",
                        "answer": "$answer",
                        "documents": "$documents",
                        "average_relevance": "$average_relevance",
                    }},
                    "count": {"$sum": 1}
                }},
                {"$match": {"count": {"$gte": 2}}},
            ]

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=1000)

            preference_pairs = []

            for result in results:
                query = result["_id"]
                feedbacks = result["feedback"]

                # Separate positive and negative
                positive = [f for f in feedbacks if f["type"] == FeedbackType.POSITIVE]
                negative = [f for f in feedbacks if f["type"] == FeedbackType.NEGATIVE]

                if positive and negative:
                    # Create preference pair
                    for pos in positive:
                        for neg in negative:
                            if pos.get("answer") and neg.get("answer"):
                                pair = {
                                    "query": query,
                                    "chosen": pos["answer"],
                                    "rejected": neg["answer"],
                                    "chosen_relevance": pos.get("average_relevance", 0),
                                    "rejected_relevance": neg.get("average_relevance", 0),
                                }

                                if output_format == "rm":
                                    # Reward model format
                                    pair["chosen_score"] = 1.0
                                    pair["rejected_score"] = 0.0

                                preference_pairs.append(pair)

            self.logger.info(f"Built {len(preference_pairs)} preference pairs")
            return preference_pairs

        except Exception as e:
            self.logger.exception("Failed to build preference dataset")
            return []
