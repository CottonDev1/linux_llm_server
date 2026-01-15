"""
Agent Learning Service
=======================

MongoDB-based service for storing and retrieving agent learning data.
Supports both ewr_sql_agent and ewr_code_agent with feedback loops
for continuous improvement.

Collections:
- agent_query_history: All queries processed by agents
- agent_successes: Successfully executed queries
- agent_failures: Failed queries with error information
- agent_corrections: User-provided corrections
- agent_validation_results: Code agent validation decisions
- agent_learned_patterns: Patterns learned from feedback
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson import ObjectId
import hashlib

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MONGODB_URI as CONFIG_MONGODB_URI

logger = logging.getLogger(__name__)

# Collection names
COLLECTION_QUERY_HISTORY = "agent_query_history"
COLLECTION_SUCCESSES = "agent_successes"
COLLECTION_FAILURES = "agent_failures"
COLLECTION_CORRECTIONS = "agent_corrections"
COLLECTION_VALIDATION_RESULTS = "agent_validation_results"
COLLECTION_LEARNED_PATTERNS = "agent_learned_patterns"


class AgentLearningService:
    """
    Service for managing agent learning data in MongoDB.

    Provides methods for:
    - Recording query attempts and outcomes
    - Storing and retrieving successful patterns
    - Learning from user corrections
    - Tracking validation decisions and their accuracy
    """

    def __init__(self, mongodb_uri: str = None, database_name: str = "agent_learning"):
        self.mongodb_uri = mongodb_uri or CONFIG_MONGODB_URI
        self.database_name = database_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._initialized = False

    async def initialize(self):
        """Initialize MongoDB connection and create indexes."""
        if self._initialized:
            return

        try:
            self._client = AsyncIOMotorClient(self.mongodb_uri)
            self._db = self._client[self.database_name]

            # Create indexes for efficient queries
            await self._create_indexes()

            self._initialized = True
            logger.info(f"AgentLearningService initialized: {self.database_name}")

        except Exception as e:
            logger.error(f"Failed to initialize AgentLearningService: {e}")
            raise

    async def _create_indexes(self):
        """Create indexes for efficient queries."""
        # Query history indexes
        await self._db[COLLECTION_QUERY_HISTORY].create_index([
            ("agent_id", 1), ("timestamp", -1)
        ])
        await self._db[COLLECTION_QUERY_HISTORY].create_index([
            ("natural_language_hash", 1)
        ])
        await self._db[COLLECTION_QUERY_HISTORY].create_index([
            ("database", 1), ("success", 1)
        ])

        # Successes indexes
        await self._db[COLLECTION_SUCCESSES].create_index([
            ("database", 1), ("nl_hash", 1)
        ])
        await self._db[COLLECTION_SUCCESSES].create_index([
            ("confidence", -1)
        ])

        # Failures indexes
        await self._db[COLLECTION_FAILURES].create_index([
            ("database", 1), ("error_type", 1)
        ])
        await self._db[COLLECTION_FAILURES].create_index([
            ("nl_hash", 1), ("timestamp", -1)
        ])

        # Corrections indexes
        await self._db[COLLECTION_CORRECTIONS].create_index([
            ("nl_hash", 1)
        ])
        await self._db[COLLECTION_CORRECTIONS].create_index([
            ("database", 1), ("verified", 1)
        ])

        # Validation results indexes
        await self._db[COLLECTION_VALIDATION_RESULTS].create_index([
            ("sql_hash", 1), ("outcome", 1)
        ])
        await self._db[COLLECTION_VALIDATION_RESULTS].create_index([
            ("agent_decision", 1), ("actual_outcome", 1)
        ])

        # Learned patterns indexes
        await self._db[COLLECTION_LEARNED_PATTERNS].create_index([
            ("pattern_type", 1), ("confidence", -1)
        ])

    def _hash_text(self, text: str) -> str:
        """Generate a hash for text comparison."""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._initialized = False

    # ==========================================================================
    # Query History Methods
    # ==========================================================================

    async def record_query_attempt(
        self,
        agent_id: str,
        natural_language: str,
        database: str,
        generated_sql: str,
        success: bool,
        execution_time_ms: int = None,
        error_message: str = None,
        row_count: int = None,
        validation_passed: bool = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Record a query attempt for learning purposes.

        Args:
            agent_id: Identifier of the agent (ewr_sql_agent, ewr_code_agent)
            natural_language: Original natural language query
            database: Target database
            generated_sql: SQL generated by the agent
            success: Whether the query executed successfully
            execution_time_ms: Query execution time in milliseconds
            error_message: Error message if failed
            row_count: Number of rows returned
            validation_passed: Whether validation (code agent) approved
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc = {
            "agent_id": agent_id,
            "natural_language": natural_language,
            "natural_language_hash": self._hash_text(natural_language),
            "database": database,
            "generated_sql": generated_sql,
            "sql_hash": self._hash_text(generated_sql),
            "success": success,
            "execution_time_ms": execution_time_ms,
            "error_message": error_message,
            "row_count": row_count,
            "validation_passed": validation_passed,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc),
            "learned_from": False
        }

        result = await self._db[COLLECTION_QUERY_HISTORY].insert_one(doc)

        # Also record to success/failure collections for quick lookup
        if success:
            await self._record_success(doc)
        else:
            await self._record_failure(doc)

        return str(result.inserted_id)

    async def _record_success(self, query_doc: Dict[str, Any]):
        """Record a successful query for pattern learning."""
        success_doc = {
            "natural_language": query_doc["natural_language"],
            "nl_hash": query_doc["natural_language_hash"],
            "database": query_doc["database"],
            "sql": query_doc["generated_sql"],
            "sql_hash": query_doc["sql_hash"],
            "execution_time_ms": query_doc.get("execution_time_ms"),
            "row_count": query_doc.get("row_count"),
            "agent_id": query_doc["agent_id"],
            "confidence": 0.8,  # Initial confidence
            "use_count": 1,
            "timestamp": query_doc["timestamp"],
            "last_used": query_doc["timestamp"]
        }

        # Upsert: update existing or insert new
        await self._db[COLLECTION_SUCCESSES].update_one(
            {"nl_hash": query_doc["natural_language_hash"], "database": query_doc["database"]},
            {
                "$set": success_doc,
                "$inc": {"use_count": 1}
            },
            upsert=True
        )

    async def _record_failure(self, query_doc: Dict[str, Any]):
        """Record a failed query for error pattern learning."""
        # Categorize error type
        error_msg = query_doc.get("error_message", "")
        error_type = self._categorize_error(error_msg)

        failure_doc = {
            "natural_language": query_doc["natural_language"],
            "nl_hash": query_doc["natural_language_hash"],
            "database": query_doc["database"],
            "sql": query_doc["generated_sql"],
            "sql_hash": query_doc["sql_hash"],
            "error_message": error_msg,
            "error_type": error_type,
            "agent_id": query_doc["agent_id"],
            "timestamp": query_doc["timestamp"],
            "corrected": False,
            "correction_id": None
        }

        await self._db[COLLECTION_FAILURES].insert_one(failure_doc)

    def _categorize_error(self, error_message: str) -> str:
        """Categorize SQL error for pattern learning."""
        error_lower = error_message.lower()

        if "invalid column" in error_lower or "invalid column name" in error_lower:
            return "invalid_column"
        elif "invalid object" in error_lower or "table" in error_lower and "not exist" in error_lower:
            return "invalid_table"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "conversion" in error_lower or "convert" in error_lower:
            return "type_conversion"
        elif "timeout" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "denied" in error_lower:
            return "permission_denied"
        elif "ambiguous" in error_lower:
            return "ambiguous_column"
        else:
            return "other"

    # ==========================================================================
    # Correction Methods
    # ==========================================================================

    async def record_correction(
        self,
        natural_language: str,
        database: str,
        original_sql: str,
        corrected_sql: str,
        correction_reason: str = None,
        user_id: str = None,
        failure_id: str = None
    ) -> str:
        """
        Record a user-provided correction for learning.

        Args:
            natural_language: Original natural language query
            database: Target database
            original_sql: SQL that was originally generated (wrong)
            corrected_sql: User-provided correct SQL
            correction_reason: Why the correction was needed
            user_id: ID of user who provided correction
            failure_id: ID of the failure record being corrected

        Returns:
            Correction document ID
        """
        doc = {
            "natural_language": natural_language,
            "nl_hash": self._hash_text(natural_language),
            "database": database,
            "original_sql": original_sql,
            "original_sql_hash": self._hash_text(original_sql),
            "corrected_sql": corrected_sql,
            "corrected_sql_hash": self._hash_text(corrected_sql),
            "correction_reason": correction_reason,
            "user_id": user_id,
            "failure_id": failure_id,
            "timestamp": datetime.now(timezone.utc),
            "verified": False,  # Set to True after SQL executes successfully
            "learned_from": False
        }

        result = await self._db[COLLECTION_CORRECTIONS].insert_one(doc)

        # Mark the failure as corrected
        if failure_id:
            await self._db[COLLECTION_FAILURES].update_one(
                {"_id": ObjectId(failure_id)},
                {"$set": {"corrected": True, "correction_id": str(result.inserted_id)}}
            )

        logger.info(f"Correction recorded for '{natural_language[:50]}...'")
        return str(result.inserted_id)

    async def verify_correction(self, correction_id: str, success: bool):
        """
        Mark a correction as verified (SQL executed successfully).

        Args:
            correction_id: ID of the correction document
            success: Whether the corrected SQL executed successfully
        """
        await self._db[COLLECTION_CORRECTIONS].update_one(
            {"_id": ObjectId(correction_id)},
            {"$set": {
                "verified": success,
                "verified_at": datetime.now(timezone.utc)
            }}
        )

        # If verified, add to successes for future use
        if success:
            correction = await self._db[COLLECTION_CORRECTIONS].find_one(
                {"_id": ObjectId(correction_id)}
            )
            if correction:
                await self._db[COLLECTION_SUCCESSES].update_one(
                    {"nl_hash": correction["nl_hash"], "database": correction["database"]},
                    {
                        "$set": {
                            "natural_language": correction["natural_language"],
                            "sql": correction["corrected_sql"],
                            "sql_hash": correction["corrected_sql_hash"],
                            "confidence": 1.0,  # User-verified = highest confidence
                            "source": "user_correction",
                            "timestamp": datetime.now(timezone.utc)
                        }
                    },
                    upsert=True
                )

    async def get_corrections_for_learning(
        self,
        database: str = None,
        verified_only: bool = True,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get corrections that can be used for learning.

        Args:
            database: Filter by database
            verified_only: Only return verified corrections
            limit: Maximum number of corrections

        Returns:
            List of correction documents
        """
        query = {"learned_from": False}
        if database:
            query["database"] = database
        if verified_only:
            query["verified"] = True

        cursor = self._db[COLLECTION_CORRECTIONS].find(query).limit(limit)
        return await cursor.to_list(length=limit)

    # ==========================================================================
    # Validation Results Methods (for Code Agent)
    # ==========================================================================

    async def record_validation_result(
        self,
        sql: str,
        database: str,
        agent_decision: str,  # "approved" or "rejected"
        agent_reason: str,
        actual_outcome: str = None,  # "success" or "failure" after execution
        validation_details: Dict[str, Any] = None
    ) -> str:
        """
        Record a validation decision and its outcome.

        Used to track code agent accuracy and learn from mistakes.

        Args:
            sql: The SQL that was validated
            database: Target database
            agent_decision: What the agent decided (approved/rejected)
            agent_reason: Why the agent made that decision
            actual_outcome: What actually happened after execution
            validation_details: Additional validation info

        Returns:
            Document ID
        """
        doc = {
            "sql": sql,
            "sql_hash": self._hash_text(sql),
            "database": database,
            "agent_decision": agent_decision,
            "agent_reason": agent_reason,
            "actual_outcome": actual_outcome,
            "validation_details": validation_details or {},
            "timestamp": datetime.now(timezone.utc),
            "outcome_recorded": actual_outcome is not None
        }

        # Determine if this was a correct decision
        if actual_outcome:
            doc["correct_decision"] = (
                (agent_decision == "approved" and actual_outcome == "success") or
                (agent_decision == "rejected" and actual_outcome == "failure")
            )

        result = await self._db[COLLECTION_VALIDATION_RESULTS].insert_one(doc)
        return str(result.inserted_id)

    async def update_validation_outcome(
        self,
        validation_id: str,
        actual_outcome: str,
        error_message: str = None
    ):
        """
        Update a validation record with the actual execution outcome.

        Args:
            validation_id: ID of the validation record
            actual_outcome: "success" or "failure"
            error_message: Error message if failed
        """
        # Get the original decision
        doc = await self._db[COLLECTION_VALIDATION_RESULTS].find_one(
            {"_id": ObjectId(validation_id)}
        )

        if not doc:
            return

        agent_decision = doc.get("agent_decision")
        correct_decision = (
            (agent_decision == "approved" and actual_outcome == "success") or
            (agent_decision == "rejected" and actual_outcome == "failure")
        )

        await self._db[COLLECTION_VALIDATION_RESULTS].update_one(
            {"_id": ObjectId(validation_id)},
            {"$set": {
                "actual_outcome": actual_outcome,
                "error_message": error_message,
                "outcome_recorded": True,
                "correct_decision": correct_decision,
                "outcome_timestamp": datetime.now(timezone.utc)
            }}
        )

        # Log the outcome for monitoring
        decision_type = "correct" if correct_decision else "incorrect"
        logger.info(f"Code agent made {decision_type} decision: {agent_decision} -> {actual_outcome}")

    async def get_validation_accuracy(
        self,
        database: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get validation accuracy statistics for the code agent.

        Returns:
            Dictionary with accuracy metrics
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        query = {"outcome_recorded": True, "timestamp": {"$gte": cutoff}}
        if database:
            query["database"] = database

        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": None,
                "total": {"$sum": 1},
                "correct": {"$sum": {"$cond": ["$correct_decision", 1, 0]}},
                "false_positives": {"$sum": {
                    "$cond": [
                        {"$and": [
                            {"$eq": ["$agent_decision", "approved"]},
                            {"$eq": ["$actual_outcome", "failure"]}
                        ]},
                        1, 0
                    ]
                }},
                "false_negatives": {"$sum": {
                    "$cond": [
                        {"$and": [
                            {"$eq": ["$agent_decision", "rejected"]},
                            {"$eq": ["$actual_outcome", "success"]}
                        ]},
                        1, 0
                    ]
                }}
            }}
        ]

        result = await self._db[COLLECTION_VALIDATION_RESULTS].aggregate(pipeline).to_list(1)

        if not result:
            return {"total": 0, "accuracy": 0, "false_positive_rate": 0, "false_negative_rate": 0}

        stats = result[0]
        total = stats["total"]

        return {
            "total": total,
            "correct": stats["correct"],
            "accuracy": stats["correct"] / total if total > 0 else 0,
            "false_positives": stats["false_positives"],
            "false_negatives": stats["false_negatives"],
            "false_positive_rate": stats["false_positives"] / total if total > 0 else 0,
            "false_negative_rate": stats["false_negatives"] / total if total > 0 else 0
        }

    # ==========================================================================
    # Learned Patterns Methods
    # ==========================================================================

    async def add_learned_pattern(
        self,
        pattern_type: str,  # "sql_generation", "validation", "error_fix"
        input_pattern: str,
        output_pattern: str,
        database: str = None,
        confidence: float = 0.8,
        source: str = "auto",
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add a learned pattern for future use.

        Args:
            pattern_type: Type of pattern
            input_pattern: Input that triggers this pattern
            output_pattern: Expected output
            database: Database this pattern applies to
            confidence: Confidence score (0-1)
            source: Source of pattern (auto, user_correction, etc.)
            metadata: Additional metadata

        Returns:
            Pattern document ID
        """
        doc = {
            "pattern_type": pattern_type,
            "input_pattern": input_pattern,
            "input_hash": self._hash_text(input_pattern),
            "output_pattern": output_pattern,
            "database": database,
            "confidence": confidence,
            "source": source,
            "metadata": metadata or {},
            "use_count": 0,
            "success_count": 0,
            "created_at": datetime.now(timezone.utc),
            "last_used": None,
            "active": True
        }

        result = await self._db[COLLECTION_LEARNED_PATTERNS].insert_one(doc)
        return str(result.inserted_id)

    async def find_matching_pattern(
        self,
        input_text: str,
        pattern_type: str,
        database: str = None,
        min_confidence: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Find a learned pattern matching the input.

        Args:
            input_text: Input to match
            pattern_type: Type of pattern to find
            database: Database to match
            min_confidence: Minimum confidence threshold

        Returns:
            Matching pattern or None
        """
        query = {
            "pattern_type": pattern_type,
            "input_hash": self._hash_text(input_text),
            "confidence": {"$gte": min_confidence},
            "active": True
        }

        if database:
            query["$or"] = [
                {"database": database},
                {"database": None}
            ]

        pattern = await self._db[COLLECTION_LEARNED_PATTERNS].find_one(
            query,
            sort=[("confidence", -1)]
        )

        if pattern:
            # Update use count
            await self._db[COLLECTION_LEARNED_PATTERNS].update_one(
                {"_id": pattern["_id"]},
                {
                    "$inc": {"use_count": 1},
                    "$set": {"last_used": datetime.now(timezone.utc)}
                }
            )

        return pattern

    async def update_pattern_success(self, pattern_id: str, success: bool):
        """Update a pattern's success count based on outcome."""
        update = {"$inc": {"use_count": 1}}
        if success:
            update["$inc"]["success_count"] = 1

        await self._db[COLLECTION_LEARNED_PATTERNS].update_one(
            {"_id": ObjectId(pattern_id)},
            update
        )

        # Recalculate confidence
        pattern = await self._db[COLLECTION_LEARNED_PATTERNS].find_one(
            {"_id": ObjectId(pattern_id)}
        )
        if pattern and pattern["use_count"] >= 5:
            new_confidence = pattern["success_count"] / pattern["use_count"]
            await self._db[COLLECTION_LEARNED_PATTERNS].update_one(
                {"_id": ObjectId(pattern_id)},
                {"$set": {"confidence": new_confidence}}
            )

            # Deactivate low-confidence patterns
            if new_confidence < 0.3:
                await self._db[COLLECTION_LEARNED_PATTERNS].update_one(
                    {"_id": ObjectId(pattern_id)},
                    {"$set": {"active": False}}
                )

    # ==========================================================================
    # Search Methods for RAG Enhancement
    # ==========================================================================

    async def find_similar_successful_query(
        self,
        natural_language: str,
        database: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find successful queries similar to the input.

        This is used by the SQL agent to find examples for few-shot learning.

        Args:
            natural_language: Input query
            database: Target database
            limit: Maximum results

        Returns:
            List of similar successful queries with their SQL
        """
        # First try exact hash match
        exact = await self._db[COLLECTION_SUCCESSES].find_one({
            "nl_hash": self._hash_text(natural_language),
            "database": database
        })

        if exact:
            return [exact]

        # Otherwise, use text search (requires text index)
        # For now, return recent high-confidence successes
        cursor = self._db[COLLECTION_SUCCESSES].find(
            {"database": database, "confidence": {"$gte": 0.7}}
        ).sort([("confidence", -1), ("use_count", -1)]).limit(limit)

        return await cursor.to_list(length=limit)

    async def find_similar_failures(
        self,
        sql: str,
        database: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar failed queries to avoid same mistakes.

        Args:
            sql: SQL to check
            database: Target database
            limit: Maximum results

        Returns:
            List of similar failed queries with error info
        """
        cursor = self._db[COLLECTION_FAILURES].find({
            "database": database,
            "corrected": False  # Uncorrected failures are patterns to avoid
        }).sort("timestamp", -1).limit(limit)

        return await cursor.to_list(length=limit)


# Global instance
_learning_service: Optional[AgentLearningService] = None


async def get_learning_service(
    mongodb_uri: str = None,
    database_name: str = "agent_learning"
) -> AgentLearningService:
    """Get or create the global learning service instance."""
    global _learning_service

    if _learning_service is None:
        _learning_service = AgentLearningService(mongodb_uri, database_name)
        await _learning_service.initialize()

    return _learning_service
