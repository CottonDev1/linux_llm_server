"""
Agent Learning API Routes
=========================

API endpoints for agent learning and feedback integration.
These endpoints support the ewr_sql_agent and ewr_code_agent
learning from query outcomes and user corrections.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from config import MONGODB_URI

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent-learning", tags=["Agent Learning"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SQLGenerationRequest(BaseModel):
    """Request for SQL generation with learning support."""
    natural_language: str = Field(..., description="Natural language query")
    database: str = Field(..., description="Target database name")
    server: str = Field(default="CHAD-PC", description="SQL Server hostname")
    schema_context: Optional[str] = Field(None, description="Schema context for generation")
    similar_examples: Optional[List[Dict[str, Any]]] = Field(None, description="Similar examples")
    max_retries: int = Field(default=3, description="Max retry attempts")


class SQLGenerationResponse(BaseModel):
    """Response from SQL generation."""
    success: bool
    sql: Optional[str] = None
    explanation: Optional[str] = None
    confidence: float = 0.0
    source: str = "llm"  # "llm", "cached", "learned_pattern"
    attempts: int = 1
    validation_passed: bool = False
    error: Optional[str] = None


class SQLValidationRequest(BaseModel):
    """Request for SQL validation by code agent."""
    sql: str = Field(..., description="SQL query to validate")
    database: str = Field(..., description="Target database name")
    natural_language: Optional[str] = Field(None, description="Original NL query")
    schema_info: Optional[Dict[str, Any]] = Field(None, description="Schema information")
    check_security: bool = Field(default=True, description="Check for security issues")
    check_syntax: bool = Field(default=True, description="Check SQL syntax")


class SQLValidationResponse(BaseModel):
    """Response from SQL validation."""
    valid: bool
    decision: str  # "approved" or "rejected"
    reason: str
    issues: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    security_score: float = 1.0
    validation_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request for recording feedback."""
    natural_language: str
    database: str
    original_sql: str
    corrected_sql: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    user_correction: bool = False
    correction_reason: Optional[str] = None
    validation_id: Optional[str] = None
    execution_time_ms: Optional[int] = None
    row_count: Optional[int] = None


class FeedbackResponse(BaseModel):
    """Response from feedback recording."""
    success: bool
    record_id: Optional[str] = None
    message: str


class LearningStatsResponse(BaseModel):
    """Learning statistics response."""
    sql_agent: Dict[str, Any]
    code_agent: Dict[str, Any]
    patterns_learned: int
    corrections_pending: int


# =============================================================================
# Agent Learning Service
# =============================================================================

class AgentLearningAPI:
    """
    API service for agent learning operations.
    """

    def __init__(self):
        self._learning_service = None
        self._initialized = False

    async def initialize(self):
        """Initialize the learning service."""
        if self._initialized:
            return

        try:
            from services.agent_learning_service import get_learning_service
            self._learning_service = await get_learning_service(
                mongodb_uri=MONGODB_URI,
                database_name="agent_learning"
            )
            self._initialized = True
            logger.info("AgentLearningAPI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgentLearningAPI: {e}")
            # Don't raise - allow service to start even if MongoDB is down
            self._initialized = False

    async def check_for_cached_sql(self, request: SQLGenerationRequest) -> SQLGenerationResponse:
        """
        Check for cached SQL before LLM generation.

        Returns cached result if found, otherwise indicates LLM generation needed.
        """
        if not self._initialized:
            await self.initialize()

        if not self._learning_service:
            return SQLGenerationResponse(
                success=True,
                sql=None,
                source="llm",
                explanation="Learning service not available, proceeding with LLM"
            )

        try:
            # Check for exact learned pattern
            pattern = await self._learning_service.find_matching_pattern(
                input_text=request.natural_language,
                pattern_type="sql_generation",
                database=request.database,
                min_confidence=0.9
            )

            if pattern:
                logger.info(f"Found learned pattern for: {request.natural_language[:50]}...")
                return SQLGenerationResponse(
                    success=True,
                    sql=pattern["output_pattern"],
                    explanation="Using learned pattern from previous successful queries",
                    confidence=pattern.get("confidence", 0.9),
                    source="learned_pattern",
                    validation_passed=True
                )

            # Check for similar successful queries
            similar = await self._learning_service.find_similar_successful_query(
                natural_language=request.natural_language,
                database=request.database,
                limit=3
            )

            if similar:
                best = similar[0]
                nl_hash = self._learning_service._hash_text(request.natural_language)
                if best.get("nl_hash") == nl_hash:
                    logger.info(f"Found exact cached query")
                    return SQLGenerationResponse(
                        success=True,
                        sql=best["sql"],
                        explanation="Using cached successful query",
                        confidence=best.get("confidence", 0.9),
                        source="cached",
                        validation_passed=True
                    )

            # No cached result, LLM generation needed
            return SQLGenerationResponse(
                success=True,
                sql=None,
                source="llm",
                explanation="No cached result, LLM generation required"
            )

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return SQLGenerationResponse(
                success=True,
                sql=None,
                source="llm",
                explanation=f"Cache check failed: {str(e)}"
            )

    async def validate_sql(self, request: SQLValidationRequest) -> SQLValidationResponse:
        """
        Validate SQL using code agent patterns.
        """
        if not self._initialized:
            await self.initialize()

        issues = []
        suggestions = []
        security_score = 1.0

        # Check for dangerous patterns
        dangerous_patterns = [
            ("DROP TABLE", "Attempt to drop table", "critical"),
            ("DROP DATABASE", "Attempt to drop database", "critical"),
            ("TRUNCATE", "Attempt to truncate table", "critical"),
            ("xp_cmdshell", "Attempt to execute shell command", "critical"),
            ("sp_executesql", "Dynamic SQL execution", "warning"),
            ("EXEC(", "Dynamic execution", "warning"),
        ]

        sql_upper = request.sql.upper()
        for pattern, description, severity in dangerous_patterns:
            if pattern in sql_upper:
                issues.append({
                    "type": "security",
                    "severity": severity,
                    "message": description
                })
                if severity == "critical":
                    security_score = 0.0
                else:
                    security_score -= 0.2

        # Check for DELETE/UPDATE without WHERE
        if "DELETE FROM" in sql_upper and "WHERE" not in sql_upper:
            issues.append({
                "type": "security",
                "severity": "high",
                "message": "DELETE without WHERE clause"
            })
            security_score -= 0.3

        if "UPDATE " in sql_upper and " SET " in sql_upper and "WHERE" not in sql_upper:
            issues.append({
                "type": "security",
                "severity": "high",
                "message": "UPDATE without WHERE clause"
            })
            security_score -= 0.3

        # Check for known failure patterns if learning service available
        if self._learning_service:
            try:
                failures = await self._learning_service.find_similar_failures(
                    sql=request.sql,
                    database=request.database,
                    limit=3
                )
                if failures:
                    for failure in failures:
                        if failure.get("error_type") in ["syntax_error", "invalid_column", "invalid_table"]:
                            issues.append({
                                "type": "known_failure",
                                "severity": "warning",
                                "message": f"Similar to known failure: {failure.get('error_type')}"
                            })
            except Exception as e:
                logger.warning(f"Could not check failure patterns: {e}")

        # Determine decision
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        high_issues = [i for i in issues if i.get("severity") == "high"]

        if critical_issues:
            decision = "rejected"
            reason = f"Critical security issue: {critical_issues[0]['message']}"
        elif len(high_issues) >= 2:
            decision = "rejected"
            reason = "Multiple high-severity issues found"
        elif security_score < 0.5:
            decision = "rejected"
            reason = "Low security score"
        else:
            decision = "approved"
            reason = "Query passed validation checks"

        # Record validation if learning service available
        validation_id = None
        if self._learning_service:
            try:
                validation_id = await self._learning_service.record_validation_result(
                    sql=request.sql,
                    database=request.database,
                    agent_decision=decision,
                    agent_reason=reason,
                    validation_details={"issues": issues, "security_score": security_score}
                )
            except Exception as e:
                logger.warning(f"Could not record validation: {e}")

        return SQLValidationResponse(
            valid=decision == "approved",
            decision=decision,
            reason=reason,
            issues=issues,
            suggestions=suggestions,
            security_score=max(0, security_score),
            validation_id=validation_id
        )

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """
        Record feedback for agent learning.
        """
        if not self._initialized:
            await self.initialize()

        if not self._learning_service:
            return FeedbackResponse(
                success=False,
                message="Learning service not available"
            )

        try:
            if request.user_correction and request.corrected_sql:
                # Record user correction
                record_id = await self._learning_service.record_correction(
                    natural_language=request.natural_language,
                    database=request.database,
                    original_sql=request.original_sql,
                    corrected_sql=request.corrected_sql,
                    correction_reason=request.correction_reason
                )
                return FeedbackResponse(
                    success=True,
                    record_id=record_id,
                    message="Correction recorded for learning"
                )
            else:
                # Record query outcome
                record_id = await self._learning_service.record_query_attempt(
                    agent_id="ewr_sql_agent",
                    natural_language=request.natural_language,
                    database=request.database,
                    generated_sql=request.original_sql,
                    success=request.success,
                    error_message=request.error_message,
                    execution_time_ms=request.execution_time_ms,
                    row_count=request.row_count
                )

                # Update validation outcome if provided
                if request.validation_id:
                    await self._learning_service.update_validation_outcome(
                        validation_id=request.validation_id,
                        actual_outcome="success" if request.success else "failure",
                        error_message=request.error_message
                    )

                return FeedbackResponse(
                    success=True,
                    record_id=record_id,
                    message="Feedback recorded"
                )

        except Exception as e:
            logger.error(f"Feedback recording error: {e}")
            return FeedbackResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    async def get_stats(self) -> LearningStatsResponse:
        """Get learning statistics."""
        if not self._initialized:
            await self.initialize()

        if not self._learning_service:
            return LearningStatsResponse(
                sql_agent={"status": "unavailable"},
                code_agent={"status": "unavailable"},
                patterns_learned=0,
                corrections_pending=0
            )

        try:
            code_accuracy = await self._learning_service.get_validation_accuracy(days=30)
            patterns_count = await self._learning_service._db["agent_learned_patterns"].count_documents(
                {"active": True}
            )
            corrections_pending = await self._learning_service._db["agent_corrections"].count_documents(
                {"verified": False}
            )

            return LearningStatsResponse(
                sql_agent={
                    "status": "active",
                    "queries_processed": await self._learning_service._db["agent_query_history"].count_documents({})
                },
                code_agent={
                    "status": "active",
                    "accuracy": code_accuracy.get("accuracy", 0),
                    "false_positives": code_accuracy.get("false_positives", 0),
                    "false_negatives": code_accuracy.get("false_negatives", 0)
                },
                patterns_learned=patterns_count,
                corrections_pending=corrections_pending
            )

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Global instance
_api_service: Optional[AgentLearningAPI] = None


async def get_api_service() -> AgentLearningAPI:
    """Get or create the API service."""
    global _api_service
    if _api_service is None:
        _api_service = AgentLearningAPI()
        await _api_service.initialize()
    return _api_service


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/sql/check-cache", response_model=SQLGenerationResponse)
async def check_sql_cache(request: SQLGenerationRequest):
    """
    Check for cached SQL before LLM generation.

    Returns cached result if available, otherwise indicates LLM needed.
    """
    service = await get_api_service()
    return await service.check_for_cached_sql(request)


@router.post("/sql/validate", response_model=SQLValidationResponse)
async def validate_sql(request: SQLValidationRequest):
    """
    Validate SQL query using code agent patterns.

    Checks for security issues, known failures, and dangerous patterns.
    """
    service = await get_api_service()
    return await service.validate_sql(request)


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """
    Record feedback for agent learning.

    Called after query execution to record outcomes and corrections.
    """
    service = await get_api_service()
    return await service.record_feedback(request)


@router.get("/stats", response_model=LearningStatsResponse)
async def get_learning_stats():
    """Get learning statistics for both agents."""
    service = await get_api_service()
    return await service.get_stats()


@router.post("/correction/verify")
async def verify_correction(correction_id: str, success: bool):
    """
    Verify that a user correction worked.

    Called after executing corrected SQL to confirm it works.
    """
    service = await get_api_service()
    if service._learning_service:
        await service._learning_service.verify_correction(correction_id, success)
        return {"success": True, "message": "Correction verified"}
    return {"success": False, "message": "Learning service not available"}


@router.get("/health")
async def learning_health():
    """Check learning service health."""
    try:
        service = await get_api_service()
        return {
            "status": "healthy" if service._initialized else "degraded",
            "learning_service_available": service._learning_service is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
