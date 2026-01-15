"""
Feedback routes for collecting user feedback on RAG responses.

Provides endpoints for:
- Storing user feedback (ratings, corrections, refinements)
- Retrieving feedback statistics
- Identifying low-performing documents
"""
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Request, Query

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import (
    FeedbackCreate, FeedbackResponse, FeedbackStatsResponse, LowPerformingDocument
)
from mongodb import get_mongodb_service
from log_service import log_pipeline

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/feedback", tags=["Feedback"])


# ============================================================================
# Feedback Routes
# ============================================================================

@router.post("", response_model=FeedbackResponse)
async def store_feedback(feedback: FeedbackCreate, request: Request):
    """
    Store user feedback on a RAG system response.

    Supports multiple feedback types:
    - **rating**: Thumbs up/down or star rating with optional comment
    - **correction**: User provides the correct answer (used for learning)
    - **refinement**: User clarifies their original query
    - **resolution**: User marks the query as resolved

    Feedback automatically updates quality scores for referenced documents,
    helping identify content that needs improvement.

    Example request body for rating:
    ```json
    {
        "feedback_type": "rating",
        "query": "Show me all orders from last month",
        "response": "SELECT * FROM Orders WHERE OrderDate >= DATEADD(month, -1, GETDATE())",
        "database": "CentralData",
        "rating": {
            "is_helpful": true,
            "rating": 5,
            "comment": "Perfect query!"
        },
        "document_ids": ["schema_centraldata_dbo.Orders"]
    }
    ```

    Example request body for correction:
    ```json
    {
        "feedback_type": "correction",
        "query": "Show me all orders from last month",
        "response": "SELECT * FROM Orders WHERE OrderDate >= DATEADD(month, -1, GETDATE())",
        "database": "CentralData",
        "correction": {
            "original_response": "SELECT * FROM Orders WHERE OrderDate >= DATEADD(month, -1, GETDATE())",
            "corrected_response": "SELECT * FROM dbo.Orders WHERE OrderDate >= DATEADD(MONTH, -1, CAST(GETDATE() AS DATE))",
            "error_type": "date_handling",
            "affected_tables": ["dbo.Orders"],
            "comment": "Should cast to DATE for proper date comparison"
        }
    }
    ```
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("FEEDBACK", user_ip, f"Storing {feedback.feedback_type.value} feedback",
                 feedback.query[:100] if feedback.query else "",
                 details={
                     "feedback_type": feedback.feedback_type.value,
                     "database": feedback.database,
                     "has_rating": feedback.rating is not None,
                     "has_correction": feedback.correction is not None,
                     "document_count": len(feedback.document_ids)
                 })

    mongodb = get_mongodb_service()

    # Convert Pydantic models to dicts for storage
    rating_dict = feedback.rating.model_dump() if feedback.rating else None
    correction_dict = feedback.correction.model_dump() if feedback.correction else None
    refinement_dict = feedback.refinement.model_dump() if feedback.refinement else None

    result = await mongodb.store_feedback(
        feedback_type=feedback.feedback_type.value,
        query=feedback.query,
        response=feedback.response,
        query_id=feedback.query_id,
        database=feedback.database,
        rating=rating_dict,
        correction=correction_dict,
        refinement=refinement_dict,
        document_ids=feedback.document_ids,
        user_id=feedback.user_id,
        session_id=feedback.session_id,
        metadata=feedback.metadata
    )

    log_pipeline("FEEDBACK", user_ip, "Feedback stored",
                 details={
                     "feedback_id": result.get("feedback_id"),
                     "quality_scores_updated": result.get("quality_scores_updated", 0)
                 })

    return FeedbackResponse(**result)


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    request: Request,
    database: Optional[str] = Query(None, description="Filter by database"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format: YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format: YYYY-MM-DD)")
):
    """
    Get aggregated feedback statistics.

    Returns comprehensive statistics about user feedback including:
    - Total feedback count and breakdown by type
    - Helpfulness rate (percentage of positive ratings)
    - Average star rating (if used)
    - Corrections by error type
    - Per-database statistics
    - Recent activity (last 24h, 7d)

    Use these statistics to:
    - Monitor RAG system quality over time
    - Identify databases with poor performance
    - Track improvement after content updates
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Parse dates if provided
    parsed_start = datetime.fromisoformat(start_date) if start_date else None
    parsed_end = datetime.fromisoformat(end_date) if end_date else None

    log_pipeline("FEEDBACK", user_ip, "Fetching feedback stats",
                 details={
                     "database": database,
                     "start_date": start_date,
                     "end_date": end_date
                 })

    mongodb = get_mongodb_service()
    stats = await mongodb.get_feedback_stats(
        database=database,
        start_date=parsed_start,
        end_date=parsed_end
    )

    log_pipeline("FEEDBACK", user_ip, "Feedback stats retrieved",
                 details={
                     "total_feedback": stats.get("total_feedback", 0),
                     "helpfulness_rate": stats.get("helpfulness_rate", 0)
                 })

    return FeedbackStatsResponse(**stats)


@router.get("/low-performing")
async def get_low_performing_documents(
    request: Request,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Quality score threshold (documents below this are returned)"),
    min_feedback: int = Query(3, ge=1, description="Minimum feedback count to be included"),
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of documents to return")
) -> List[LowPerformingDocument]:
    """
    Get documents with low quality scores based on feedback.

    Returns documents that:
    - Have quality_score below the threshold
    - Have received at least min_feedback feedback entries

    These documents should be prioritized for review and improvement.

    Quality scores are automatically updated when users provide feedback:
    - Positive ratings increase scores slightly
    - Negative ratings decrease scores more significantly
    - Corrections have the largest negative impact

    Use this endpoint to:
    - Identify content that consistently produces poor results
    - Prioritize schema/procedure documentation improvements
    - Track whether improvements are working
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("FEEDBACK", user_ip, "Fetching low-performing documents",
                 details={
                     "threshold": threshold,
                     "min_feedback": min_feedback,
                     "database": database,
                     "limit": limit
                 })

    mongodb = get_mongodb_service()
    documents = await mongodb.get_low_performing_documents(
        threshold=threshold,
        min_feedback_count=min_feedback,
        database=database,
        limit=limit
    )

    log_pipeline("FEEDBACK", user_ip, "Low-performing documents retrieved",
                 details={"count": len(documents)})

    return [LowPerformingDocument(**doc) for doc in documents]
