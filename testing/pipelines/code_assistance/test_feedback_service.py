"""
Feedback Service Tests
======================

Test feedback handling for the code assistance pipeline including:
- Feedback submission with CodeFeedbackRequest
- Rating feedback
- Correction feedback with error_category
- expected_methods handling
- Feedback storage and retrieval
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from utils import generate_test_id, assert_document_stored


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_response_id() -> str:
    """Generate a sample response ID."""
    return f"resp_{generate_test_id()}"


@pytest.fixture
def positive_feedback_request(sample_response_id):
    """Create a positive feedback request."""
    from code_assistance_pipeline.models.query_models import CodeFeedbackRequest

    return CodeFeedbackRequest(
        response_id=sample_response_id,
        rating=5,
        feedback="Very helpful explanation of the code!",
        was_helpful=True,
        is_helpful=True,
        comment="Very helpful explanation of the code!",
    )


@pytest.fixture
def negative_feedback_request(sample_response_id):
    """Create a negative feedback request."""
    from code_assistance_pipeline.models.query_models import (
        CodeFeedbackRequest,
        ErrorCategory,
    )

    return CodeFeedbackRequest(
        response_id=sample_response_id,
        rating=2,
        feedback="The response referenced wrong methods",
        was_helpful=False,
        is_helpful=False,
        comment="The response referenced wrong methods",
        error_category=ErrorCategory.WRONG_METHOD,
        expected_methods=["BaleService.ValidateBale", "BaleRepository.Insert"],
    )


@pytest.fixture
def correction_feedback_request(sample_response_id):
    """Create a correction feedback request with expected methods."""
    from code_assistance_pipeline.models.query_models import (
        CodeFeedbackRequest,
        ErrorCategory,
    )

    return CodeFeedbackRequest(
        response_id=sample_response_id,
        rating=1,
        feedback="Missing important context",
        was_helpful=False,
        is_helpful=False,
        comment="Missing important context",
        error_category=ErrorCategory.MISSING_CONTEXT,
        expected_methods=[
            "OrderService.ProcessOrder",
            "OrderValidator.Validate",
            "InventoryService.UpdateStock",
        ],
    )


# =============================================================================
# CodeFeedbackRequest Model Tests
# =============================================================================

class TestCodeFeedbackRequestModel:
    """Test the CodeFeedbackRequest Pydantic model."""

    def test_create_basic_feedback_request(self, pipeline_config):
        """Test creating a basic feedback request."""
        from code_assistance_pipeline.models.query_models import CodeFeedbackRequest

        request = CodeFeedbackRequest(
            response_id="test_response_123",
            rating=4,
        )

        assert request.response_id == "test_response_123"
        assert request.rating == 4
        assert request.was_helpful is True  # Default

    def test_create_feedback_with_all_fields(self, pipeline_config):
        """Test creating feedback with all fields."""
        from code_assistance_pipeline.models.query_models import (
            CodeFeedbackRequest,
            ErrorCategory,
        )

        request = CodeFeedbackRequest(
            response_id="test_response_123",
            rating=2,
            feedback="Not helpful",
            was_helpful=False,
            is_helpful=False,
            comment="Not helpful",
            error_category=ErrorCategory.INCORRECT_EXPLANATION,
            expected_methods=["Method1", "Method2"],
        )

        assert request.response_id == "test_response_123"
        assert request.rating == 2
        assert request.was_helpful is False
        assert request.is_helpful is False
        assert request.error_category == ErrorCategory.INCORRECT_EXPLANATION
        assert len(request.expected_methods) == 2

    def test_rating_validation_min(self, pipeline_config):
        """Test that rating has minimum validation."""
        from code_assistance_pipeline.models.query_models import CodeFeedbackRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CodeFeedbackRequest(
                response_id="test",
                rating=0,  # Below minimum
            )

    def test_rating_validation_max(self, pipeline_config):
        """Test that rating has maximum validation."""
        from code_assistance_pipeline.models.query_models import CodeFeedbackRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CodeFeedbackRequest(
                response_id="test",
                rating=6,  # Above maximum
            )

    def test_error_category_enum_values(self, pipeline_config):
        """Test ErrorCategory enum values."""
        from code_assistance_pipeline.models.query_models import ErrorCategory

        assert ErrorCategory.WRONG_METHOD.value == "wrong_method"
        assert ErrorCategory.WRONG_CLASS.value == "wrong_class"
        assert ErrorCategory.MISSING_CONTEXT.value == "missing_context"
        assert ErrorCategory.INCORRECT_EXPLANATION.value == "incorrect_explanation"
        assert ErrorCategory.OUTDATED_CODE.value == "outdated_code"
        assert ErrorCategory.OTHER.value == "other"


# =============================================================================
# CodeFeedbackResponse Model Tests
# =============================================================================

class TestCodeFeedbackResponseModel:
    """Test the CodeFeedbackResponse Pydantic model."""

    def test_create_success_response(self, pipeline_config):
        """Test creating a success response."""
        from code_assistance_pipeline.models.query_models import CodeFeedbackResponse

        response = CodeFeedbackResponse(
            success=True,
            feedback_id="fb_123",
        )

        assert response.success is True
        assert response.feedback_id == "fb_123"
        assert response.error is None

    def test_create_error_response(self, pipeline_config):
        """Test creating an error response."""
        from code_assistance_pipeline.models.query_models import CodeFeedbackResponse

        response = CodeFeedbackResponse(
            success=False,
            feedback_id=None,
            error="Failed to store feedback",
        )

        assert response.success is False
        assert response.error == "Failed to store feedback"


# =============================================================================
# Pipeline Feedback Submission Tests
# =============================================================================

class TestPipelineFeedbackSubmission:
    """Test feedback submission through the pipeline."""

    @pytest.mark.asyncio
    async def test_submit_positive_feedback(
        self, pipeline_config, positive_feedback_request
    ):
        """Test submitting positive feedback."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import CodeFeedbackResponse

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            MockRetriever.return_value = AsyncMock()
            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            # Mock the feedback service
            with patch('sql_pipeline.services.feedback_service.FeedbackService') as MockFeedbackService:
                mock_service = AsyncMock()
                mock_service.store_feedback.return_value = {
                    "feedback_id": "fb_test_123",
                    "success": True,
                }
                MockFeedbackService.return_value = mock_service

                # Set up mock MongoDB service
                pipeline._mongodb_service = MagicMock()

                response = await pipeline.submit_feedback(positive_feedback_request)

                assert isinstance(response, CodeFeedbackResponse)
                assert response.success is True

    @pytest.mark.asyncio
    async def test_submit_negative_feedback_with_error_category(
        self, pipeline_config, negative_feedback_request
    ):
        """Test submitting negative feedback with error category."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            MockRetriever.return_value = AsyncMock()
            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            with patch('sql_pipeline.services.feedback_service.FeedbackService') as MockFeedbackService:
                mock_service = AsyncMock()
                mock_service.store_feedback.return_value = {"feedback_id": "fb_123"}
                MockFeedbackService.return_value = mock_service

                pipeline._mongodb_service = MagicMock()

                response = await pipeline.submit_feedback(negative_feedback_request)

                # Verify feedback service was called with correction info
                mock_service.store_feedback.assert_called_once()
                call_kwargs = mock_service.store_feedback.call_args.kwargs

                # Should include error category in metadata
                assert "metadata" in call_kwargs
                assert call_kwargs["metadata"]["error_category"] == "wrong_method"

    @pytest.mark.asyncio
    async def test_submit_feedback_with_expected_methods(
        self, pipeline_config, correction_feedback_request
    ):
        """Test submitting feedback with expected methods."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            MockRetriever.return_value = AsyncMock()
            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            with patch('sql_pipeline.services.feedback_service.FeedbackService') as MockFeedbackService:
                mock_service = AsyncMock()
                mock_service.store_feedback.return_value = {"feedback_id": "fb_123"}
                MockFeedbackService.return_value = mock_service

                pipeline._mongodb_service = MagicMock()

                response = await pipeline.submit_feedback(correction_feedback_request)

                # Verify expected_methods were included
                mock_service.store_feedback.assert_called_once()
                call_kwargs = mock_service.store_feedback.call_args.kwargs

                # Should include expected_methods in correction or metadata
                assert "expected_methods" in call_kwargs.get("metadata", {})

    @pytest.mark.asyncio
    async def test_submit_feedback_without_mongodb(self, pipeline_config):
        """Test feedback submission when MongoDB is not available."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline
        from code_assistance_pipeline.models.query_models import CodeFeedbackRequest

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            MockRetriever.return_value = AsyncMock()
            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            # No MongoDB service
            pipeline._mongodb_service = None

            feedback = CodeFeedbackRequest(
                response_id="test_123",
                rating=4,
                is_helpful=True,
            )

            response = await pipeline.submit_feedback(feedback)

            # Should still succeed (fallback behavior)
            assert response.success is True

    @pytest.mark.asyncio
    async def test_submit_feedback_handles_storage_error(
        self, pipeline_config, positive_feedback_request
    ):
        """Test handling of storage errors during feedback submission."""
        from code_assistance_pipeline.pipeline import CodeAssistancePipeline

        with patch('code_assistance_pipeline.pipeline.CodeRetriever') as MockRetriever, \
             patch('code_assistance_pipeline.pipeline.ContextBuilder') as MockContextBuilder, \
             patch('code_assistance_pipeline.pipeline.ResponseGenerator') as MockGenerator:

            MockRetriever.return_value = AsyncMock()
            MockContextBuilder.return_value = MagicMock()
            MockGenerator.return_value = AsyncMock()

            pipeline = CodeAssistancePipeline()
            await pipeline.initialize()

            with patch('sql_pipeline.services.feedback_service.FeedbackService') as MockFeedbackService:
                mock_service = AsyncMock()
                mock_service.store_feedback.side_effect = Exception("Storage error")
                MockFeedbackService.return_value = mock_service

                pipeline._mongodb_service = MagicMock()

                response = await pipeline.submit_feedback(positive_feedback_request)

                # Should return failure response
                assert response.success is False


# =============================================================================
# MongoDB Feedback Storage Tests
# =============================================================================

class TestFeedbackStorage:
    """Test feedback storage in MongoDB."""

    @pytest.mark.requires_mongodb
    def test_store_feedback_document(
        self,
        mongodb_database,
        pipeline_config,
        sample_response_id,
    ):
        """Test storing a feedback document in MongoDB."""
        collection = mongodb_database["code_feedback"]

        feedback_doc = {
            "_id": f"test_{generate_test_id()}",
            "feedback_id": f"fb_{generate_test_id()}",
            "response_id": sample_response_id,
            "is_helpful": True,
            "rating": 5,
            "comment": "Great explanation!",
            "error_category": None,
            "expected_methods": [],
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
        }

        collection.insert_one(feedback_doc)

        stored = assert_document_stored(
            collection,
            feedback_doc["_id"],
            expected_fields=["response_id", "is_helpful", "rating"],
        )

        assert stored["is_helpful"] is True
        assert stored["rating"] == 5

    @pytest.mark.requires_mongodb
    def test_store_negative_feedback_with_correction(
        self,
        mongodb_database,
        pipeline_config,
        sample_response_id,
    ):
        """Test storing negative feedback with correction details."""
        collection = mongodb_database["code_feedback"]

        feedback_doc = {
            "_id": f"test_{generate_test_id()}",
            "feedback_id": f"fb_{generate_test_id()}",
            "response_id": sample_response_id,
            "is_helpful": False,
            "rating": 2,
            "comment": "Referenced wrong methods",
            "error_category": "wrong_method",
            "expected_methods": [
                "BaleService.ValidateBale",
                "BaleRepository.Insert",
            ],
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
        }

        collection.insert_one(feedback_doc)

        stored = assert_document_stored(
            collection,
            feedback_doc["_id"],
            expected_fields=["error_category", "expected_methods"],
        )

        assert stored["is_helpful"] is False
        assert stored["error_category"] == "wrong_method"
        assert len(stored["expected_methods"]) == 2

    @pytest.mark.requires_mongodb
    def test_query_feedback_by_response_id(
        self,
        mongodb_database,
        pipeline_config,
        sample_response_id,
    ):
        """Test querying feedback by response ID."""
        collection = mongodb_database["code_feedback"]

        # Insert multiple feedback items
        for i in range(3):
            collection.insert_one({
                "_id": f"test_{generate_test_id()}",
                "response_id": sample_response_id if i < 2 else "other_response",
                "is_helpful": i % 2 == 0,
                "rating": 3 + i,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            })

        # Query by response_id
        results = list(collection.find({
            "response_id": sample_response_id,
            "test_run_id": pipeline_config.test_run_id,
        }))

        assert len(results) == 2

    @pytest.mark.requires_mongodb
    def test_aggregate_feedback_by_error_category(
        self,
        mongodb_database,
        pipeline_config,
    ):
        """Test aggregating feedback by error category."""
        collection = mongodb_database["code_feedback"]

        # Insert feedback with different error categories
        categories = [
            "wrong_method",
            "wrong_method",
            "missing_context",
            "incorrect_explanation",
            "wrong_method",
        ]

        for i, cat in enumerate(categories):
            collection.insert_one({
                "_id": f"test_{generate_test_id()}",
                "response_id": f"resp_{i}",
                "is_helpful": False,
                "rating": 2,
                "error_category": cat,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            })

        # Aggregate by category
        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": "$error_category",
                "count": {"$sum": 1},
            }},
        ]

        stats = list(collection.aggregate(pipeline))

        # Should have stats for each category
        wrong_method_stat = next(
            (s for s in stats if s["_id"] == "wrong_method"), None
        )
        assert wrong_method_stat is not None
        assert wrong_method_stat["count"] == 3


# =============================================================================
# Feedback Analysis Tests
# =============================================================================

class TestFeedbackAnalysis:
    """Test feedback analysis functionality."""

    @pytest.mark.requires_mongodb
    def test_calculate_average_rating(
        self,
        mongodb_database,
        pipeline_config,
    ):
        """Test calculating average rating from feedback."""
        collection = mongodb_database["code_feedback"]

        # Insert feedback with various ratings
        ratings = [5, 4, 5, 3, 4, 5, 2, 4]
        for i, rating in enumerate(ratings):
            collection.insert_one({
                "_id": f"test_{generate_test_id()}",
                "response_id": f"resp_{i}",
                "is_helpful": rating >= 4,
                "rating": rating,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            })

        # Calculate average
        pipeline = [
            {"$match": {"test_run_id": pipeline_config.test_run_id}},
            {"$group": {
                "_id": None,
                "avg_rating": {"$avg": "$rating"},
                "total_count": {"$sum": 1},
                "helpful_count": {
                    "$sum": {"$cond": [{"$eq": ["$is_helpful", True]}, 1, 0]}
                },
            }},
        ]

        stats = list(collection.aggregate(pipeline))

        assert len(stats) == 1
        assert stats[0]["total_count"] == 8
        assert stats[0]["helpful_count"] == 6  # ratings >= 4
        assert 3.5 <= stats[0]["avg_rating"] <= 4.5

    @pytest.mark.requires_mongodb
    def test_find_common_expected_methods(
        self,
        mongodb_database,
        pipeline_config,
    ):
        """Test finding commonly expected methods from feedback."""
        collection = mongodb_database["code_feedback"]

        # Insert feedback with expected methods
        expected_methods_sets = [
            ["BaleService.SaveBale", "BaleService.ValidateBale"],
            ["BaleService.ValidateBale", "BaleRepository.Insert"],
            ["BaleService.ValidateBale"],  # Most common
        ]

        for i, methods in enumerate(expected_methods_sets):
            collection.insert_one({
                "_id": f"test_{generate_test_id()}",
                "response_id": f"resp_{i}",
                "is_helpful": False,
                "expected_methods": methods,
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            })

        # Find most frequently expected methods
        pipeline = [
            {"$match": {
                "test_run_id": pipeline_config.test_run_id,
                "expected_methods": {"$exists": True, "$ne": []},
            }},
            {"$unwind": "$expected_methods"},
            {"$group": {
                "_id": "$expected_methods",
                "count": {"$sum": 1},
            }},
            {"$sort": {"count": -1}},
        ]

        results = list(collection.aggregate(pipeline))

        # BaleService.ValidateBale should be most common
        assert results[0]["_id"] == "BaleService.ValidateBale"
        assert results[0]["count"] == 3


# =============================================================================
# Integration with Interaction Update Tests
# =============================================================================

class TestFeedbackInteractionUpdate:
    """Test feedback integration with interaction updates."""

    @pytest.mark.requires_mongodb
    def test_update_interaction_on_feedback(
        self,
        mongodb_database,
        pipeline_config,
        sample_response_id,
    ):
        """Test that interaction is updated when feedback is received."""
        interactions_collection = mongodb_database["code_interactions"]
        feedback_collection = mongodb_database["code_feedback"]

        # Create an interaction
        interaction_id = f"test_{generate_test_id()}"
        interactions_collection.insert_one({
            "_id": interaction_id,
            "response_id": sample_response_id,
            "query": "Test query",
            "answer": "Test answer",
            "feedback_received": False,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        })

        # Store feedback
        feedback_collection.insert_one({
            "_id": f"test_{generate_test_id()}",
            "response_id": sample_response_id,
            "is_helpful": True,
            "rating": 5,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        })

        # Update interaction
        interactions_collection.update_one(
            {"response_id": sample_response_id},
            {"$set": {
                "feedback_received": True,
                "feedback_rating": 5,
                "feedback_timestamp": datetime.utcnow(),
            }},
        )

        # Verify update
        updated = interactions_collection.find_one({"_id": interaction_id})

        assert updated["feedback_received"] is True
        assert updated["feedback_rating"] == 5
        assert "feedback_timestamp" in updated
