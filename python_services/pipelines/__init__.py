"""
Pipelines Package - Modular pipeline components for RAG workflows

This package contains pipeline implementations following a step-based architecture
where each step is an independent, testable unit that can be composed into
larger workflows.

Available Pipelines:
-------------------
- DocumentRetrievalPipeline: Full-featured document retrieval with hybrid search
- SimplifiedRetrievalPipeline: Fast retrieval without LLM overhead

Pipeline Steps:
--------------
- QueryUnderstandingStep: Query classification, rewriting, and expansion
- HybridRetrievalStep: Vector + BM25 search with RRF fusion
- DocumentGradingStep: LLM-based relevance grading
- LearningFeedbackStep: Feedback recording for continuous improvement
"""

# Base classes from parent pipelines module
from .base import (
    PipelineStep as BasePipelineStep,
    PipelineContext as BasePipelineContext,
    StepResult as BaseStepResult,
    Pipeline,
    PipelineStatus,
)

# Document pipeline - the main implementation
from .document import (
    # Document-specific base classes (extend the base)
    PipelineStep,
    PipelineContext,
    StepResult,
    RetrievedDocument,
    QueryType,

    # Steps
    QueryUnderstandingStep,
    HybridRetrievalStep,
    VectorOnlyRetrievalStep,
    DocumentGradingStep,
    LearningFeedbackStep,

    # Feedback
    FeedbackType,

    # Orchestrators
    DocumentRetrievalPipeline,
    SimplifiedRetrievalPipeline,
    PipelineResult,
)

__all__ = [
    # Base pipeline classes
    "BasePipelineStep",
    "BasePipelineContext",
    "BaseStepResult",
    "Pipeline",
    "PipelineStatus",

    # Document pipeline classes
    "PipelineStep",
    "PipelineContext",
    "StepResult",
    "RetrievedDocument",
    "QueryType",

    # Pipeline steps
    "QueryUnderstandingStep",
    "HybridRetrievalStep",
    "VectorOnlyRetrievalStep",
    "DocumentGradingStep",
    "LearningFeedbackStep",

    # Feedback
    "FeedbackType",

    # Orchestrators
    "DocumentRetrievalPipeline",
    "SimplifiedRetrievalPipeline",
    "PipelineResult",
]
