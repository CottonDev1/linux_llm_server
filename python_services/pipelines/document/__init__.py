"""
Document Retrieval Pipeline - Agentic RAG pipeline for knowledge base queries

This module implements a multi-step document retrieval pipeline with:
- Query understanding and expansion
- Hybrid retrieval (vector + BM25 with RRF fusion)
- Document grading for relevance filtering
- Learning feedback loop for continuous improvement

Architecture follows the CRAG (Corrective RAG) pattern with agentic enhancements.

Usage Example:
-------------
```python
from pipelines.document import DocumentRetrievalPipeline

# Initialize pipeline with required services
pipeline = DocumentRetrievalPipeline(
    mongodb_service=mongodb,
    embedding_service=embeddings,
    llm_service=llm,  # Optional for enhanced grading
)

# Execute retrieval
result = await pipeline.retrieve("What is the vacation policy?")

# Access graded documents
for doc in result.graded_documents:
    print(f"{doc.title}: {doc.grading_score:.2f}")

# Record feedback after user interaction
await pipeline.record_feedback(
    result.context,
    feedback_type="positive"
)
```
"""

# Base classes and data structures
from .base import (
    PipelineStep,
    PipelineContext,
    StepResult,
    RetrievedDocument,
    QueryType,
)

# Pipeline steps
from .query_understanding import QueryUnderstandingStep
from .hybrid_retrieval import HybridRetrievalStep, VectorOnlyRetrievalStep
from .document_grading import DocumentGradingStep
from .learning_feedback import LearningFeedbackStep, FeedbackType

# Orchestrators
from .orchestrator import (
    DocumentRetrievalPipeline,
    SimplifiedRetrievalPipeline,
    PipelineResult,
)

__all__ = [
    # Base classes
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

    # Feedback types
    "FeedbackType",

    # Orchestrators
    "DocumentRetrievalPipeline",
    "SimplifiedRetrievalPipeline",
    "PipelineResult",
]
