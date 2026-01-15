"""
Document Retrieval Pipeline

Main pipeline orchestrator for document query and retrieval.
Follows the same pattern as SQL pipeline for consistency.
"""

from typing import Dict, Optional

from ..base import Pipeline
from .context import DocumentPipelineContext
from .steps import (
    QueryUnderstandingStep,
    HybridRetrievalStep,
    DocumentGradingStep,
    AnswerGenerationStep,
    ValidationStep,
    LearningFeedbackStep,
)


class DocumentRetrievalPipeline(Pipeline):
    """
    Complete document retrieval pipeline.

    6-step RAG workflow:
    1. Query Understanding - Enhance and clarify queries
    2. Hybrid Retrieval - Vector + keyword search
    3. Document Grading - Relevance scoring
    4. Answer Generation - Context-aware synthesis
    5. Validation - Quality checks
    6. Learning Feedback - Continuous improvement

    Example:
        pipeline = DocumentRetrievalPipeline()
        context = DocumentPipelineContext(
            query="How do I configure MongoDB?",
            top_k=5
        )
        result = await pipeline.execute(context)
        print(result.answer)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the document retrieval pipeline.

        Args:
            config: Optional configuration dictionary
                - use_query_understanding: Enable query enhancement (default: True)
                - use_reranking: Enable cross-encoder reranking (default: True)
                - use_llm_grading: Enable LLM-based grading (default: True)
                - use_validation: Enable answer validation (default: True)
                - learning_enabled: Enable learning feedback (default: True)
                - vector_weight: Vector search weight (default: 0.8)
                - keyword_weight: Keyword search weight (default: 0.2)
                - min_relevance_score: Minimum grading score (default: 5.0)
                - max_context_tokens: Max tokens for generation (default: 4000)
        """
        self.config = config or {}

        steps = [
            QueryUnderstandingStep(
                enabled=self.config.get("use_query_understanding", True),
                llm_model=self.config.get("llm_model", "general"),
            ),
            HybridRetrievalStep(
                vector_weight=self.config.get("vector_weight", 0.8),
                keyword_weight=self.config.get("keyword_weight", 0.2),
                use_reranking=self.config.get("use_reranking", True),
                candidate_multiplier=self.config.get("candidate_multiplier", 4),
            ),
            DocumentGradingStep(
                min_score=self.config.get("min_relevance_score", 5.0),
                use_llm_grading=self.config.get("use_llm_grading", True),
                recency_weight=self.config.get("recency_weight", 0.2),
                trust_weight=self.config.get("trust_weight", 0.1),
            ),
            AnswerGenerationStep(
                llm_model=self.config.get("llm_model", "general"),
                max_context_tokens=self.config.get("max_context_tokens", 4000),
                temperature=self.config.get("temperature", 0.3),
            ),
            ValidationStep(
                enabled=self.config.get("use_validation", True),
                check_hallucinations=self.config.get("check_hallucinations", True),
                check_completeness=self.config.get("check_completeness", True),
            ),
            LearningFeedbackStep(
                enabled=self.config.get("learning_enabled", True),
                auto_update_embeddings=self.config.get("auto_update_embeddings", False),
            ),
        ]

        super().__init__(steps, config)

    @classmethod
    async def from_request(cls, request: Dict, config: Optional[Dict] = None) -> DocumentPipelineContext:
        """
        Create context from API request.

        Args:
            request: API request dictionary with fields:
                - query: User query string (required)
                - filters: Optional filter criteria
                - top_k: Number of results (default: 5)
                - include_citations: Include citations (default: True)
                - context: Optional conversation history
            config: Optional pipeline configuration

        Returns:
            DocumentPipelineContext ready for execution
        """
        return DocumentPipelineContext(
            query=request["query"],
            filters=request.get("filters", {}),
            top_k=request.get("top_k", 5),
            include_citations=request.get("include_citations", True),
            context_history=request.get("context", []),
        )

    async def execute(self, context: DocumentPipelineContext) -> DocumentPipelineContext:
        """
        Execute the pipeline.

        Args:
            context: Pipeline context with input query

        Returns:
            Updated context with answer and citations
        """
        import time
        start_time = time.time()

        # Run pipeline
        result = await super().execute(context)

        # Calculate total time
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        return result
