"""
Document Pipeline Context

Extended PipelineContext with document-specific fields.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from ..base import PipelineContext


@dataclass
class Citation:
    """Citation information for a document chunk."""
    source: str  # File path or document title
    chunk_index: int
    relevance_score: float
    excerpt: str  # Brief excerpt from document
    chunk_id: Optional[str] = None


@dataclass
class DocumentPipelineContext(PipelineContext):
    """
    Document retrieval pipeline context.

    Extends PipelineContext with document-specific fields for
    the 6-step retrieval and generation workflow.
    """

    # Input parameters
    query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    context_history: List[str] = field(default_factory=list)
    include_citations: bool = True

    # Step 1: Query Understanding
    expanded_query: Optional[str] = None
    query_type: Optional[str] = None  # factual, procedural, exploratory
    extracted_filters: Dict[str, Any] = field(default_factory=dict)
    clarification_needed: bool = False
    suggested_refinements: List[str] = field(default_factory=list)

    # Step 2: Hybrid Retrieval
    retrieved_chunks: List[Any] = field(default_factory=list)  # DocumentChunk objects
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_methods: Dict[str, str] = field(default_factory=dict)  # chunk_id -> "vector"/"keyword"/"both"

    # Step 3: Document Grading
    graded_chunks: List[Any] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    rejected_chunks: List[Any] = field(default_factory=list)

    # Step 4: Answer Generation
    answer: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    context_used: List[str] = field(default_factory=list)  # chunk IDs used

    # Step 5: Validation
    is_valid: bool = False
    validation_issues: List[str] = field(default_factory=list)
    hallucination_score: float = 0.0
    completeness_score: float = 0.0

    # Step 6: Learning
    learning_record_id: Optional[str] = None

    # Metadata
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunks_retrieved: int = 0
    chunks_used: int = 0
    processing_time_ms: int = 0

    def to_response(self) -> Dict[str, Any]:
        """
        Convert context to API response format.

        Returns:
            Dict suitable for JSON serialization
        """
        return {
            "query_id": self.query_id,
            "answer": self.answer or "",
            "citations": [
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "relevance_score": c.relevance_score,
                    "excerpt": c.excerpt,
                }
                for c in self.citations
            ] if self.include_citations else [],
            "confidence": self.confidence,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_used": self.chunks_used,
            "processing_time_ms": self.processing_time_ms,
            "status": self.status.value,
            "errors": self.errors,
        }

    def to_learning_record(self) -> Dict[str, Any]:
        """
        Convert context to learning database record.

        Returns:
            Dict suitable for MongoDB storage
        """
        return {
            "query_id": self.query_id,
            "query": self.query,
            "expanded_query": self.expanded_query,
            "query_type": self.query_type,
            "answer": self.answer,
            "citations": [
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "relevance_score": c.relevance_score,
                }
                for c in self.citations
            ],
            "retrieved_chunks": [c.id if hasattr(c, 'id') else str(c) for c in self.retrieved_chunks],
            "graded_scores": self.relevance_scores,
            "validation_result": {
                "is_valid": self.is_valid,
                "hallucination_score": self.hallucination_score,
                "completeness_score": self.completeness_score,
                "issues": self.validation_issues,
            },
            "metrics": {
                "retrieval_time_ms": int(self.step_timings.get("hybrid_retrieval", 0) * 1000),
                "generation_time_ms": int(self.step_timings.get("answer_generation", 0) * 1000),
                "total_time_ms": self.processing_time_ms,
                "chunks_retrieved": self.chunks_retrieved,
                "chunks_used": self.chunks_used,
                "confidence": self.confidence,
            },
            "errors": self.errors,
        }
