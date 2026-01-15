"""
Document Grading Step - Grades retrieved documents for relevance

This step implements the grading phase of Corrective RAG (CRAG), which
evaluates each retrieved document for relevance to the query before
passing them to the generation phase.

Why Document Grading?
--------------------
Retrieval often returns documents that are topically related but not
actually useful for answering the query. Without grading:
- Irrelevant documents dilute the context
- LLM may hallucinate from tangential information
- Response quality degrades with noisy retrieval

Benefits of grading:
- 30-45% reduction in hallucinations (based on research)
- Smaller, more focused context for LLM
- Enables corrective retrieval when quality is low
- Provides explainability for retrieval decisions

Grading Approach:
----------------
We use a lightweight LLM call to grade each document on a 0-1 scale
with reasoning. This is more accurate than threshold-based filtering
on retrieval scores, which don't capture semantic relevance.

For efficiency:
- Documents are graded in parallel (batched LLM calls if supported)
- High-confidence retrieval can skip grading (configurable)
- Grading prompt is optimized for speed (short, structured)
- Optional: Use a smaller model for grading vs generation
"""

import asyncio
from typing import Any, Dict, List, Optional, Set
import logging
import re

from .base import (
    PipelineStep,
    PipelineContext,
    StepResult,
    RetrievedDocument,
    QueryType,
)


class DocumentGradingStep(PipelineStep):
    """
    Grades retrieved documents for query relevance.

    Each document is evaluated on whether it contains information
    useful for answering the query. Documents below the threshold
    are filtered out, and the remaining documents are re-ranked
    by their relevance scores.

    The grading can be:
    - LLM-based: Uses a language model for semantic relevance
    - Heuristic: Falls back to retrieval score analysis
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        relevance_threshold: float = 0.5,
        min_documents: int = 1,
        max_documents: int = 5,
        batch_size: int = 5,
        include_reasoning: bool = True,
        skip_for_high_scores: bool = True,
        high_score_threshold: float = 0.85,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the document grading step.

        Args:
            llm_service: Optional LLM service for semantic grading.
                        If None, uses heuristic scoring.
            relevance_threshold: Minimum score (0-1) to keep a document
            min_documents: Minimum documents to return (even below threshold)
            max_documents: Maximum documents to return
            batch_size: Number of documents to grade in parallel
            include_reasoning: Whether to generate grading explanations
            skip_for_high_scores: Skip grading if retrieval scores are high
            high_score_threshold: Threshold for skipping grading
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.llm_service = llm_service
        self.relevance_threshold = relevance_threshold
        self.min_documents = min_documents
        self.max_documents = max_documents
        self.batch_size = batch_size
        self.include_reasoning = include_reasoning
        self.skip_for_high_scores = skip_for_high_scores
        self.high_score_threshold = high_score_threshold

    @property
    def name(self) -> str:
        return "DocumentGrading"

    @property
    def requires(self) -> Set[str]:
        return {"original_query", "retrieved_documents"}

    @property
    def produces(self) -> Set[str]:
        return {
            "graded_documents",
            "average_relevance",
            "documents_filtered_count",
            "require_web_fallback",
        }

    async def execute(self, context: PipelineContext) -> StepResult:
        """
        Grade retrieved documents for relevance.

        Steps:
        1. Check if grading should be skipped (high scores or flag)
        2. Grade each document using LLM or heuristics
        3. Filter documents below threshold
        4. Re-rank by relevance score
        5. Signal if corrective retrieval is needed
        """
        documents = context.retrieved_documents

        if not documents:
            return StepResult(
                success=True,
                data={
                    "graded_documents": [],
                    "average_relevance": 0.0,
                    "documents_filtered_count": 0,
                    "require_web_fallback": True,  # No docs, need fallback
                },
                warnings=["No documents to grade"],
            )

        # Check if we should skip grading
        if context.skip_grading or self._should_skip_grading(documents):
            self.logger.debug("Skipping grading due to high retrieval scores")
            # Use retrieval scores as grading scores
            for doc in documents:
                doc.grading_score = doc.score
                doc.grading_reasoning = "Skipped (high retrieval confidence)"

            return StepResult(
                success=True,
                data={
                    "graded_documents": documents[:self.max_documents],
                    "average_relevance": sum(d.score for d in documents) / len(documents),
                    "documents_filtered_count": 0,
                    "require_web_fallback": False,
                },
                metadata={"grading_skipped": True},
            )

        try:
            # Grade documents
            query = context.original_query
            graded_docs = await self._grade_documents(query, documents)

            # Filter and rank by relevance
            relevant_docs = [d for d in graded_docs if d.is_relevant]

            # Ensure minimum documents
            if len(relevant_docs) < self.min_documents:
                # Include top documents even if below threshold
                all_sorted = sorted(graded_docs, key=lambda d: d.grading_score or 0, reverse=True)
                relevant_docs = all_sorted[:self.min_documents]
                self.logger.debug(
                    f"Including {len(relevant_docs)} docs below threshold to meet minimum"
                )

            # Limit to max documents
            relevant_docs = sorted(
                relevant_docs,
                key=lambda d: d.grading_score or 0,
                reverse=True
            )[:self.max_documents]

            # Calculate metrics
            filtered_count = len(documents) - len(relevant_docs)
            avg_relevance = (
                sum(d.grading_score or 0 for d in relevant_docs) / len(relevant_docs)
                if relevant_docs else 0.0
            )

            # Determine if we need corrective retrieval
            need_fallback = avg_relevance < 0.5 or len(relevant_docs) < self.min_documents

            return StepResult(
                success=True,
                data={
                    "graded_documents": relevant_docs,
                    "average_relevance": avg_relevance,
                    "documents_filtered_count": filtered_count,
                    "require_web_fallback": need_fallback,
                },
                metadata={
                    "grading_method": "llm" if self.llm_service else "heuristic",
                    "documents_graded": len(documents),
                    "documents_passed": len(relevant_docs),
                }
            )

        except Exception as e:
            self.logger.exception("Document grading failed")
            # Fall back to using retrieval scores
            for doc in documents:
                doc.grading_score = doc.score
                doc.grading_reasoning = f"Grading failed: {str(e)}"

            return StepResult(
                success=True,  # Partial success - we can proceed
                data={
                    "graded_documents": documents[:self.max_documents],
                    "average_relevance": sum(d.score for d in documents) / len(documents),
                    "documents_filtered_count": 0,
                    "require_web_fallback": False,
                },
                warnings=[f"Grading failed, using retrieval scores: {str(e)}"],
            )

    def _should_skip_grading(self, documents: List[RetrievedDocument]) -> bool:
        """
        Determine if grading should be skipped based on retrieval scores.

        If all documents have high retrieval scores, grading adds latency
        without much benefit. This is an optimization for high-confidence
        retrievals.
        """
        if not self.skip_for_high_scores:
            return False

        if not documents:
            return True

        # Skip if average score is very high
        avg_score = sum(d.score for d in documents) / len(documents)
        return avg_score >= self.high_score_threshold

    async def _grade_documents(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Grade each document for relevance to the query.

        Uses LLM for semantic grading if available, otherwise falls
        back to heuristic analysis based on keyword overlap and
        structural features.
        """
        if self.llm_service is not None:
            return await self._grade_with_llm(query, documents)
        else:
            return self._grade_with_heuristics(query, documents)

    async def _grade_with_llm(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Use LLM to grade documents for relevance.

        Grades documents in batches for efficiency. Each document
        receives a score (0-1) and optional reasoning.
        """
        async def grade_single(doc: RetrievedDocument) -> RetrievedDocument:
            """Grade a single document."""
            # Truncate content for efficiency
            content = doc.content[:1500] if len(doc.content) > 1500 else doc.content

            prompt = f"""Grade this document's relevance to the query.

Query: {query}

Document:
Title: {doc.title}
Content: {content}

Instructions:
- Score from 0.0 to 1.0 where 1.0 is perfectly relevant
- Consider if the document helps answer the query
- 0.5+ means the document is useful
- 0.3-0.5 means tangentially related
- Below 0.3 means not relevant

{"Provide your response in this format: SCORE: [number] REASON: [brief explanation]" if self.include_reasoning else "Respond with only the numeric score (e.g., 0.7)"}"""

            try:
                response = await self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=50 if self.include_reasoning else 10,
                    temperature=0.0,
                )

                # Parse response
                score, reasoning = self._parse_grading_response(response)
                doc.grading_score = score
                doc.grading_reasoning = reasoning

            except Exception as e:
                self.logger.warning(f"LLM grading failed for doc {doc.id}: {e}")
                # Fall back to retrieval score
                doc.grading_score = doc.score
                doc.grading_reasoning = f"LLM grading failed: {str(e)}"

            return doc

        # Grade in batches for parallel efficiency
        graded_docs = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = await asyncio.gather(*[grade_single(d) for d in batch])
            graded_docs.extend(batch_results)

        return graded_docs

    def _parse_grading_response(self, response: str) -> tuple[float, str]:
        """
        Parse the LLM grading response to extract score and reasoning.

        Handles various response formats:
        - "SCORE: 0.8 REASON: The document explains..."
        - "0.8"
        - "Score is 0.8 because..."
        """
        response = response.strip()

        # Try structured format first
        score_match = re.search(r'SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)

        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to 0-1
                reasoning = reason_match.group(1).strip() if reason_match else ""
                return score, reasoning
            except ValueError:
                pass

        # Try to find any number in response
        numbers = re.findall(r'(\d+\.?\d*)', response)
        for num_str in numbers:
            try:
                score = float(num_str)
                if 0 <= score <= 1:
                    return score, response
                elif score <= 10:
                    # Assume 0-10 scale, normalize
                    return score / 10.0, response
                elif score <= 100:
                    # Assume percentage
                    return score / 100.0, response
            except ValueError:
                continue

        # Default to medium relevance if parsing fails
        self.logger.warning(f"Could not parse grading response: {response[:100]}")
        return 0.5, f"Parse failed: {response[:100]}"

    def _grade_with_heuristics(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Grade documents using heuristic analysis.

        This fallback method analyzes:
        - Keyword overlap between query and document
        - Query term frequency in document
        - Document structure and length
        - Title relevance
        """
        query_terms = set(self._tokenize(query.lower()))

        for doc in documents:
            # Calculate keyword overlap
            doc_terms = set(self._tokenize(doc.content.lower()))
            title_terms = set(self._tokenize(doc.title.lower()))

            # Query term coverage in document
            content_overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            title_overlap = len(query_terms & title_terms) / len(query_terms) if query_terms else 0

            # Query term frequency (how often query terms appear)
            term_frequency = sum(
                doc.content.lower().count(term) for term in query_terms
            ) / max(len(doc.content.split()), 1)

            # Combine signals
            # Title match is very important
            # Content overlap is important
            # Frequency adds some signal
            heuristic_score = (
                0.3 * title_overlap +
                0.5 * content_overlap +
                0.1 * min(term_frequency * 10, 1.0) +
                0.1 * doc.score  # Original retrieval score
            )

            doc.grading_score = max(0.0, min(1.0, heuristic_score))
            doc.grading_reasoning = f"Heuristic: title={title_overlap:.2f}, content={content_overlap:.2f}"

        return documents

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for heuristic grading.

        Splits on non-alphanumeric characters and filters
        short tokens and common stop words.
        """
        # Split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Filter stop words and short tokens
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
            'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
        }

        return [t for t in tokens if len(t) > 2 and t not in stop_words]
