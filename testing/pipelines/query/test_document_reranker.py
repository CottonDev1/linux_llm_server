"""
Document Reranker Tests for Query/RAG Pipeline.

Tests for document reranking functionality including:
- Cross-encoder reranking simulation
- Score distribution validation
- Metadata preservation during reranking
- Top-K selection after reranking
- Reranker performance characteristics
- Handling of edge cases (empty results, single document)

Reranking is used to improve retrieval quality by rescoring
initial retrieval results with a more powerful model.
"""

import pytest
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_document_chunk,
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)


@dataclass
class RerankerResult:
    """Result from reranking operation."""
    document_id: str
    content: str
    initial_score: float
    reranked_score: float
    metadata: Dict[str, Any]


class MockCrossEncoderReranker:
    """
    Mock cross-encoder reranker for testing.

    Simulates cross-encoder behavior by computing relevance scores
    based on query-document overlap. In production, this would use
    a trained cross-encoder model.
    """

    def __init__(self, model_name: str = "mock-cross-encoder"):
        """
        Initialize mock reranker.

        Args:
            model_name: Name of the model (for logging)
        """
        self.model_name = model_name

    def _compute_relevance(self, query: str, document: str) -> float:
        """
        Compute mock relevance score based on term overlap.

        This simulates cross-encoder scoring. Real cross-encoders
        use transformer models for semantic matching.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score between 0 and 1
        """
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())

        if not query_terms or not doc_terms:
            return 0.0

        # Compute Jaccard-like similarity with boosting for exact matches
        intersection = query_terms & doc_terms
        union = query_terms | doc_terms

        base_score = len(intersection) / len(union) if union else 0.0

        # Boost for consecutive term matches (phrase matching)
        query_lower = query.lower()
        doc_lower = document.lower()
        phrase_boost = 0.0

        for term in query_terms:
            if term in doc_lower:
                phrase_boost += 0.1

        return min(1.0, base_score + phrase_boost)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        content_field: str = "content",
        top_k: int = None
    ) -> List[RerankerResult]:
        """
        Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of documents to rerank
            content_field: Field containing document text
            top_k: Optional limit on results

        Returns:
            List of RerankerResult sorted by reranked_score descending
        """
        results = []

        for doc in documents:
            content = doc.get(content_field, "")
            relevance_score = self._compute_relevance(query, content)

            result = RerankerResult(
                document_id=str(doc.get("_id", "")),
                content=content,
                initial_score=doc.get("initial_score", doc.get("similarity", 0.0)),
                reranked_score=relevance_score,
                metadata={
                    k: v for k, v in doc.items()
                    if k not in ["_id", content_field, "embedding", "initial_score", "similarity"]
                }
            )
            results.append(result)

        # Sort by reranked score
        results.sort(key=lambda x: x.reranked_score, reverse=True)

        if top_k:
            results = results[:top_k]

        return results


class TestCrossEncoderReranking:
    """Tests for cross-encoder based reranking."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_reranker_changes_ordering(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that reranker can change document ordering.

        Initial retrieval order may differ from final reranked order.
        """
        collection = mongodb_database["documents"]

        # Create documents with initial scores that don't match true relevance
        documents = [
            {
                "content": "Equipment maintenance procedures",
                "initial_score": 0.9,  # Ranked high initially
                "expected_relevance": "low",
            },
            {
                "content": "Safety PPE hard hat requirements for all personnel",
                "initial_score": 0.5,  # Ranked low initially
                "expected_relevance": "high",
            },
            {
                "content": "Cotton bale processing workflow",
                "initial_score": 0.7,
                "expected_relevance": "medium",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"Rerank Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["initial_score"] = doc_data["initial_score"]
            doc["expected_relevance"] = doc_data["expected_relevance"]
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Initial ordering by initial_score
        initial_order = sorted(results, key=lambda x: x["initial_score"], reverse=True)

        # Rerank with query about safety
        reranker = MockCrossEncoderReranker()
        query = "safety PPE hard hat requirements"
        reranked = reranker.rerank(query, results)

        # Get reranked order
        reranked_order = [r.content for r in reranked]
        initial_order_content = [d["content"] for d in initial_order]

        # Orders should be different
        assert reranked_order != initial_order_content, (
            "Reranking should change document order when query relevance differs"
        )

        # Top reranked result should be about safety (highest expected_relevance)
        top_reranked = reranked[0]
        assert "safety" in top_reranked.content.lower() or "ppe" in top_reranked.content.lower(), (
            f"Top reranked document should be about safety, got: {top_reranked.content[:50]}"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_reranker_preserves_metadata(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that reranking preserves document metadata.

        Important fields like project, department, etc. should survive reranking.
        """
        collection = mongodb_database["code_context"]

        # Create documents with rich metadata
        documents = [
            {
                "method_name": "ProcessBale",
                "class_name": "BaleProcessor",
                "project": "gin",
                "department": "Operations",
                "content": "Method for processing cotton bales",
            },
            {
                "method_name": "ValidateSafety",
                "class_name": "SafetyChecker",
                "project": "warehouse",
                "department": "Safety",
                "content": "Validates safety requirements for warehouse",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_code_method(
                method_name=doc_data["method_name"],
                class_name=doc_data["class_name"],
                project=doc_data["project"],
                code=f"public void {doc_data['method_name']}() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["content"] = doc_data["content"]
            doc["department"] = doc_data["department"]
            doc["initial_score"] = 0.8
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("safety warehouse", results)

        # Verify metadata is preserved
        for result in reranked:
            assert "project" in result.metadata, "Project should be in metadata"
            assert "method_name" in result.metadata, "Method name should be in metadata"
            assert "class_name" in result.metadata, "Class name should be in metadata"

        # Find the warehouse safety doc
        warehouse_doc = next(
            (r for r in reranked if "warehouse" in r.content.lower()),
            None
        )
        assert warehouse_doc is not None
        assert warehouse_doc.metadata["project"] == "warehouse"
        assert warehouse_doc.metadata["department"] == "Safety"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestScoreDistributionValidation:
    """Tests for reranker score distribution and normalization."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_scores_are_normalized(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that reranker scores are in valid range [0, 1].
        """
        collection = mongodb_database["documents"]

        # Create documents with varied content
        contents = [
            "Safety equipment for warehouse workers",
            "Cotton gin processing machinery",
            "Quality control procedures",
            "Emergency evacuation routes",
        ]

        stored_docs = []
        for i, content in enumerate(contents):
            doc = create_mock_document_chunk(
                title=f"Score Test {i}",
                content=content,
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["initial_score"] = 0.5
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("safety warehouse emergency", results)

        # Verify all scores are in valid range
        for result in reranked:
            assert 0.0 <= result.reranked_score <= 1.0, (
                f"Reranked score {result.reranked_score} is outside [0, 1]"
            )
            assert 0.0 <= result.initial_score <= 1.0, (
                f"Initial score {result.initial_score} is outside [0, 1]"
            )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_score_distribution_is_reasonable(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that score distribution has expected properties.

        - Highly relevant docs should score high
        - Irrelevant docs should score low
        - Scores should be spread across the range
        """
        collection = mongodb_database["documents"]

        # Create documents with varying relevance to "bale processing procedures"
        documents = [
            {
                "content": "Bale processing procedures for cotton gin operations",
                "expected_tier": "high",
            },
            {
                "content": "Processing bales requires careful procedures",
                "expected_tier": "high",
            },
            {
                "content": "Equipment maintenance for processing machinery",
                "expected_tier": "medium",
            },
            {
                "content": "Safety requirements for warehouse operations",
                "expected_tier": "low",
            },
            {
                "content": "Employee break room schedule",
                "expected_tier": "very_low",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"Distribution Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["expected_tier"] = doc_data["expected_tier"]
            doc["initial_score"] = 0.5
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()
        query = "bale processing procedures"
        reranked = reranker.rerank(query, results)

        # Group by expected tier
        tier_scores = {"high": [], "medium": [], "low": [], "very_low": []}
        for result in reranked:
            tier = result.metadata.get("expected_tier", "unknown")
            if tier in tier_scores:
                tier_scores[tier].append(result.reranked_score)

        # High tier should have higher average than low tier
        avg_high = sum(tier_scores["high"]) / len(tier_scores["high"]) if tier_scores["high"] else 0
        avg_low = sum(tier_scores["low"]) / len(tier_scores["low"]) if tier_scores["low"] else 0

        assert avg_high > avg_low, (
            f"High relevance docs ({avg_high:.2f}) should score higher than low ({avg_low:.2f})"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_reranker_handles_ties(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that reranker handles documents with identical scores.

        Ties should be resolved consistently (e.g., by original order).
        """
        collection = mongodb_database["documents"]

        # Create documents that will have identical relevance
        identical_docs = []
        for i in range(3):
            doc = create_mock_document_chunk(
                title=f"Identical Doc {i}",
                content="Same exact content for all documents",
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["initial_score"] = 0.5
            doc["doc_index"] = i  # Track original order
            doc["embedding"] = [0.5] * 384
            identical_docs.append(doc)

        insert_test_documents(collection, identical_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("same exact content", results)

        # All should have the same score
        scores = [r.reranked_score for r in reranked]
        assert len(set(scores)) == 1, "Identical docs should have identical scores"

        # Should still return all documents
        assert len(reranked) == 3, "Should return all tied documents"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestRerankerTopKSelection:
    """Tests for top-K selection after reranking."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_top_k_limits_results(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that top_k parameter correctly limits results.
        """
        collection = mongodb_database["documents"]

        # Create many documents
        num_docs = 20
        docs = []
        for i in range(num_docs):
            doc = create_mock_document_chunk(
                title=f"TopK Test {i}",
                content=f"Document {i} about various topics",
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["initial_score"] = 0.5
            doc["embedding"] = [0.5] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()

        # Test different top_k values
        for k in [1, 5, 10]:
            reranked = reranker.rerank("document topics", results, top_k=k)
            assert len(reranked) == k, f"top_k={k} should return {k} results"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_top_k_returns_highest_scores(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that top_k returns the documents with highest reranked scores.
        """
        collection = mongodb_database["documents"]

        # Create documents with known relevance
        documents = [
            {"content": "Bale processing procedures manual", "expected_score": "high"},
            {"content": "Cotton bale handling guide", "expected_score": "high"},
            {"content": "Equipment maintenance guide", "expected_score": "low"},
            {"content": "Break room schedule", "expected_score": "very_low"},
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"TopK Score Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["expected_score"] = doc_data["expected_score"]
            doc["initial_score"] = 0.5
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("bale processing", results, top_k=2)

        # Top 2 should be the "high" expected score documents
        for result in reranked:
            assert result.metadata.get("expected_score") == "high", (
                f"Top-2 should be high-scoring docs, got {result.metadata.get('expected_score')}"
            )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestRerankerEdgeCases:
    """Tests for reranker edge cases."""

    @pytest.mark.e2e
    def test_reranker_handles_empty_results(self):
        """
        Test that reranker handles empty document list gracefully.
        """
        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("any query", [])

        assert len(reranked) == 0, "Empty input should return empty output"

    @pytest.mark.e2e
    def test_reranker_handles_single_document(self):
        """
        Test that reranker handles single document correctly.
        """
        single_doc = {
            "_id": "test_id",
            "content": "Single document about safety procedures",
            "initial_score": 0.8,
        }

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("safety", [single_doc])

        assert len(reranked) == 1, "Should return the single document"
        assert reranked[0].document_id == "test_id"

    @pytest.mark.e2e
    def test_reranker_handles_empty_content(self):
        """
        Test that reranker handles documents with empty content.
        """
        docs = [
            {"_id": "empty", "content": "", "initial_score": 0.5},
            {"_id": "whitespace", "content": "   ", "initial_score": 0.5},
            {"_id": "normal", "content": "Normal document content", "initial_score": 0.5},
        ]

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("document content", docs)

        # Should return all documents
        assert len(reranked) == 3

        # Empty/whitespace docs should have low scores
        for result in reranked:
            if result.document_id in ["empty", "whitespace"]:
                assert result.reranked_score == 0.0, (
                    f"Empty content should have zero score, got {result.reranked_score}"
                )

    @pytest.mark.e2e
    def test_reranker_handles_missing_content_field(self):
        """
        Test that reranker handles documents missing content field.
        """
        docs = [
            {"_id": "missing", "title": "Document with no content", "initial_score": 0.5},
        ]

        reranker = MockCrossEncoderReranker()
        reranked = reranker.rerank("any query", docs)

        assert len(reranked) == 1
        assert reranked[0].reranked_score == 0.0, "Missing content should score zero"

    @pytest.mark.e2e
    def test_reranker_handles_special_characters(self):
        """
        Test that reranker handles special characters in content.
        """
        docs = [
            {
                "_id": "special",
                "content": "Document with special chars: @#$%^&*()!",
                "initial_score": 0.5,
            },
            {
                "_id": "unicode",
                "content": "Unicode content: cafe",
                "initial_score": 0.5,
            },
            {
                "_id": "normal",
                "content": "Normal document content here",
                "initial_score": 0.5,
            },
        ]

        reranker = MockCrossEncoderReranker()
        # Should not raise exception
        reranked = reranker.rerank("document content", docs)

        assert len(reranked) == 3, "Should handle all documents including special chars"


class TestRerankerPerformance:
    """Tests for reranker performance characteristics."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.slow
    def test_reranker_scales_with_document_count(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test reranker performance with increasing document counts.

        Reranking time should scale linearly with document count.
        """
        import time

        collection = mongodb_database["documents"]

        # Create large dataset
        num_docs = 100
        docs = []
        for i in range(num_docs):
            doc = create_mock_document_chunk(
                title=f"Perf Test {i}",
                content=f"Document {i} about cotton bale processing and safety procedures",
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["initial_score"] = 0.5
            doc["embedding"] = [0.5] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        reranker = MockCrossEncoderReranker()
        query = "cotton bale safety processing procedures"

        # Time reranking
        start = time.time()
        reranked = reranker.rerank(query, results)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0, f"Reranking {num_docs} docs took {elapsed:.2f}s (should be <5s)"
        assert len(reranked) == num_docs, "Should rerank all documents"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    def test_reranker_with_long_content(self):
        """
        Test reranker handles documents with very long content.
        """
        import time

        # Create document with long content
        long_content = " ".join(["word"] * 10000)  # 10K words
        docs = [
            {
                "_id": "long",
                "content": long_content,
                "initial_score": 0.5,
            },
            {
                "_id": "short",
                "content": "Short document",
                "initial_score": 0.5,
            },
        ]

        reranker = MockCrossEncoderReranker()

        start = time.time()
        reranked = reranker.rerank("word document", docs)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 2.0, f"Reranking long content took {elapsed:.2f}s"
        assert len(reranked) == 2
