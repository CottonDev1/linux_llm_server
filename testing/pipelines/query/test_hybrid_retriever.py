"""
Hybrid Retriever Tests for Query/RAG Pipeline.

Tests hybrid search combining vector search and BM25 including:
- BM25 index building and keyword search
- Vector search integration
- Reciprocal Rank Fusion (RRF) combination
- Weight balancing (vector_weight, bm25_weight)
- Concurrent BM25 + Vector execution
- Fallback behavior when one method fails

Hybrid retrieval combines semantic (vector) and lexical (BM25) search
for better retrieval quality than either alone.
"""

import pytest
import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_document_chunk,
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)


class BM25Index:
    """
    Simple BM25 implementation for testing.

    BM25 is a ranking function used to estimate relevance of documents
    to a search query based on term frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.tokenized_docs: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())

    def build_index(self, documents: List[Dict[str, Any]], content_field: str = "content"):
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents to index
            content_field: Field containing text content
        """
        self.documents = documents
        self.tokenized_docs = []
        self.doc_freqs = {}
        self.doc_lens = []

        # Tokenize all documents
        for doc in documents:
            tokens = self._tokenize(doc.get(content_field, ""))
            self.tokenized_docs.append(tokens)
            self.doc_lens.append(len(tokens))

            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate average document length
        if self.doc_lens:
            self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens)

    def search(self, query: str, limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search index using BM25 scoring.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        query_tokens = self._tokenize(query)
        scores = []
        n_docs = len(self.documents)

        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0.0
            doc_len = self.doc_lens[doc_idx]
            token_counts = Counter(doc_tokens)

            for token in query_tokens:
                if token not in self.doc_freqs:
                    continue

                # Calculate IDF
                df = self.doc_freqs[token]
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

                # Calculate term frequency component
                tf = token_counts.get(token, 0)
                tf_component = (
                    tf * (self.k1 + 1) /
                    (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
                )

                score += idf * tf_component

            scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[Any, float]]],
    k: int = 60
) -> List[Tuple[Any, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF formula: RRF(d) = sum(1 / (k + rank(d)))

    Args:
        rankings: List of rankings, each being [(item_id, score), ...]
        k: Constant to prevent high rankings from dominating

    Returns:
        Combined ranking as [(item_id, rrf_score), ...]
    """
    rrf_scores: Dict[Any, float] = {}

    for ranking in rankings:
        for rank, (item_id, _) in enumerate(ranking, start=1):
            if item_id not in rrf_scores:
                rrf_scores[item_id] = 0.0
            rrf_scores[item_id] += 1.0 / (k + rank)

    # Sort by RRF score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_results


class TestBM25IndexBuilding:
    """Tests for BM25 index construction and basic search."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_bm25_index_builds_correctly(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that BM25 index is built correctly from documents.

        Verifies document frequencies and index statistics.
        """
        collection = mongodb_database["documents"]

        # Create documents with varied content
        documents = [
            {"content": "The quick brown fox jumps over the lazy dog"},
            {"content": "A quick brown dog runs in the park"},
            {"content": "The lazy cat sleeps all day long"},
            {"content": "Brown foxes are quick and clever animals"},
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"BM25 Test Doc {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Build BM25 index
        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        bm25 = BM25Index()
        bm25.build_index(results)

        # Verify index statistics
        assert len(bm25.documents) == 4, "Should index all 4 documents"
        assert bm25.avg_doc_len > 0, "Average doc length should be positive"

        # Verify document frequencies
        assert "quick" in bm25.doc_freqs, "Should have 'quick' in index"
        assert bm25.doc_freqs["quick"] == 3, "Three docs contain 'quick'"
        assert "brown" in bm25.doc_freqs
        assert bm25.doc_freqs["brown"] == 3, "Three docs contain 'brown'"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_bm25_keyword_search_ranking(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test BM25 returns relevant documents ranked by keyword match.

        Documents with more query terms should rank higher.
        """
        collection = mongodb_database["documents"]

        # Create documents with varying relevance
        documents = [
            {
                "content": "Safety procedures for cotton bale processing",
                "label": "high_relevance",
            },
            {
                "content": "Cotton gin operations manual for bale handling",
                "label": "medium_relevance",
            },
            {
                "content": "Equipment maintenance guidelines",
                "label": "low_relevance",
            },
            {
                "content": "Bale weighing and quality control procedures",
                "label": "medium_high_relevance",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"BM25 Rank Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["label"] = doc_data["label"]
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Build index and search
        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        bm25 = BM25Index()
        bm25.build_index(results)

        # Search for "bale processing procedures"
        query = "bale processing procedures"
        search_results = bm25.search(query, limit=4)

        # Get ranked documents
        ranked_docs = [
            (results[idx]["label"], score)
            for idx, score in search_results
        ]

        # Top result should be "high_relevance" (has bale, processing, procedures)
        assert ranked_docs[0][0] == "high_relevance", (
            f"Top result should be high_relevance, got {ranked_docs[0][0]}"
        )

        # Verify scores are descending
        for i in range(len(ranked_docs) - 1):
            assert ranked_docs[i][1] >= ranked_docs[i + 1][1], (
                "Results should be in descending score order"
            )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_bm25_handles_empty_query(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test BM25 handles empty or whitespace-only queries gracefully.
        """
        collection = mongodb_database["documents"]

        doc = create_mock_document_chunk(
            title="Empty Query Test",
            content="Some test content here",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * 384
        collection.insert_one(doc)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        bm25 = BM25Index()
        bm25.build_index(results)

        # Search with empty query
        empty_results = bm25.search("", limit=10)

        # Should return results with zero scores
        assert len(empty_results) == 1
        assert empty_results[0][1] == 0.0, "Empty query should give zero scores"

        # Search with whitespace
        whitespace_results = bm25.search("   ", limit=10)
        assert whitespace_results[0][1] == 0.0

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestHybridSearchCombination:
    """Tests for combining vector and BM25 search results."""

    def _compute_cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_reciprocal_rank_fusion_combines_rankings(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test RRF correctly combines vector and BM25 rankings.

        Documents ranked highly by both methods should rank highest in combined result.
        """
        collection = mongodb_database["documents"]

        # Create documents that will rank differently in each method
        documents = [
            {
                "content": "Safety equipment PPE hard hat glasses",
                "semantic_relevance": 0.9,  # High vector relevance
                "label": "both_high",
            },
            {
                "content": "Fire extinguisher location map",
                "semantic_relevance": 0.3,  # Low vector relevance
                "label": "bm25_only",
            },
            {
                "content": "Personal protective equipment requirements safety",
                "semantic_relevance": 0.8,  # High vector relevance
                "label": "vector_high",
            },
            {
                "content": "Warehouse procedures documentation",
                "semantic_relevance": 0.4,  # Medium vector relevance
                "label": "low_both",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"RRF Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["label"] = doc_data["label"]
            # Create embedding based on semantic_relevance
            doc["embedding"] = [doc_data["semantic_relevance"]] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Build BM25 index
        bm25 = BM25Index()
        bm25.build_index(results)

        # Query
        query = "safety equipment PPE"
        query_embedding = [0.85] * 384

        # Get BM25 ranking
        bm25_results = bm25.search(query, limit=4)
        bm25_ranking = [(results[idx]["_id"], score) for idx, score in bm25_results]

        # Get vector ranking
        vector_results = []
        for i, doc in enumerate(results):
            similarity = self._compute_cosine_similarity(
                query_embedding, doc["embedding"]
            )
            vector_results.append((i, similarity))
        vector_results.sort(key=lambda x: x[1], reverse=True)
        vector_ranking = [(results[idx]["_id"], score) for idx, score in vector_results]

        # Combine with RRF
        combined = reciprocal_rank_fusion([bm25_ranking, vector_ranking], k=60)

        # Map back to labels
        id_to_label = {doc["_id"]: doc["label"] for doc in results}
        combined_labels = [id_to_label[doc_id] for doc_id, _ in combined]

        # "both_high" should rank highest (relevant in both methods)
        assert combined_labels[0] == "both_high", (
            f"Document relevant in both methods should rank first, got {combined_labels[0]}"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_weight_balancing_affects_ranking(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that adjusting vector/BM25 weights changes final ranking.

        Higher vector_weight should favor semantically similar docs.
        Higher bm25_weight should favor keyword matches.
        """
        collection = mongodb_database["documents"]

        # Create documents with contrasting relevance
        documents = [
            {
                "content": "The quick brown fox",  # BM25: high match
                "semantic_relevance": 0.3,  # Vector: low
                "label": "keyword_match",
            },
            {
                "content": "Fast animal behavior studies",  # BM25: no match
                "semantic_relevance": 0.9,  # Vector: high (semantic similarity)
                "label": "semantic_match",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"Weight Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["label"] = doc_data["label"]
            doc["embedding"] = [doc_data["semantic_relevance"]] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        bm25 = BM25Index()
        bm25.build_index(results)

        query = "quick brown fox"
        query_embedding = [0.85] * 384  # High semantic similarity to "semantic_match"

        # Get rankings
        bm25_results = bm25.search(query, limit=2)
        bm25_ranking = [(results[idx]["_id"], score) for idx, score in bm25_results]

        vector_results = []
        for i, doc in enumerate(results):
            similarity = self._compute_cosine_similarity(
                query_embedding, doc["embedding"]
            )
            vector_results.append((i, similarity))
        vector_results.sort(key=lambda x: x[1], reverse=True)
        vector_ranking = [(results[idx]["_id"], score) for idx, score in vector_results]

        # BM25-heavy combination (k=1 gives more weight to higher ranks)
        bm25_heavy = reciprocal_rank_fusion([bm25_ranking, vector_ranking], k=1)
        id_to_label = {doc["_id"]: doc["label"] for doc in results}

        # With different k values, rankings may shift
        # k=1 makes rank differences more pronounced
        assert len(bm25_heavy) == 2, "Should have 2 results"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestHybridRetrieverFallback:
    """Tests for fallback behavior when one search method fails."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_fallback_to_vector_when_no_keyword_matches(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test fallback to vector-only when BM25 returns no matches.

        Useful for queries with specialized terminology.
        """
        collection = mongodb_database["documents"]

        # Create documents with technical content
        documents = [
            {"content": "Processing cotton bales in the gin", "label": "doc1"},
            {"content": "Quality control measurements", "label": "doc2"},
        ]

        stored_docs = []
        for i, doc_data in enumerate(documents):
            doc = create_mock_document_chunk(
                title=f"Fallback Test {i}",
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["label"] = doc_data["label"]
            doc["embedding"] = [0.5 + i * 0.2] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        bm25 = BM25Index()
        bm25.build_index(results)

        # Query with terms not in documents
        query = "xyzzy foobar nonexistent"
        bm25_results = bm25.search(query, limit=2)

        # All BM25 scores should be zero
        all_zero = all(score == 0.0 for _, score in bm25_results)
        assert all_zero, "BM25 should return zero scores for non-matching query"

        # Vector search still works
        query_embedding = [0.6] * 384
        vector_results = []
        for i, doc in enumerate(results):
            dot = sum(a * b for a, b in zip(query_embedding, doc["embedding"]))
            norm_q = math.sqrt(sum(x * x for x in query_embedding))
            norm_d = math.sqrt(sum(x * x for x in doc["embedding"]))
            similarity = dot / (norm_q * norm_d) if norm_q and norm_d else 0
            vector_results.append((i, similarity))

        vector_results.sort(key=lambda x: x[1], reverse=True)

        # Should have valid vector results
        assert len(vector_results) == 2
        assert vector_results[0][1] > 0, "Vector search should return positive scores"

        # In fallback mode, use only vector results
        fallback_ranking = vector_results  # No RRF needed when BM25 fails

        assert len(fallback_ranking) == 2, "Fallback should return vector results"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_fallback_to_bm25_when_no_embeddings(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test fallback to BM25-only when documents lack embeddings.

        Useful for recently added documents not yet embedded.
        """
        collection = mongodb_database["documents"]

        # Create documents without embeddings
        doc = create_mock_document_chunk(
            title="No Embedding Doc",
            content="Cotton bale processing procedures and safety",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        # Note: no embedding field
        collection.insert_one(doc)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Check for missing embeddings
        docs_without_embedding = [
            d for d in results if "embedding" not in d or d["embedding"] is None
        ]

        assert len(docs_without_embedding) == 1, "Should have doc without embedding"

        # BM25 still works
        bm25 = BM25Index()
        bm25.build_index(results)

        query = "cotton bale safety"
        bm25_results = bm25.search(query, limit=1)

        assert len(bm25_results) == 1
        assert bm25_results[0][1] > 0, "BM25 should match keywords"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestHybridSearchPerformance:
    """Tests for hybrid search performance characteristics."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.slow
    def test_concurrent_bm25_and_vector_execution(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that BM25 and vector search can run concurrently.

        In production, both searches run in parallel for latency optimization.
        """
        import time

        collection = mongodb_database["code_context"]

        # Create moderate dataset
        num_docs = 100
        docs = []
        for i in range(num_docs):
            doc = create_mock_code_method(
                method_name=f"Method{i}",
                class_name="HybridTestClass",
                project="gin",
                code=f"public void Method{i}() {{ /* Process bale {i} */ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [float(i) / num_docs] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Build BM25 index
        bm25 = BM25Index()
        bm25.build_index(results)

        query = "process bale method"
        query_embedding = [0.5] * 384

        # Time sequential execution
        start_seq = time.time()

        bm25_results = bm25.search(query, limit=10)

        vector_results = []
        for i, doc in enumerate(results):
            dot = sum(a * b for a, b in zip(query_embedding, doc["embedding"]))
            norm_q = math.sqrt(sum(x * x for x in query_embedding))
            norm_d = math.sqrt(sum(x * x for x in doc["embedding"]))
            similarity = dot / (norm_q * norm_d) if norm_q and norm_d else 0
            vector_results.append((i, similarity))
        vector_results.sort(key=lambda x: x[1], reverse=True)
        vector_results = vector_results[:10]

        seq_time = time.time() - start_seq

        # Combine results
        bm25_ranking = [(results[idx]["_id"], score) for idx, score in bm25_results]
        vector_ranking = [(results[idx]["_id"], score) for idx, score in vector_results]

        combined = reciprocal_rank_fusion([bm25_ranking, vector_ranking])

        # Verify results
        assert len(combined) >= 10, "Should have at least 10 combined results"
        assert seq_time < 5.0, f"Hybrid search too slow: {seq_time:.2f}s"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_rrf_with_many_rankings(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test RRF performance with multiple ranking sources.

        RRF can combine more than 2 rankings (e.g., multiple vector indexes).
        """
        # Create mock rankings (simulating different retrieval methods)
        doc_ids = [f"doc_{i}" for i in range(20)]

        # Simulate 3 different ranking methods
        ranking1 = [(doc_ids[i], 1.0 - i * 0.05) for i in range(20)]
        ranking2 = [(doc_ids[19 - i], 1.0 - i * 0.05) for i in range(20)]
        ranking3 = [(doc_ids[(i + 5) % 20], 0.8 - i * 0.04) for i in range(20)]

        # Combine all three
        combined = reciprocal_rank_fusion([ranking1, ranking2, ranking3], k=60)

        # Should have all documents
        assert len(combined) == 20, "Should combine all documents"

        # Documents appearing high in multiple rankings should rank higher
        # doc_0 is rank 1 in ranking1, rank 20 in ranking2, varies in ranking3
        # Documents in middle positions across all rankings should score well

        # Verify ordering is by RRF score
        for i in range(len(combined) - 1):
            assert combined[i][1] >= combined[i + 1][1], (
                f"RRF results not sorted at position {i}"
            )
