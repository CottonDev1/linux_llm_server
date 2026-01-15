"""
Hybrid Retrieval Step - Combines vector search with BM25 using Reciprocal Rank Fusion

This step implements hybrid retrieval which consistently outperforms either
vector-only or keyword-only search by 30-40% on standard benchmarks. The
approach combines:

1. **Vector Search**: Semantic similarity using dense embeddings
   - Captures meaning and context
   - Handles synonyms and paraphrases
   - Works well for natural language queries

2. **BM25/Text Search**: Lexical matching using MongoDB text indexes
   - Exact keyword matching
   - Important for technical terms, product names, acronyms
   - Fast and efficient

3. **Reciprocal Rank Fusion (RRF)**: Combines rankings from both methods
   - No need to normalize scores across different systems
   - Simple but highly effective fusion technique
   - Configurable k parameter (typically 60)

Design Rationale:
-----------------
Vector search alone can miss documents that use different terminology,
while keyword search misses semantically similar content. Hybrid search
captures both aspects, significantly improving recall and precision.

RRF is preferred over other fusion methods (CombSUM, CombMNZ) because:
- It doesn't require score normalization
- It's robust to different score distributions
- It emphasizes documents ranked highly by both methods

Implementation Notes:
--------------------
- Vector search uses existing MongoDB $vectorSearch or in-memory fallback
- BM25 uses MongoDB text indexes with weighted fields
- Results are merged using RRF with configurable k parameter
- Supports filtering by department, type, subject, and custom filters
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import numpy as np

from .base import (
    PipelineStep,
    PipelineContext,
    StepResult,
    RetrievedDocument,
    QueryType,
)


class HybridRetrievalStep(PipelineStep):
    """
    Performs hybrid retrieval combining vector and BM25 search.

    This step executes both retrieval methods in parallel and combines
    results using Reciprocal Rank Fusion (RRF). It leverages the
    existing MongoDBService for vector operations and adds BM25
    text search using MongoDB's text indexes.
    """

    def __init__(
        self,
        mongodb_service: Any,
        embedding_service: Any,
        collection_name: str = "documents",
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 10,
        retrieval_multiplier: int = 3,
        min_score: float = 0.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hybrid retrieval step.

        Args:
            mongodb_service: MongoDBService instance for database operations
            embedding_service: EmbeddingService for generating query embeddings
            collection_name: MongoDB collection to search
            rrf_k: RRF constant (typically 60, higher = more weight to lower ranks)
            vector_weight: Weight for vector search in RRF (0-1)
            bm25_weight: Weight for BM25 search in RRF (0-1)
            top_k: Number of final results to return
            retrieval_multiplier: Fetch this many times top_k from each method
            min_score: Minimum score threshold for results
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.mongodb_service = mongodb_service
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.retrieval_multiplier = retrieval_multiplier
        self.min_score = min_score

    @property
    def name(self) -> str:
        return "HybridRetrieval"

    @property
    def requires(self) -> Set[str]:
        return {"rewritten_query", "expanded_queries"}

    @property
    def produces(self) -> Set[str]:
        return {
            "retrieved_documents",
            "retrieval_method",
            "vector_results_count",
            "bm25_results_count",
        }

    async def execute(self, context: PipelineContext) -> StepResult:
        """
        Execute hybrid retrieval with vector + BM25 and RRF fusion.

        Steps:
        1. Prepare queries (rewritten + expanded)
        2. Execute vector search in parallel for all queries
        3. Execute BM25 search in parallel for all queries
        4. Combine results using Reciprocal Rank Fusion
        5. Deduplicate and rank final results
        """
        queries = [context.rewritten_query] + context.expanded_queries
        queries = list(set(q for q in queries if q))  # Dedupe and filter empty

        if not queries:
            return StepResult(
                success=False,
                errors=["No queries available for retrieval"],
            )

        try:
            # Build filter from context
            filter_query = self._build_filter(context.filters, context.extracted_entities)

            # Execute vector and BM25 search in parallel
            vector_task = self._vector_search_multi(queries, filter_query)
            bm25_task = self._bm25_search_multi(queries, filter_query)

            vector_results, bm25_results = await asyncio.gather(
                vector_task,
                bm25_task,
                return_exceptions=True
            )

            # Handle potential exceptions
            if isinstance(vector_results, Exception):
                self.logger.warning(f"Vector search failed: {vector_results}")
                vector_results = []

            if isinstance(bm25_results, Exception):
                self.logger.warning(f"BM25 search failed: {bm25_results}")
                bm25_results = []

            # Determine retrieval method used
            if vector_results and bm25_results:
                retrieval_method = "hybrid"
            elif vector_results:
                retrieval_method = "vector_only"
            elif bm25_results:
                retrieval_method = "bm25_only"
            else:
                return StepResult(
                    success=False,
                    errors=["Both vector and BM25 search returned no results"],
                )

            # Combine using Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                vector_results,
                bm25_results
            )

            # Convert to RetrievedDocument objects
            retrieved_docs = self._to_retrieved_documents(fused_results)

            return StepResult(
                success=True,
                data={
                    "retrieved_documents": retrieved_docs,
                    "retrieval_method": retrieval_method,
                    "vector_results_count": len(vector_results),
                    "bm25_results_count": len(bm25_results),
                },
                metadata={
                    "queries_executed": len(queries),
                    "fusion_method": "rrf",
                    "rrf_k": self.rrf_k,
                    "total_candidates": len(fused_results),
                }
            )

        except Exception as e:
            self.logger.exception("Hybrid retrieval failed")
            return StepResult(
                success=False,
                errors=[f"Retrieval failed: {str(e)}"],
            )

    def _build_filter(
        self,
        user_filters: Dict[str, Any],
        extracted_entities: Dict[str, List[str]]
    ) -> Optional[Dict]:
        """
        Build MongoDB filter from user filters and extracted entities.

        Combines explicit user filters with entities extracted from
        the query (e.g., department mentions, document types).
        """
        filter_parts = []

        # Add explicit user filters
        if user_filters:
            if user_filters.get("department"):
                filter_parts.append({"department": user_filters["department"]})
            if user_filters.get("type"):
                filter_parts.append({"type": user_filters["type"]})
            if user_filters.get("subject"):
                filter_parts.append({"subject": user_filters["subject"]})
            if user_filters.get("tags"):
                filter_parts.append({"tags": {"$in": user_filters["tags"]}})

        # Add filters from extracted entities (lower priority - use $or)
        entity_filters = []
        if extracted_entities.get("departments"):
            entity_filters.append({
                "department": {"$in": extracted_entities["departments"]}
            })
        if extracted_entities.get("document_types"):
            entity_filters.append({
                "type": {"$in": extracted_entities["document_types"]}
            })

        # Combine filters
        if filter_parts and entity_filters:
            # User filters are required, entity filters are optional boost
            return {"$and": filter_parts}
        elif filter_parts:
            return {"$and": filter_parts}
        elif entity_filters:
            # Only entity filters - make them optional with $or
            return {"$or": entity_filters} if len(entity_filters) > 1 else entity_filters[0]

        return None

    async def _vector_search_multi(
        self,
        queries: List[str],
        filter_query: Optional[Dict]
    ) -> List[Tuple[str, float, Dict]]:
        """
        Execute vector search for multiple queries in parallel.

        Returns a list of (document_id, score, document) tuples.
        Documents are deduplicated by ID, keeping the highest score.
        """
        candidates_per_query = self.top_k * self.retrieval_multiplier

        # Generate embeddings for all queries
        embeddings = await asyncio.gather(*[
            self.embedding_service.generate_embedding(q)
            for q in queries
        ])

        # Search for each query embedding
        all_results = []

        for query_embedding in embeddings:
            try:
                # Use existing MongoDBService vector search
                results = await self.mongodb_service._vector_search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=candidates_per_query,
                    filter_query=filter_query,
                    threshold=self.min_score
                )

                for doc in results:
                    doc_id = str(doc.get("_id", doc.get("id", "")))
                    score = doc.get("_similarity", 0.0)
                    all_results.append((doc_id, score, doc))

            except Exception as e:
                self.logger.warning(f"Vector search query failed: {e}")
                continue

        # Deduplicate by document ID, keeping highest score
        best_by_id: Dict[str, Tuple[float, Dict]] = {}
        for doc_id, score, doc in all_results:
            if doc_id not in best_by_id or score > best_by_id[doc_id][0]:
                best_by_id[doc_id] = (score, doc)

        # Sort by score descending
        sorted_results = sorted(
            [(doc_id, score, doc) for doc_id, (score, doc) in best_by_id.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:self.top_k * self.retrieval_multiplier]

    async def _bm25_search_multi(
        self,
        queries: List[str],
        filter_query: Optional[Dict]
    ) -> List[Tuple[str, float, Dict]]:
        """
        Execute BM25/text search for multiple queries.

        Uses MongoDB's text search with $text operator and text indexes.
        Returns a list of (document_id, score, document) tuples.
        """
        candidates_per_query = self.top_k * self.retrieval_multiplier
        collection = self.mongodb_service.db[self.collection_name]

        all_results = []

        for query in queries:
            try:
                # Build text search query
                text_query = {"$text": {"$search": query}}

                # Combine with filter if provided
                if filter_query:
                    search_query = {"$and": [text_query, filter_query]}
                else:
                    search_query = text_query

                # Execute search with text score
                cursor = collection.find(
                    search_query,
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(candidates_per_query)

                docs = await cursor.to_list(length=candidates_per_query)

                for doc in docs:
                    doc_id = str(doc.get("_id", doc.get("id", "")))
                    # Normalize text score to 0-1 range (text scores can be > 1)
                    raw_score = doc.get("score", 0.0)
                    # Use sigmoid-like normalization for text scores
                    normalized_score = raw_score / (1 + raw_score)
                    all_results.append((doc_id, normalized_score, doc))

            except Exception as e:
                # Text search may fail if no text index exists
                self.logger.debug(f"BM25 search query failed: {e}")
                continue

        # Deduplicate by document ID, keeping highest score
        best_by_id: Dict[str, Tuple[float, Dict]] = {}
        for doc_id, score, doc in all_results:
            if doc_id not in best_by_id or score > best_by_id[doc_id][0]:
                best_by_id[doc_id] = (score, doc)

        # Sort by score descending
        sorted_results = sorted(
            [(doc_id, score, doc) for doc_id, (score, doc) in best_by_id.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:self.top_k * self.retrieval_multiplier]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float, Dict]],
        bm25_results: List[Tuple[str, float, Dict]]
    ) -> List[Tuple[str, float, Dict]]:
        """
        Combine results from vector and BM25 search using RRF.

        Reciprocal Rank Fusion formula:
        RRF(d) = sum(1 / (k + rank_i(d))) for each ranking i

        Where k is a constant (typically 60) that determines how much
        weight is given to lower-ranked documents.

        The algorithm:
        1. For each document, calculate its RRF score across both rankings
        2. Weight the contributions based on vector_weight and bm25_weight
        3. Sort by combined RRF score

        Args:
            vector_results: Ranked results from vector search
            bm25_results: Ranked results from BM25 search

        Returns:
            Combined and re-ranked results
        """
        # Build document lookup for metadata
        doc_lookup: Dict[str, Dict] = {}
        for doc_id, score, doc in vector_results + bm25_results:
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}

        # Vector search contribution
        for rank, (doc_id, score, doc) in enumerate(vector_results):
            rrf_contribution = self.vector_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

        # BM25 search contribution
        for rank, (doc_id, score, doc) in enumerate(bm25_results):
            rrf_contribution = self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build result list with RRF score and document
        results = []
        for doc_id, rrf_score in sorted_docs[:self.top_k]:
            doc = doc_lookup.get(doc_id, {})
            # Normalize RRF score to 0-1 range for consistency
            # Max possible RRF score is (vector_weight + bm25_weight) / (k + 1)
            max_score = (self.vector_weight + self.bm25_weight) / (self.rrf_k + 1)
            normalized_score = rrf_score / max_score if max_score > 0 else rrf_score
            results.append((doc_id, normalized_score, doc))

        return results

    def _to_retrieved_documents(
        self,
        fused_results: List[Tuple[str, float, Dict]]
    ) -> List[RetrievedDocument]:
        """
        Convert fused results to RetrievedDocument objects.

        Handles document schema mapping from MongoDB documents to
        our standard RetrievedDocument structure.
        """
        documents = []

        for doc_id, score, doc in fused_results:
            # Extract fields with fallbacks for different schemas
            content = doc.get("content", doc.get("text", ""))
            title = doc.get("title", doc.get("name", "Untitled"))
            parent_id = doc.get("parent_id", doc.get("document_id", doc_id))

            # Build metadata from remaining fields
            metadata = {}
            for key in ["department", "type", "subject", "tags", "upload_date",
                        "file_name", "chunk_index", "total_chunks"]:
                if key in doc:
                    metadata[key] = doc[key]

            # Determine source based on document markers
            source = "hybrid"
            if "_similarity" in doc:
                source = "vector"
            elif "score" in doc:
                source = "bm25"

            documents.append(RetrievedDocument(
                id=doc_id,
                parent_id=parent_id,
                content=content,
                title=title,
                score=score,
                source=source,
                metadata=metadata,
            ))

        return documents


class VectorOnlyRetrievalStep(HybridRetrievalStep):
    """
    Vector-only retrieval for when BM25 is not available or not desired.

    This is a simplified version of HybridRetrievalStep that only uses
    vector search. Useful for:
    - Collections without text indexes
    - Testing vector search in isolation
    - High-latency scenarios where BM25 adds overhead
    """

    def __init__(
        self,
        mongodb_service: Any,
        embedding_service: Any,
        collection_name: str = "documents",
        top_k: int = 10,
        min_score: float = 0.0,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(
            mongodb_service=mongodb_service,
            embedding_service=embedding_service,
            collection_name=collection_name,
            top_k=top_k,
            min_score=min_score,
            vector_weight=1.0,
            bm25_weight=0.0,
            logger=logger
        )

    @property
    def name(self) -> str:
        return "VectorOnlyRetrieval"

    async def execute(self, context: PipelineContext) -> StepResult:
        """Execute vector-only retrieval."""
        queries = [context.rewritten_query] + context.expanded_queries
        queries = list(set(q for q in queries if q))

        if not queries:
            return StepResult(
                success=False,
                errors=["No queries available for retrieval"],
            )

        try:
            filter_query = self._build_filter(context.filters, context.extracted_entities)
            vector_results = await self._vector_search_multi(queries, filter_query)

            if not vector_results:
                return StepResult(
                    success=False,
                    errors=["Vector search returned no results"],
                )

            retrieved_docs = self._to_retrieved_documents(
                [(doc_id, score, doc) for doc_id, score, doc in vector_results[:self.top_k]]
            )

            return StepResult(
                success=True,
                data={
                    "retrieved_documents": retrieved_docs,
                    "retrieval_method": "vector_only",
                    "vector_results_count": len(vector_results),
                    "bm25_results_count": 0,
                },
            )

        except Exception as e:
            self.logger.exception("Vector retrieval failed")
            return StepResult(
                success=False,
                errors=[f"Vector retrieval failed: {str(e)}"],
            )
