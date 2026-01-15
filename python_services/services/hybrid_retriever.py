"""
Hybrid Retriever Service
========================

Combines vector similarity search with BM25 keyword search using
Reciprocal Rank Fusion (RRF) for comprehensive document retrieval.

Architecture:
- Vector Search: MongoDB Atlas vector search (semantic similarity)
- BM25 Search: rank-bm25 library with in-memory index
- Fusion: RRF combines rankings with configurable k parameter

The BM25 index is built lazily on first query and cached for subsequent queries.
Index rebuilding can be triggered manually or on a schedule.
"""

import logging
import time
import asyncio
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

from core.log_utils import log_info, log_warning, log_error

logger = logging.getLogger(__name__)

# BM25 library (install with: pip install rank-bm25)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. BM25 search will be disabled. Install with: pip install rank-bm25")


@dataclass
class RetrievedDoc:
    """A document with retrieval scores."""
    document_id: str
    chunk_id: str
    content: str
    title: Optional[str] = None
    source_file: Optional[str] = None
    department: Optional[str] = None
    doc_type: Optional[str] = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    vector_rank: int = 0
    bm25_rank: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retriever combining vector and BM25 search with RRF fusion.

    Usage:
        retriever = HybridRetriever(mongodb_service, embedding_service)
        await retriever.initialize()

        results = await retriever.search(
            query="How do I configure the system?",
            limit=10,
            vector_weight=0.7,
            bm25_weight=0.3
        )
    """

    def __init__(
        self,
        mongodb_service: Any,
        embedding_service: Any,
        collection_name: str = "documents",
        rrf_k: int = 60,
        persist_index: bool = True,
        index_ttl: int = 3600,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            mongodb_service: MongoDB service for vector search
            embedding_service: Embedding service for query embedding
            collection_name: Name of the documents collection
            rrf_k: RRF parameter (default 60, as per original RRF paper)
            persist_index: Whether to persist BM25 index to disk
            index_ttl: Time-to-live for cached index in seconds (default: 1 hour)
            cache_dir: Directory for index cache (default: python_services/.cache/bm25)
        """
        self._mongodb_service = mongodb_service
        self._embedding_service = embedding_service
        self._collection_name = collection_name
        self._rrf_k = rrf_k
        self._persist_index = persist_index
        self._index_ttl = index_ttl

        # Set up cache directory
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            # Use python_services/.cache/bm25 by default
            base_dir = Path(__file__).parent.parent
            self._cache_dir = base_dir / ".cache" / "bm25"

        self._index_file = self._cache_dir / f"{collection_name}_bm25_index.pkl"

        # BM25 index (lazy loaded)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []
        self._bm25_doc_ids: List[str] = []
        self._bm25_doc_map: Dict[str, Dict] = {}
        self._index_built_at: Optional[float] = None
        self._index_lock = asyncio.Lock()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the retriever and try to load cached index."""
        if self._initialized:
            return

        # Try to load persisted index
        if self._persist_index and BM25_AVAILABLE:
            loaded = self._load_index()
            if loaded:
                log_info("Hybrid Retriever", f"Loaded BM25 index from cache ({len(self._bm25_doc_ids)} documents)")
            else:
                log_info("Hybrid Retriever", "No cached index found (will build on first query)")
        else:
            log_info("Hybrid Retriever", "Initialized (BM25 index will be built on first query)")

        self._initialized = True

    def _save_index(self) -> bool:
        """
        Save BM25 index to disk.

        Returns:
            True if save succeeded, False otherwise
        """
        if not self._persist_index or self._bm25_index is None:
            return False

        try:
            # Create cache directory if needed
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Prepare index data
            index_data = {
                "bm25_corpus": self._bm25_corpus,
                "bm25_doc_ids": self._bm25_doc_ids,
                "bm25_doc_map": self._bm25_doc_map,
                "index_built_at": self._index_built_at,
                "collection_name": self._collection_name,
            }

            # Save to disk
            with open(self._index_file, "wb") as f:
                pickle.dump(index_data, f)

            log_info("Hybrid Retriever", f"Saved BM25 index to {self._index_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            return False

    def _load_index(self) -> bool:
        """
        Load BM25 index from disk if available and not expired.

        Returns:
            True if load succeeded, False otherwise
        """
        if not self._persist_index or not BM25_AVAILABLE:
            return False

        if not self._index_file.exists():
            return False

        try:
            # Load index data
            with open(self._index_file, "rb") as f:
                index_data = pickle.load(f)

            # Check if index is for correct collection
            if index_data.get("collection_name") != self._collection_name:
                logger.warning("Cached index is for different collection, ignoring")
                return False

            # Check TTL
            built_at = index_data.get("index_built_at", 0)
            if time.time() - built_at > self._index_ttl:
                log_info("Hybrid Retriever", "Cached index expired, will rebuild")
                return False

            # Restore index data
            self._bm25_corpus = index_data["bm25_corpus"]
            self._bm25_doc_ids = index_data["bm25_doc_ids"]
            self._bm25_doc_map = index_data["bm25_doc_map"]
            self._index_built_at = index_data["index_built_at"]

            # Rebuild BM25 index from corpus
            self._bm25_index = BM25Okapi(self._bm25_corpus)

            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def _is_index_expired(self) -> bool:
        """Check if the current index has expired based on TTL."""
        if self._index_built_at is None:
            return True
        return time.time() - self._index_built_at > self._index_ttl

    def invalidate_index(self) -> bool:
        """
        Invalidate and remove the cached BM25 index.

        This forces a rebuild on the next search operation.

        Returns:
            True if cache was cleared, False otherwise
        """
        self._bm25_index = None
        self._bm25_corpus = []
        self._bm25_doc_ids = []
        self._bm25_doc_map = {}
        self._index_built_at = None

        # Remove cached file
        if self._index_file.exists():
            try:
                self._index_file.unlink()
                log_info("Hybrid Retriever", "BM25 index cache invalidated")
                return True
            except Exception as e:
                logger.error(f"Failed to remove cached index: {e}")
                return False

        return True

    async def build_bm25_index(self, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the BM25 index from documents in MongoDB.

        Args:
            force_rebuild: Force rebuild even if index exists
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available - skipping index build")
            return

        async with self._index_lock:
            if self._bm25_index is not None and not force_rebuild:
                log_info("Hybrid Retriever", "BM25 index already exists, skipping rebuild")
                return

            log_info("Hybrid Retriever", "Building BM25 index...")
            start_time = time.time()

            try:
                # Fetch all document chunks from MongoDB
                collection = self._mongodb_service.db[self._collection_name]
                cursor = collection.find(
                    {"content": {"$exists": True, "$ne": ""}},
                    {"_id": 1, "content": 1, "title": 1, "parent_id": 1,
                     "source_file": 1, "department": 1, "type": 1,
                     "chunk_index": 1, "total_chunks": 1, "metadata": 1}
                )

                corpus = []
                doc_ids = []
                doc_map = {}

                async for doc in cursor:
                    doc_id = str(doc.get("_id", ""))
                    content = doc.get("content", "")

                    if not content or not doc_id:
                        continue

                    # Tokenize for BM25
                    tokens = self._tokenize(content)
                    if not tokens:
                        continue

                    corpus.append(tokens)
                    doc_ids.append(doc_id)
                    doc_map[doc_id] = {
                        "content": content,
                        "title": doc.get("title"),
                        "parent_id": doc.get("parent_id"),
                        "source_file": doc.get("source_file"),
                        "department": doc.get("department"),
                        "type": doc.get("type"),
                        "chunk_index": doc.get("chunk_index", 0),
                        "total_chunks": doc.get("total_chunks", 1),
                        "metadata": doc.get("metadata", {}),
                    }

                if not corpus:
                    logger.warning("No documents found for BM25 index")
                    return

                # Build BM25 index
                self._bm25_index = BM25Okapi(corpus)
                self._bm25_corpus = corpus
                self._bm25_doc_ids = doc_ids
                self._bm25_doc_map = doc_map
                self._index_built_at = time.time()

                build_time = time.time() - start_time
                log_info("Hybrid Retriever", f"BM25 index built: {len(corpus)} documents in {build_time:.2f}s")

                # Persist index to disk
                if self._persist_index:
                    self._save_index()

            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")
                raise

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Uses simple whitespace tokenization with lowercasing and
        basic punctuation handling.
        """
        # Lowercase and remove non-alphanumeric (except spaces)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split on whitespace and filter empty
        tokens = [t for t in text.split() if t and len(t) > 1]
        return tokens

    async def _vector_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedDoc]:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters
            min_score: Minimum similarity threshold

        Returns:
            List of RetrievedDoc with vector_score populated
        """
        try:
            # Generate query embedding
            query_embedding = await self._embedding_service.generate_embedding(query)

            # Build filter query for MongoDB
            filter_query = None
            if filters:
                filter_query = {}
                if "project" in filters:
                    filter_query["project"] = filters["project"]
                if "department" in filters:
                    filter_query["department"] = filters["department"]
                if "type" in filters:
                    filter_query["type"] = filters["type"]

            # Perform vector search
            results = await self._mongodb_service._vector_search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=limit,
                filter_query=filter_query,
                threshold=min_score,
            )

            # Convert to RetrievedDoc
            docs = []
            for rank, doc in enumerate(results):
                docs.append(RetrievedDoc(
                    document_id=doc.get("parent_id", str(doc.get("_id", ""))),
                    chunk_id=str(doc.get("_id", "")),
                    content=doc.get("content", ""),
                    title=doc.get("title"),
                    source_file=doc.get("source_file"),
                    department=doc.get("department"),
                    doc_type=doc.get("type"),
                    vector_score=doc.get("_similarity", 0.0),
                    vector_rank=rank + 1,
                    chunk_index=doc.get("chunk_index", 0),
                    total_chunks=doc.get("total_chunks", 1),
                    metadata=doc.get("metadata", {}),
                ))

            return docs

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDoc]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of RetrievedDoc with bm25_score populated
        """
        if not BM25_AVAILABLE or self._bm25_index is None:
            logger.debug("BM25 search skipped (index not available)")
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            # Get BM25 scores
            scores = self._bm25_index.get_scores(query_tokens)

            # Get top-k indices
            scored_indices = [(i, s) for i, s in enumerate(scores) if s > 0]
            scored_indices.sort(key=lambda x: x[1], reverse=True)

            # Apply filters and limit
            docs = []
            for rank, (idx, score) in enumerate(scored_indices):
                if len(docs) >= limit:
                    break

                doc_id = self._bm25_doc_ids[idx]
                doc_data = self._bm25_doc_map.get(doc_id, {})

                # Apply filters
                if filters:
                    if "project" in filters and doc_data.get("metadata", {}).get("project") != filters["project"]:
                        continue
                    if "department" in filters and doc_data.get("department") != filters["department"]:
                        continue
                    if "type" in filters and doc_data.get("type") != filters["type"]:
                        continue

                docs.append(RetrievedDoc(
                    document_id=doc_data.get("parent_id", doc_id),
                    chunk_id=doc_id,
                    content=doc_data.get("content", ""),
                    title=doc_data.get("title"),
                    source_file=doc_data.get("source_file"),
                    department=doc_data.get("department"),
                    doc_type=doc_data.get("type"),
                    bm25_score=score,
                    bm25_rank=rank + 1,
                    chunk_index=doc_data.get("chunk_index", 0),
                    total_chunks=doc_data.get("total_chunks", 1),
                    metadata=doc_data.get("metadata", {}),
                ))

            return docs

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedDoc],
        bm25_results: List[RetrievedDoc],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[RetrievedDoc]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion.

        RRF formula: score = sum(weight / (k + rank)) for each result list

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results

        Returns:
            Fused results sorted by RRF score
        """
        k = self._rrf_k
        doc_scores: Dict[str, Tuple[float, RetrievedDoc]] = {}

        # Add vector scores
        for doc in vector_results:
            rrf_contrib = vector_weight / (k + doc.vector_rank)
            if doc.chunk_id in doc_scores:
                existing_score, existing_doc = doc_scores[doc.chunk_id]
                existing_doc.vector_score = doc.vector_score
                existing_doc.vector_rank = doc.vector_rank
                doc_scores[doc.chunk_id] = (existing_score + rrf_contrib, existing_doc)
            else:
                doc.rrf_score = rrf_contrib
                doc_scores[doc.chunk_id] = (rrf_contrib, doc)

        # Add BM25 scores
        for doc in bm25_results:
            rrf_contrib = bm25_weight / (k + doc.bm25_rank)
            if doc.chunk_id in doc_scores:
                existing_score, existing_doc = doc_scores[doc.chunk_id]
                existing_doc.bm25_score = doc.bm25_score
                existing_doc.bm25_rank = doc.bm25_rank
                doc_scores[doc.chunk_id] = (existing_score + rrf_contrib, existing_doc)
            else:
                doc.rrf_score = rrf_contrib
                doc_scores[doc.chunk_id] = (rrf_contrib, doc)

        # Sort by RRF score and return
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )

        # Update RRF scores in documents
        result = []
        for score, doc in sorted_docs:
            doc.rrf_score = score
            result.append(doc)

        return result

    async def search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        enable_bm25: bool = True,
    ) -> Tuple[List[RetrievedDoc], Dict[str, Any]]:
        """
        Perform hybrid search with RRF fusion.

        Args:
            query: Search query
            limit: Maximum final results
            vector_weight: Weight for vector search in RRF
            bm25_weight: Weight for BM25 search in RRF
            filters: Optional metadata filters
            min_score: Minimum vector similarity threshold
            enable_bm25: Whether to include BM25 search

        Returns:
            Tuple of (results, stats)
        """
        start_time = time.time()
        stats = {
            "vector_candidates": 0,
            "bm25_candidates": 0,
            "total_unique": 0,
            "vector_search_ms": 0,
            "bm25_search_ms": 0,
            "fusion_ms": 0,
            "total_time_ms": 0,
            "bm25_index_docs": len(self._bm25_doc_ids) if self._bm25_index else 0,
        }

        # Fetch more candidates for fusion
        candidate_limit = limit * 3

        # Build BM25 index if needed (or expired) and BM25 is enabled
        if enable_bm25 and BM25_AVAILABLE:
            if self._bm25_index is None or self._is_index_expired():
                await self.build_bm25_index(force_rebuild=self._is_index_expired())

        # Run vector and BM25 search in parallel
        vector_start = time.time()
        vector_task = self._vector_search(query, candidate_limit, filters, min_score)

        if enable_bm25 and BM25_AVAILABLE and self._bm25_index is not None:
            bm25_task = self._bm25_search(query, candidate_limit, filters)
            vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
            stats["bm25_search_ms"] = int((time.time() - vector_start) * 1000) // 2
        else:
            vector_results = await vector_task
            bm25_results = []

        stats["vector_search_ms"] = int((time.time() - vector_start) * 1000)
        stats["vector_candidates"] = len(vector_results)
        stats["bm25_candidates"] = len(bm25_results)

        # Fuse results
        fusion_start = time.time()
        if bm25_results:
            fused_results = self._reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                vector_weight,
                bm25_weight,
            )
        else:
            # No BM25, use vector results directly with RRF score = vector score
            fused_results = vector_results
            for doc in fused_results:
                doc.rrf_score = doc.vector_score

        stats["fusion_ms"] = int((time.time() - fusion_start) * 1000)
        stats["total_unique"] = len(fused_results)
        stats["total_time_ms"] = int((time.time() - start_time) * 1000)

        # Return top-k
        return fused_results[:limit], stats


# Singleton instance
_retriever_instance: Optional[HybridRetriever] = None


async def get_hybrid_retriever(
    mongodb_service: Any,
    embedding_service: Any,
    collection_name: str = "documents",
    persist_index: bool = True,
    index_ttl: int = 3600,
    cache_dir: Optional[str] = None,
) -> HybridRetriever:
    """
    Get or create the global hybrid retriever instance.

    Args:
        mongodb_service: MongoDB service for vector search
        embedding_service: Embedding service for query embedding
        collection_name: Name of the documents collection
        persist_index: Whether to persist BM25 index to disk
        index_ttl: Time-to-live for cached index in seconds (default: 1 hour)
        cache_dir: Directory for index cache (default: python_services/.cache/bm25)

    Returns:
        Configured HybridRetriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever(
            mongodb_service,
            embedding_service,
            collection_name=collection_name,
            persist_index=persist_index,
            index_ttl=index_ttl,
            cache_dir=cache_dir,
        )
        await _retriever_instance.initialize()
    return _retriever_instance


async def close_hybrid_retriever() -> None:
    """Close the global hybrid retriever instance."""
    global _retriever_instance
    _retriever_instance = None
