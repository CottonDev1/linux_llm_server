"""
Document Reranker Service
=========================

Cross-encoder re-ranking for improved document retrieval relevance.
Uses ms-marco-MiniLM model to score (query, document) pairs.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Singleton instance
_reranker_instance: Optional["DocumentRerankerService"] = None
_reranker_lock = asyncio.Lock()


class DocumentRerankerService:
    """
    Cross-encoder re-ranking service for document retrieval.

    Re-ranks initial bi-encoder results using a cross-encoder model
    that scores each (query, document) pair for true relevance.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the reranker service.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False

    def _load_model(self):
        """Load the cross-encoder model (lazy loading)."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info("Cross-encoder model loaded successfully")

    def _rerank_sync(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Synchronous re-ranking (runs in thread pool).

        Args:
            query: Search query
            documents: List of document texts to re-rank
            top_k: Number of top results to return

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        self._load_model()

        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Pair indices with scores and sort
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to re-rank
            top_k: Number of top results to return

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._rerank_sync,
            query,
            documents,
            top_k,
        )

    async def rerank_with_metadata(
        self,
        query: str,
        documents: List[dict],
        content_key: str = "content",
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        Re-rank documents while preserving metadata.

        Args:
            query: Search query
            documents: List of document dicts with content and metadata
            content_key: Key to extract text content from documents
            top_k: Number of top results to return

        Returns:
            List of (document_dict, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Extract text content
        texts = [doc.get(content_key, "") for doc in documents]

        # Get ranked indices
        ranked_indices = await self.rerank(query, texts, top_k)

        # Map back to original documents
        return [(documents[idx], score) for idx, score in ranked_indices]

    def health_check(self) -> dict:
        """Check service health."""
        return {
            "service": "DocumentRerankerService",
            "model": self.model_name,
            "initialized": self._initialized,
            "status": "healthy" if self._initialized else "not_loaded",
        }

    def preload(self):
        """Preload the model (call during startup)."""
        self._load_model()


async def get_reranker_service(
    model_name: str = DocumentRerankerService.DEFAULT_MODEL,
) -> DocumentRerankerService:
    """
    Get the singleton reranker service instance.

    Args:
        model_name: Model to use (only applies on first call)

    Returns:
        DocumentRerankerService instance
    """
    global _reranker_instance

    async with _reranker_lock:
        if _reranker_instance is None:
            _reranker_instance = DocumentRerankerService(model_name)
        return _reranker_instance
