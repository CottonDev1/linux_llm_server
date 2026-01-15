"""
Document Embedder Service
=========================

Local embedding generation using sentence-transformers.
Provides fast local embedding inference.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton instance
_embedder_instance: Optional["LocalEmbeddingService"] = None
_embedder_lock = asyncio.Lock()


class LocalEmbeddingService:
    """
    Local embedding service using sentence-transformers.

    Generates embeddings locally without external API calls,
    providing faster inference and better retrieval quality.
    """

    DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
    FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_WORKERS = 8  # Increased from 2 for better throughput

    def __init__(self, model_name: str = DEFAULT_MODEL, max_workers: int = DEFAULT_WORKERS):
        """
        Initialize the embedding service.

        Args:
            model_name: HuggingFace model name for embeddings
            max_workers: Number of worker threads for embedding (default: 8)
        """
        self.model_name = model_name
        self._max_workers = max_workers
        self._model: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._initialized = False
        self._embedding_dim: Optional[int] = None

    def _load_model(self):
        """Load the embedding model (lazy loading)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load {self.model_name}: {e}")
                logger.info(f"Falling back to {self.FALLBACK_MODEL}")
                self.model_name = self.FALLBACK_MODEL
                self._model = SentenceTransformer(self.model_name)

            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            self._initialized = True
            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dim={self._embedding_dim})"
            )

    def _embed_sync(self, text: str) -> List[float]:
        """
        Synchronous embedding (runs in thread pool).

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous batch embedding (runs in thread pool).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self._load_model()

        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        return embeddings.tolist()

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            text,
        )

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_batch_sync,
            texts,
        )

    async def embed_with_normalize(
        self,
        text: Union[str, List[str]],
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate normalized embeddings.

        Args:
            text: Single text or list of texts

        Returns:
            Normalized embedding(s)
        """
        if isinstance(text, str):
            embedding = await self.embed(text)
            return self._normalize(embedding)
        else:
            embeddings = await self.embed_batch(text)
            return [self._normalize(e) for e in embeddings]

    def _normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        return self._embedding_dim

    def health_check(self) -> dict:
        """Check service health."""
        return {
            "service": "LocalEmbeddingService",
            "model": self.model_name,
            "initialized": self._initialized,
            "embedding_dim": self._embedding_dim,
            "max_workers": self._max_workers,
            "status": "healthy" if self._initialized else "not_loaded",
        }

    def preload(self):
        """Preload the model (call during startup)."""
        self._load_model()


async def get_embedding_service(
    model_name: str = LocalEmbeddingService.DEFAULT_MODEL,
    max_workers: int = LocalEmbeddingService.DEFAULT_WORKERS,
) -> LocalEmbeddingService:
    """
    Get the singleton embedding service instance.

    Args:
        model_name: Model to use (only applies on first call)
        max_workers: Number of worker threads (only applies on first call)

    Returns:
        LocalEmbeddingService instance
    """
    global _embedder_instance

    async with _embedder_lock:
        if _embedder_instance is None:
            _embedder_instance = LocalEmbeddingService(model_name, max_workers)
        return _embedder_instance
