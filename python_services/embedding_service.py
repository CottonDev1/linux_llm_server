"""
Embedding Service - Generates vector embeddings using sentence-transformers or llama.cpp

Supports three modes:
1. Local mode: Loads SentenceTransformer model locally (~500 MB RAM)
2. Remote sentence-transformers API: Uses EMBEDDING_SERVICE_URL with /embed endpoint
3. Remote llama.cpp API: Uses EMBEDDING_SERVICE_URL with /embedding endpoint (for nomic, bge, etc.)
"""
import asyncio
import hashlib
import logging
from typing import List, Optional
import numpy as np

import aiohttp
from aiohttp import ClientTimeout

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_SERVICE_URL
from core.log_utils import log_info

logger = logging.getLogger(__name__)

# Models that use llama.cpp API format (/embedding with {"content": ...})
LLAMACPP_MODELS = {"nomic-embed-text-v1.5", "bge-small-en-v1.5", "mxbai-embed-large-v1"}


class EmbeddingService:
    """
    Service for generating text embeddings.
    Uses remote API if EMBEDDING_SERVICE_URL is configured, otherwise loads model locally.
    """

    _instance: Optional['EmbeddingService'] = None

    def __init__(self):
        self.model = None  # Only loaded in local mode
        self.model_name = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        self._cache = {}
        self._cache_max_size = 5000
        self.is_initialized = False
        self.use_remote = bool(EMBEDDING_SERVICE_URL)
        self.remote_url = EMBEDDING_SERVICE_URL.rstrip('/') if EMBEDDING_SERVICE_URL else None
        self._session: Optional[aiohttp.ClientSession] = None
        # Detect if using llama.cpp API format
        self.use_llamacpp_api = self.model_name in LLAMACPP_MODELS

    @classmethod
    def get_instance(cls) -> 'EmbeddingService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize the embedding service"""
        if self.is_initialized:
            return

        if self.use_remote:
            api_type = "llama.cpp" if self.use_llamacpp_api else "sentence-transformers"
            log_info("Embedding Service", f"Using remote {api_type} API at {self.remote_url}")
            self._session = aiohttp.ClientSession(timeout=ClientTimeout(total=30))
            # Test connection
            try:
                async with self._session.get(f"{self.remote_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        log_info("Embedding Service", f"Remote service healthy: {data.get('model', data.get('status', 'unknown'))}")
                    else:
                        raise ConnectionError(f"Remote service returned {response.status}")
            except Exception as e:
                # Close the session if initialization fails to avoid "Unclosed client session" warnings
                if self._session and not self._session.closed:
                    await self._session.close()
                    await asyncio.sleep(0.25)  # Give event loop time to finalize
                    self._session = None
                logger.error(f"Failed to connect to remote embedding service: {e}")
                raise
        else:
            log_info("Embedding Service", f"Loading local model: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            log_info("Embedding Service", "Local model loaded successfully")

        self.is_initialized = True

    async def close(self):
        """Close HTTP session with proper cleanup for asyncio"""
        if self._session and not self._session.closed:
            await self._session.close()
            # Give the event loop time to finalize the connection cleanup
            # This prevents "Unclosed client session" warnings
            await asyncio.sleep(0.25)
            self._session = None
            log_info("Embedding Service", "HTTP session closed")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        Uses caching for repeated texts.
        """
        if not self.is_initialized:
            await self.initialize()

        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        if self.use_remote:
            embedding_list = await self._remote_embed(text)
        else:
            embedding = self.model.encode(text, normalize_embeddings=True)
            embedding_list = embedding.tolist()

        # Cache management
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = embedding_list
        return embedding_list

    async def _remote_embed(self, text: str) -> List[float]:
        """Call remote embedding API for single text"""
        if self.use_llamacpp_api:
            # llama.cpp API format: POST /embedding {"content": "..."}
            # Response: [{"index": 0, "embedding": [[...floats...]]}]
            endpoint = f"{self.remote_url}/embedding"
            payload = {"content": text}
        else:
            # sentence-transformers API format
            endpoint = f"{self.remote_url}/embed"
            payload = {"text": text}

        async with self._session.post(endpoint, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"Remote embed failed: {error}")
            data = await response.json()

            if self.use_llamacpp_api:
                # llama.cpp returns array of objects with nested embedding
                # Extract: data[0]["embedding"][0]
                return data[0]["embedding"][0]
            else:
                return data["embedding"]

    async def _remote_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Call remote embedding API for batch of texts"""
        if self.use_llamacpp_api:
            # llama.cpp doesn't have batch endpoint - call single embeddings concurrently
            semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

            async def embed_one(text: str) -> List[float]:
                async with semaphore:
                    return await self._remote_embed(text)

            embeddings = await asyncio.gather(*[embed_one(t) for t in texts])
            return list(embeddings)
        else:
            # sentence-transformers batch API
            async with self._session.post(
                f"{self.remote_url}/embed/batch",
                json={"texts": texts}
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"Remote batch embed failed: {error}")
                data = await response.json()
                return data["embeddings"]

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        More efficient than calling generate_embedding multiple times.
        """
        if not self.is_initialized:
            await self.initialize()

        if not texts:
            return []

        # Separate cached and uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Generate embeddings for uncached texts
        if uncached_texts:
            if self.use_remote:
                embeddings = await self._remote_embed_batch(uncached_texts)
            else:
                embeddings_np = self.model.encode(uncached_texts, normalize_embeddings=True)
                embeddings = [e.tolist() for e in embeddings_np]

            for idx, embedding_list in zip(uncached_indices, embeddings):
                results[idx] = embedding_list

                # Cache the result
                cache_key = self._get_cache_key(texts[idx])
                if len(self._cache) < self._cache_max_size:
                    self._cache[cache_key] = embedding_list

        return results

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        Returns value between 0 and 1 (1 = identical).
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def l2_distance_to_similarity(self, distance: float) -> float:
        """
        Convert L2 distance to similarity score.
        Similarity = 1 / (1 + distance)
        """
        return 1.0 / (1.0 + distance)

    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Get current cache size"""
        return len(self._cache)

    @property
    def mode(self) -> str:
        """Get current mode (remote or local)"""
        return "remote" if self.use_remote else "local"


# Global instance getter
def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance"""
    return EmbeddingService.get_instance()
