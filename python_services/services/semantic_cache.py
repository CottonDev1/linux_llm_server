"""
Semantic Cache Service
======================

Redis-backed semantic caching for document retrieval pipeline.

Features:
1. Query embedding cache - Avoid regenerating embeddings
2. Search results cache - Cache retrieval results by query hash
3. Response cache - Cache full responses for identical queries
4. Semantic similarity matching - Find similar cached queries

Architecture:
- Uses Redis as backing store
- Configurable TTL for different cache types
- Async operations for non-blocking I/O
- Graceful degradation if Redis unavailable
"""

import logging
import time
import hashlib
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)

# Redis library
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis library not installed. Semantic caching will be disabled. Install with: pip install redis")


@dataclass
class CacheEntry:
    """A cached entry with metadata."""
    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    hit_count: int = 0
    query_hash: Optional[str] = None


class SemanticCache:
    """
    Redis-backed semantic cache for RAG pipeline.

    Usage:
        cache = SemanticCache(redis_url="redis://localhost:6379")
        await cache.initialize()

        # Cache query embedding
        await cache.set_embedding("How do I configure X?", [0.1, 0.2, ...])
        embedding = await cache.get_embedding("How do I configure X?")

        # Cache search results
        await cache.set_results("How do I configure X?", results)
        cached_results = await cache.get_results("How do I configure X?")

        # Cache full response
        await cache.set_response("How do I configure X?", response_dict)
        cached_response = await cache.get_response("How do I configure X?")
    """

    # Cache key prefixes
    PREFIX_EMBEDDING = "emb:"
    PREFIX_RESULTS = "res:"
    PREFIX_RESPONSE = "resp:"
    PREFIX_STATS = "stats:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        embedding_ttl: int = 3600,  # 1 hour
        results_ttl: int = 1800,    # 30 minutes
        response_ttl: int = 900,    # 15 minutes
        max_connections: int = 10,
    ):
        """
        Initialize semantic cache.

        Args:
            redis_url: Redis connection URL
            db: Redis database number
            embedding_ttl: TTL for cached embeddings (seconds)
            results_ttl: TTL for cached search results (seconds)
            response_ttl: TTL for cached full responses (seconds)
            max_connections: Maximum Redis connections
        """
        self._redis_url = redis_url
        self._db = db
        self._embedding_ttl = embedding_ttl
        self._results_ttl = results_ttl
        self._response_ttl = response_ttl
        self._max_connections = max_connections

        self._redis: Optional[redis.Redis] = None
        self._initialized = False
        self._available = False

    async def initialize(self) -> bool:
        """
        Initialize Redis connection.

        Returns:
            True if Redis is available, False otherwise
        """
        if self._initialized:
            return self._available

        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, caching disabled")
            self._initialized = True
            self._available = False
            return False

        try:
            self._redis = redis.from_url(
                self._redis_url,
                db=self._db,
                decode_responses=True,
                max_connections=self._max_connections,
            )

            # Test connection
            await self._redis.ping()
            self._available = True
            self._initialized = True
            logger.info(f"Semantic cache initialized (Redis at {self._redis_url})")
            return True

        except Exception as e:
            logger.warning(f"Redis not available: {e}. Caching disabled.")
            self._initialized = True
            self._available = False
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        self._available = False
        self._initialized = False

    def _hash_query(self, query: str) -> str:
        """Generate consistent hash for a query string."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _serialize(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        return json.dumps(value, default=str)

    def _deserialize(self, value: str) -> Any:
        """Deserialize value from Redis storage."""
        return json.loads(value) if value else None

    # ==========================================================================
    # Embedding Cache
    # ==========================================================================

    async def get_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get cached embedding for a query.

        Args:
            query: Query string

        Returns:
            Cached embedding vector or None if not cached
        """
        if not self._available:
            return None

        try:
            key = f"{self.PREFIX_EMBEDDING}{self._hash_query(query)}"
            cached = await self._redis.get(key)
            if cached:
                await self._increment_hits(key)
                return self._deserialize(cached)
            return None
        except Exception as e:
            logger.warning(f"Error getting cached embedding: {e}")
            return None

    async def set_embedding(self, query: str, embedding: List[float]) -> bool:
        """
        Cache an embedding for a query.

        Args:
            query: Query string
            embedding: Embedding vector

        Returns:
            True if cached successfully
        """
        if not self._available:
            return False

        try:
            key = f"{self.PREFIX_EMBEDDING}{self._hash_query(query)}"
            await self._redis.setex(
                key,
                self._embedding_ttl,
                self._serialize(embedding)
            )
            return True
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
            return False

    # ==========================================================================
    # Search Results Cache
    # ==========================================================================

    async def get_results(self, query: str, filters: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Get cached search results for a query.

        Args:
            query: Query string
            filters: Optional filters (included in cache key)

        Returns:
            Cached results or None if not cached
        """
        if not self._available:
            return None

        try:
            filter_hash = self._hash_query(json.dumps(filters or {}, sort_keys=True))
            key = f"{self.PREFIX_RESULTS}{self._hash_query(query)}:{filter_hash}"
            cached = await self._redis.get(key)
            if cached:
                await self._increment_hits(key)
                return self._deserialize(cached)
            return None
        except Exception as e:
            logger.warning(f"Error getting cached results: {e}")
            return None

    async def set_results(
        self,
        query: str,
        results: List[Dict],
        filters: Optional[Dict] = None
    ) -> bool:
        """
        Cache search results for a query.

        Args:
            query: Query string
            results: Search results to cache
            filters: Optional filters (included in cache key)

        Returns:
            True if cached successfully
        """
        if not self._available:
            return False

        try:
            filter_hash = self._hash_query(json.dumps(filters or {}, sort_keys=True))
            key = f"{self.PREFIX_RESULTS}{self._hash_query(query)}:{filter_hash}"
            await self._redis.setex(
                key,
                self._results_ttl,
                self._serialize(results)
            )
            return True
        except Exception as e:
            logger.warning(f"Error caching results: {e}")
            return False

    # ==========================================================================
    # Full Response Cache
    # ==========================================================================

    async def get_response(self, query: str, context_hash: Optional[str] = None) -> Optional[Dict]:
        """
        Get cached full response for a query.

        Args:
            query: Query string
            context_hash: Optional hash of conversation context

        Returns:
            Cached response or None if not cached
        """
        if not self._available:
            return None

        try:
            ctx = context_hash or "none"
            key = f"{self.PREFIX_RESPONSE}{self._hash_query(query)}:{ctx}"
            cached = await self._redis.get(key)
            if cached:
                await self._increment_hits(key)
                result = self._deserialize(cached)
                if result:
                    result["cached"] = True
                return result
            return None
        except Exception as e:
            logger.warning(f"Error getting cached response: {e}")
            return None

    async def set_response(
        self,
        query: str,
        response: Dict,
        context_hash: Optional[str] = None
    ) -> bool:
        """
        Cache full response for a query.

        Args:
            query: Query string
            response: Full response dictionary
            context_hash: Optional hash of conversation context

        Returns:
            True if cached successfully
        """
        if not self._available:
            return False

        try:
            ctx = context_hash or "none"
            key = f"{self.PREFIX_RESPONSE}{self._hash_query(query)}:{ctx}"
            await self._redis.setex(
                key,
                self._response_ttl,
                self._serialize(response)
            )
            return True
        except Exception as e:
            logger.warning(f"Error caching response: {e}")
            return False

    # ==========================================================================
    # Cache Statistics
    # ==========================================================================

    async def _increment_hits(self, key: str) -> None:
        """Increment hit counter for a cache key."""
        try:
            stats_key = f"{self.PREFIX_STATS}hits"
            await self._redis.hincrby(stats_key, key, 1)
        except Exception:
            pass  # Silently ignore stats errors

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self._available:
            return {"available": False}

        try:
            stats_key = f"{self.PREFIX_STATS}hits"
            hits = await self._redis.hgetall(stats_key)

            # Count keys by type
            embedding_count = 0
            results_count = 0
            response_count = 0

            async for key in self._redis.scan_iter(match=f"{self.PREFIX_EMBEDDING}*"):
                embedding_count += 1
            async for key in self._redis.scan_iter(match=f"{self.PREFIX_RESULTS}*"):
                results_count += 1
            async for key in self._redis.scan_iter(match=f"{self.PREFIX_RESPONSE}*"):
                response_count += 1

            total_hits = sum(int(v) for v in hits.values()) if hits else 0

            return {
                "available": True,
                "embedding_entries": embedding_count,
                "results_entries": results_count,
                "response_entries": response_count,
                "total_entries": embedding_count + results_count + response_count,
                "total_hits": total_hits,
                "embedding_ttl_seconds": self._embedding_ttl,
                "results_ttl_seconds": self._results_ttl,
                "response_ttl_seconds": self._response_ttl,
            }

        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"available": True, "error": str(e)}

    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match (e.g., "emb:*" for embeddings only)

        Returns:
            Number of keys deleted
        """
        if not self._available:
            return 0

        try:
            if pattern:
                keys = [key async for key in self._redis.scan_iter(match=pattern)]
            else:
                # Clear all cache keys
                patterns = [
                    f"{self.PREFIX_EMBEDDING}*",
                    f"{self.PREFIX_RESULTS}*",
                    f"{self.PREFIX_RESPONSE}*",
                ]
                keys = []
                for p in patterns:
                    keys.extend([key async for key in self._redis.scan_iter(match=p)])

            if keys:
                deleted = await self._redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return deleted
            return 0

        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
            return 0

    @property
    def is_available(self) -> bool:
        """Check if cache is available."""
        return self._available


# Singleton instance
_cache_instance: Optional[SemanticCache] = None


async def get_semantic_cache(
    redis_url: str = "redis://localhost:6379",
) -> SemanticCache:
    """Get or create the global semantic cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(redis_url=redis_url)
        await _cache_instance.initialize()
    return _cache_instance


async def close_semantic_cache() -> None:
    """Close the global semantic cache instance."""
    global _cache_instance
    if _cache_instance:
        await _cache_instance.close()
        _cache_instance = None
