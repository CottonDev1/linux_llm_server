"""
Response Cache Service
======================

LRU cache with TTL for RAG query responses.

Design Rationale:
-----------------
Caching RAG responses provides significant performance benefits:
1. Reduces LLM API calls for repeated questions
2. Provides instant responses for common queries
3. Reduces load on vector search backend

The cache uses a simple LRU (Least Recently Used) eviction policy
with TTL (Time To Live) for freshness. This balances memory usage
with hit rate.

Cache Key Strategy:
- Query text (normalized)
- Project scope
- Limit parameter
- Include EWRLibrary flag

Best Practices for RAG Caching:
-------------------------------
1. Short TTL (5 minutes default) to handle dynamic content
2. Cache full responses, not just LLM output (includes sources)
3. Normalize queries before hashing (lowercase, strip whitespace)
4. Size limit to prevent memory exhaustion
5. Thread-safe access for concurrent requests

Cache Invalidation:
- TTL-based automatic expiration
- Manual invalidation via API endpoint
- LRU eviction when at capacity
"""

import asyncio
import hashlib
import logging
import time
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for response cache."""
    max_size: int = 100          # Maximum cached entries
    ttl_seconds: int = 300       # 5 minutes default
    cleanup_interval: int = 60   # Cleanup check interval


class ResponseCache:
    """
    LRU cache with TTL for RAG query responses.

    This cache stores complete query responses including:
    - Generated answer text
    - Source citations
    - Token usage statistics
    - Timing information

    The cache uses OrderedDict for O(1) LRU operations and
    asyncio.Lock for thread-safe concurrent access.

    Usage:
        cache = ResponseCache()

        # Check cache
        cached = await cache.get(query, cache_key)
        if cached:
            return cached

        # Generate response...
        response = {...}

        # Cache response
        await cache.set(query, cache_key, response, ttl=300)
    """

    _instance: Optional["ResponseCache"] = None

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the cache.

        Args:
            config: Cache configuration (optional)
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

    @classmethod
    def get_instance(cls, config: Optional[CacheConfig] = None) -> "ResponseCache":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _make_key(self, query: str, cache_params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from query and parameters.

        The key is a SHA-256 hash of:
        - Normalized query text
        - Project scope
        - Result limit
        - EWRLibrary inclusion flag

        Args:
            query: Search query text
            cache_params: Additional parameters (project, limit, etc.)

        Returns:
            Hex digest of the cache key
        """
        # Normalize query
        normalized_query = query.lower().strip()

        # Build key components
        key_parts = [
            normalized_query,
            str(cache_params.get("project", "")),
            str(cache_params.get("limit", 10)),
            str(cache_params.get("includeEWRLibrary", False))
        ]

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(
        self,
        query: str,
        cache_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if valid.

        Checks both existence and TTL. Expired entries are removed.
        Valid entries are moved to end of LRU queue.

        Args:
            query: Search query text
            cache_params: Parameters used to build cache key

        Returns:
            Cached response dict or None if not found/expired
        """
        key = self._make_key(query, cache_params)

        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            response, timestamp = self._cache[key]

            # Check TTL
            age = time.time() - timestamp
            if age > self.config.ttl_seconds:
                # Expired - remove and return miss
                del self._cache[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                logger.debug(f"Cache expired for key {key[:8]}... (age: {age:.1f}s)")
                return None

            # Valid hit - move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats["hits"] += 1

            logger.debug(f"Cache hit for key {key[:8]}... (age: {age:.1f}s)")
            return response

    async def set(
        self,
        query: str,
        cache_params: Dict[str, Any],
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache a response with TTL.

        Evicts oldest entries if at capacity (LRU).

        Args:
            query: Search query text
            cache_params: Parameters used to build cache key
            response: Response dict to cache
            ttl: Optional TTL override (uses config default if None)
        """
        key = self._make_key(query, cache_params)
        ttl = ttl or self.config.ttl_seconds

        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.config.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:8]}...")

            # Store with timestamp
            self._cache[key] = (response, time.time())
            self._cache.move_to_end(key)

            logger.debug(f"Cached response for key {key[:8]}... (ttl: {ttl}s)")

    async def invalidate(
        self,
        query: Optional[str] = None,
        cache_params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Invalidate cache entries.

        If query and cache_params provided, invalidates specific entry.
        If neither provided, clears entire cache.

        Args:
            query: Optional query to match
            cache_params: Optional parameters to match

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            if query is not None and cache_params is not None:
                # Invalidate specific entry
                key = self._make_key(query, cache_params)
                if key in self._cache:
                    del self._cache[key]
                    logger.info(f"Invalidated cache entry: {key[:8]}...")
                    return 1
                return 0
            else:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared entire cache ({count} entries)")
                return count

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Called periodically to prevent memory buildup from
        stale entries that were never accessed.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = []

        async with self._lock:
            for key, (_, timestamp) in self._cache.items():
                if now - timestamp > self.config.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats["expirations"] += 1

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hit rate, size, and other metrics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests * 100
            if total_requests > 0 else 0
        )

        return {
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "ttl_seconds": self.config.ttl_seconds,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
            "hit_rate_percent": round(hit_rate, 2)
        }

    async def clear(self) -> int:
        """
        Clear the entire cache.

        Returns:
            Number of entries cleared
        """
        return await self.invalidate()


# Module-level singleton accessor
_response_cache: Optional[ResponseCache] = None


def get_response_cache(config: Optional[CacheConfig] = None) -> ResponseCache:
    """Get or create the global response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(config)
    return _response_cache
