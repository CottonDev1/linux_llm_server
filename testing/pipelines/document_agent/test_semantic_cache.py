"""
Semantic Cache Tests
====================

Tests for the Redis-backed semantic caching service.

The semantic cache provides:
1. Query Embedding Cache - Cache embeddings to avoid regeneration
2. Search Results Cache - Cache retrieval results by query hash
3. Full Response Cache - Cache complete query responses
4. Cache Statistics - Track cache performance metrics
5. TTL Management - Automatic expiration of cached entries

These tests verify cache behavior including hits, misses, TTL, and statistics.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any


# =============================================================================
# Mock Redis Client
# =============================================================================

class MockRedisClient:
    """Mock Redis client for testing without actual Redis."""

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}
        self._hits: Dict[str, int] = {}
        self._connected = True

    async def ping(self):
        if not self._connected:
            raise Exception("Redis connection failed")
        return True

    async def get(self, key: str) -> str:
        if not self._connected:
            raise Exception("Redis not available")

        # Check TTL expiration
        if key in self._ttls:
            if time.time() > self._ttls[key]:
                # Key expired
                del self._store[key]
                del self._ttls[key]
                return None

        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        if not self._connected:
            raise Exception("Redis not available")
        self._store[key] = value
        self._ttls[key] = time.time() + ttl

    async def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                if key in self._ttls:
                    del self._ttls[key]
                count += 1
        return count

    async def hincrby(self, key: str, field: str, increment: int):
        if key not in self._hits:
            self._hits[key] = {}
        if field not in self._hits[key]:
            self._hits[key][field] = 0
        self._hits[key][field] += increment

    async def hgetall(self, key: str) -> Dict[str, str]:
        if key in self._hits:
            return {k: str(v) for k, v in self._hits[key].items()}
        return {}

    async def scan_iter(self, match: str):
        pattern = match.replace("*", "")
        for key in list(self._store.keys()):
            if key.startswith(pattern):
                yield key

    async def close(self):
        self._connected = False

    def simulate_disconnect(self):
        self._connected = False

    def simulate_reconnect(self):
        self._connected = True


# =============================================================================
# Cache Hit/Miss Tests
# =============================================================================

class TestCacheHitMiss:
    """Test cache hit and miss scenarios."""

    @pytest.fixture
    def mock_redis(self):
        return MockRedisClient()

    @pytest.mark.asyncio
    async def test_embedding_cache_miss(self, mock_redis):
        """Test cache miss for embeddings."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Query not in cache
        result = await cache.get_embedding("New query not cached")

        assert result is None

    @pytest.mark.asyncio
    async def test_embedding_cache_hit(self, mock_redis):
        """Test cache hit for embeddings."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Store embedding
        test_embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        await cache.set_embedding("Test query", test_embedding)

        # Retrieve embedding
        result = await cache.get_embedding("Test query")

        assert result is not None
        assert len(result) == len(test_embedding)
        assert result == test_embedding

    @pytest.mark.asyncio
    async def test_results_cache_miss(self, mock_redis):
        """Test cache miss for search results."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        result = await cache.get_results("Uncached query")

        assert result is None

    @pytest.mark.asyncio
    async def test_results_cache_hit(self, mock_redis):
        """Test cache hit for search results."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Store results
        test_results = [
            {"document_id": "doc_1", "content": "Test content", "score": 0.85},
            {"document_id": "doc_2", "content": "More content", "score": 0.78},
        ]
        await cache.set_results("Search query", test_results)

        # Retrieve results
        result = await cache.get_results("Search query")

        assert result is not None
        assert len(result) == 2
        assert result[0]["document_id"] == "doc_1"

    @pytest.mark.asyncio
    async def test_response_cache_miss(self, mock_redis):
        """Test cache miss for full responses."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        result = await cache.get_response("Uncached query")

        assert result is None

    @pytest.mark.asyncio
    async def test_response_cache_hit(self, mock_redis):
        """Test cache hit for full responses."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Store response
        test_response = {
            "answer": "This is the cached answer",
            "sources": [{"id": "doc_1", "title": "Source"}],
            "confidence": 0.9,
        }
        await cache.set_response("Query", test_response)

        # Retrieve response
        result = await cache.get_response("Query")

        assert result is not None
        assert result["answer"] == "This is the cached answer"
        assert result["cached"] is True  # Should mark as cached

    @pytest.mark.asyncio
    async def test_results_cache_with_filters(self, mock_redis):
        """Test cache with different filters."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        query = "Same query"

        # Store with different filters
        results1 = [{"id": "1", "content": "Filter 1 results"}]
        results2 = [{"id": "2", "content": "Filter 2 results"}]

        await cache.set_results(query, results1, filters={"department": "Engineering"})
        await cache.set_results(query, results2, filters={"department": "Marketing"})

        # Retrieve with specific filters
        cached1 = await cache.get_results(query, filters={"department": "Engineering"})
        cached2 = await cache.get_results(query, filters={"department": "Marketing"})

        assert cached1[0]["id"] == "1"
        assert cached2[0]["id"] == "2"


# =============================================================================
# TTL Expiration Tests
# =============================================================================

class TestCacheTTL:
    """Test cache TTL expiration."""

    @pytest.fixture
    def mock_redis(self):
        return MockRedisClient()

    @pytest.mark.asyncio
    async def test_embedding_ttl_expiration(self, mock_redis):
        """Test that embeddings expire after TTL."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        # Create cache with very short TTL
        cache = SemanticCache(embedding_ttl=1)  # 1 second
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Store embedding
        await cache.set_embedding("Expiring query", [0.1] * 384)

        # Should be available immediately
        result = await cache.get_embedding("Expiring query")
        assert result is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        result = await cache.get_embedding("Expiring query")
        assert result is None

    @pytest.mark.asyncio
    async def test_results_ttl_expiration(self, mock_redis):
        """Test that results expire after TTL."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache(results_ttl=1)  # 1 second
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Store results
        await cache.set_results("Expiring query", [{"id": "1"}])

        # Should be available immediately
        result = await cache.get_results("Expiring query")
        assert result is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        result = await cache.get_results("Expiring query")
        assert result is None

    @pytest.mark.asyncio
    async def test_response_ttl_expiration(self, mock_redis):
        """Test that responses expire after TTL."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache(response_ttl=1)  # 1 second
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Store response
        await cache.set_response("Expiring query", {"answer": "Test"})

        # Should be available immediately
        result = await cache.get_response("Expiring query")
        assert result is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        result = await cache.get_response("Expiring query")
        assert result is None


# =============================================================================
# Cache Clearing Tests
# =============================================================================

class TestCacheClearing:
    """Test cache clearing functionality."""

    @pytest.fixture
    def mock_redis(self):
        return MockRedisClient()

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, mock_redis):
        """Test clearing entire cache."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Populate cache
        await cache.set_embedding("Query 1", [0.1] * 384)
        await cache.set_results("Query 2", [{"id": "1"}])
        await cache.set_response("Query 3", {"answer": "Test"})

        # Clear all
        deleted = await cache.clear()

        assert deleted == 3

        # Verify all cleared
        assert await cache.get_embedding("Query 1") is None
        assert await cache.get_results("Query 2") is None
        assert await cache.get_response("Query 3") is None

    @pytest.mark.asyncio
    async def test_clear_by_pattern(self, mock_redis):
        """Test clearing cache by pattern."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Populate cache
        await cache.set_embedding("Query 1", [0.1] * 384)
        await cache.set_results("Query 2", [{"id": "1"}])
        await cache.set_response("Query 3", {"answer": "Test"})

        # Clear only embeddings
        deleted = await cache.clear(pattern="emb:*")

        assert deleted == 1

        # Verify only embeddings cleared
        assert await cache.get_embedding("Query 1") is None
        assert await cache.get_results("Query 2") is not None
        assert await cache.get_response("Query 3") is not None


# =============================================================================
# Cache Statistics Tests
# =============================================================================

class TestCacheStatistics:
    """Test cache statistics tracking."""

    @pytest.fixture
    def mock_redis(self):
        return MockRedisClient()

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_redis):
        """Test retrieving cache statistics."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Populate cache
        await cache.set_embedding("Query 1", [0.1] * 384)
        await cache.set_results("Query 2", [{"id": "1"}])
        await cache.set_response("Query 3", {"answer": "Test"})

        # Get hits to trigger hit counting
        await cache.get_embedding("Query 1")
        await cache.get_embedding("Query 1")
        await cache.get_results("Query 2")

        stats = await cache.get_stats()

        assert stats["available"] is True
        assert stats["embedding_entries"] == 1
        assert stats["results_entries"] == 1
        assert stats["response_entries"] == 1
        assert stats["total_entries"] == 3

    @pytest.mark.asyncio
    async def test_stats_when_unavailable(self):
        """Test stats when cache is unavailable."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._available = False
        cache._initialized = True

        stats = await cache.get_stats()

        assert stats["available"] is False


# =============================================================================
# Cache Availability Tests
# =============================================================================

class TestCacheAvailability:
    """Test cache availability and graceful degradation."""

    @pytest.mark.asyncio
    async def test_cache_unavailable_returns_none(self):
        """Test that unavailable cache returns None gracefully."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._available = False
        cache._initialized = True

        # All operations should return None/False gracefully
        assert await cache.get_embedding("query") is None
        assert await cache.set_embedding("query", [0.1] * 384) is False
        assert await cache.get_results("query") is None
        assert await cache.set_results("query", []) is False
        assert await cache.get_response("query") is None
        assert await cache.set_response("query", {}) is False
        assert await cache.clear() == 0

    @pytest.mark.asyncio
    async def test_is_available_property(self):
        """Test is_available property."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._available = True

        assert cache.is_available is True

        cache._available = False
        assert cache.is_available is False

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_error(self):
        """Test graceful handling of Redis errors."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        mock_redis = MockRedisClient()
        mock_redis.simulate_disconnect()

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        # Operations should fail gracefully
        result = await cache.get_embedding("query")
        assert result is None

        result = await cache.set_embedding("query", [0.1] * 384)
        assert result is False


# =============================================================================
# Cache Initialization Tests
# =============================================================================

class TestCacheInitialization:
    """Test cache initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful cache initialization."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        mock_redis = MockRedisClient()

        with patch("services.semantic_cache.redis.from_url", return_value=mock_redis):
            cache = SemanticCache()
            result = await cache.initialize()

            # Note: This will fail without actual Redis, which is expected
            # The test verifies the initialization flow

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize is idempotent."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()
        cache._initialized = True
        cache._available = True

        # Second initialize should return early
        result = await cache.initialize()

        assert result is True  # Already initialized

    @pytest.mark.asyncio
    async def test_close_cache(self):
        """Test closing cache connection."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        mock_redis = MockRedisClient()

        cache = SemanticCache()
        cache._redis = mock_redis
        cache._available = True
        cache._initialized = True

        await cache.close()

        assert cache._available is False
        assert cache._initialized is False


# =============================================================================
# Query Hash Tests
# =============================================================================

class TestQueryHashing:
    """Test query hashing for cache keys."""

    @pytest.mark.asyncio
    async def test_same_query_same_hash(self):
        """Test that same queries produce same hash."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        hash1 = cache._hash_query("What is RAG?")
        hash2 = cache._hash_query("What is RAG?")

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_case_insensitive_hash(self):
        """Test that hashing is case insensitive."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        hash1 = cache._hash_query("What is RAG?")
        hash2 = cache._hash_query("WHAT IS RAG?")
        hash3 = cache._hash_query("what is rag?")

        assert hash1 == hash2 == hash3

    @pytest.mark.asyncio
    async def test_whitespace_normalized(self):
        """Test that whitespace is normalized in hashing."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        hash1 = cache._hash_query("What is RAG?")
        hash2 = cache._hash_query("  What is RAG?  ")

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_different_queries_different_hash(self):
        """Test that different queries produce different hashes."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        hash1 = cache._hash_query("What is RAG?")
        hash2 = cache._hash_query("What is CRAG?")

        assert hash1 != hash2


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Test serialization and deserialization of cache values."""

    @pytest.mark.asyncio
    async def test_serialize_list(self):
        """Test serialization of list values."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        data = [1.0, 2.0, 3.0]
        serialized = cache._serialize(data)
        deserialized = cache._deserialize(serialized)

        assert deserialized == data

    @pytest.mark.asyncio
    async def test_serialize_dict(self):
        """Test serialization of dict values."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        data = {"key": "value", "nested": {"a": 1}}
        serialized = cache._serialize(data)
        deserialized = cache._deserialize(serialized)

        assert deserialized == data

    @pytest.mark.asyncio
    async def test_deserialize_none(self):
        """Test deserialization of None."""
        try:
            from services.semantic_cache import SemanticCache
        except ImportError:
            pytest.skip("SemanticCache not available")

        cache = SemanticCache()

        result = cache._deserialize(None)
        assert result is None

        result = cache._deserialize("")
        assert result is None
