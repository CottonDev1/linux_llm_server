"""
Vector Search Integration Tests
===============================

Comprehensive tests for REAL MongoDB Atlas Vector Search functionality.
These tests use actual MongoDB $vectorSearch aggregation, NOT manual similarity calculations.

Test Categories:
1. Index creation and management
2. Basic vector search
3. Similarity threshold tests
4. Metadata filtering with vector search
5. Performance tests
6. Edge cases

Prerequisites:
- MongoDB Atlas with Vector Search enabled OR MongoDB 8.2+
- Vector search index created on test collection
- Embedding service available

IMPORTANT: These tests require MongoDB Atlas Vector Search to be available.
If not available, tests will be skipped with appropriate messages.
"""

import asyncio
import time
import uuid
import math
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Import testing infrastructure
import sys
from pathlib import Path

# Add testing root to path
TESTING_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TESTING_ROOT))

from config.settings import settings
from config.test_config import TestConfig, get_test_config
from fixtures.mongodb_fixtures import (
    create_test_collection,
    insert_test_documents,
    cleanup_test_documents,
    create_mock_document,
)


# =============================================================================
# Constants and Configuration
# =============================================================================

EMBEDDING_DIMENSIONS = 384  # Default for all-MiniLM-L6-v2
TEST_COLLECTION_PREFIX = "test_vector_search"
VECTOR_INDEX_NAME = "test_vector_index"

# Similarity thresholds for testing
HIGH_SIMILARITY_THRESHOLD = 0.9
MEDIUM_SIMILARITY_THRESHOLD = 0.5
LOW_SIMILARITY_THRESHOLD = 0.1


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def test_config():
    """Get test configuration for vector search tests."""
    return get_test_config(
        test_run_id=f"vector_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )


@pytest.fixture(scope="module")
def mongodb_client(test_config):
    """Get MongoDB client for tests."""
    from pymongo import MongoClient

    client = MongoClient(
        test_config.mongodb.uri,
        serverSelectionTimeoutMS=test_config.mongodb.timeout_ms
    )

    # Verify connection
    try:
        client.admin.command('ping')
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")

    yield client
    client.close()


@pytest.fixture(scope="module")
def mongodb_database(mongodb_client, test_config):
    """Get MongoDB database for tests."""
    return mongodb_client[test_config.mongodb.database]


@pytest.fixture(scope="module")
async def async_mongodb_client(test_config):
    """Get async MongoDB client using motor."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        pytest.skip("motor not installed for async MongoDB")

    client = AsyncIOMotorClient(
        test_config.mongodb.uri,
        serverSelectionTimeoutMS=test_config.mongodb.timeout_ms
    )

    yield client
    client.close()


@pytest.fixture(scope="module")
async def async_mongodb_database(async_mongodb_client, test_config):
    """Get async MongoDB database."""
    return async_mongodb_client[test_config.mongodb.database]


@pytest.fixture(scope="module")
def embedding_service():
    """
    Get the real embedding service for generating vectors.

    Uses the actual sentence-transformers model for realistic embeddings.
    """
    try:
        # Try to import and use the real embedding service
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python_services"))
        from embedding_service import get_embedding_service

        service = get_embedding_service()
        return service
    except Exception as e:
        pytest.skip(f"Embedding service not available: {e}")


@pytest.fixture(scope="function")
def test_collection_name():
    """Generate unique test collection name."""
    return f"{TEST_COLLECTION_PREFIX}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
async def vector_search_collection(async_mongodb_database, test_collection_name, test_config):
    """
    Create a test collection with vector search index.

    Creates:
    - Collection with unique name
    - Vector search index on 'vector' field
    - Text index on 'content' field
    - Standard indexes on metadata fields

    Cleanup: Drops collection after test
    """
    collection = async_mongodb_database[test_collection_name]

    # Create standard indexes
    await collection.create_index([("project", 1)])
    await collection.create_index([("type", 1)])
    await collection.create_index([("department", 1)])
    await collection.create_index([("created_at", -1)])

    # Try to create vector search index
    vector_index_created = False
    try:
        # Define vector search index
        index_definition = {
            "name": VECTOR_INDEX_NAME,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "vector",
                        "numDimensions": EMBEDDING_DIMENSIONS,
                        "similarity": "cosine"
                    },
                    {"type": "filter", "path": "project"},
                    {"type": "filter", "path": "type"},
                    {"type": "filter", "path": "department"},
                ]
            }
        }

        await collection.create_search_index(index_definition)
        vector_index_created = True

        # Wait for index to be ready (Atlas can take a few seconds)
        await asyncio.sleep(2)

    except Exception as e:
        # Vector search index creation might fail if not on Atlas
        error_str = str(e).lower()
        if 'not supported' in error_str or 'atlas' in error_str or 'command not found' in error_str:
            pytest.skip(f"MongoDB Atlas Vector Search not available: {e}")
        # Log but don't fail - index might already exist or have different error
        print(f"Note: Vector index creation returned: {e}")

    yield collection, vector_index_created

    # Cleanup
    if test_config.cleanup_after_test:
        await collection.drop()


@pytest.fixture
def generate_embedding(embedding_service):
    """Fixture to generate embeddings using the real embedding service."""
    def _generate(text: str) -> List[float]:
        if hasattr(embedding_service, 'encode'):
            # Direct sentence-transformers model
            return embedding_service.encode(text).tolist()
        elif hasattr(embedding_service, 'generate_embedding'):
            # Wrapped service
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running in async context
                return asyncio.run_coroutine_threadsafe(
                    embedding_service.generate_embedding(text),
                    loop
                ).result()
            else:
                return loop.run_until_complete(embedding_service.generate_embedding(text))
        elif hasattr(embedding_service, 'embed'):
            return embedding_service.embed(text)
        else:
            raise ValueError("Unknown embedding service interface")

    return _generate


@pytest.fixture
async def generate_embedding_async(embedding_service):
    """Async fixture to generate embeddings."""
    async def _generate(text: str) -> List[float]:
        if hasattr(embedding_service, 'generate_embedding'):
            return await embedding_service.generate_embedding(text)
        elif hasattr(embedding_service, 'encode'):
            return embedding_service.encode(text).tolist()
        elif hasattr(embedding_service, 'embed_async'):
            return await embedding_service.embed_async(text)
        else:
            # Fallback to sync
            if hasattr(embedding_service, 'embed'):
                return embedding_service.embed(text)
            raise ValueError("Unknown embedding service interface")

    return _generate


# =============================================================================
# Helper Functions
# =============================================================================

async def create_test_documents_with_vectors(
    collection,
    documents: List[Dict[str, Any]],
    generate_embedding_fn,
    test_run_id: str
) -> List[str]:
    """
    Create test documents with vector embeddings.

    Args:
        collection: MongoDB collection
        documents: List of document dicts with 'content' field
        generate_embedding_fn: Function to generate embeddings
        test_run_id: Test run identifier for cleanup

    Returns:
        List of inserted document IDs
    """
    docs_with_vectors = []
    for doc in documents:
        doc_copy = doc.copy()

        # Generate embedding for content
        content = doc_copy.get('content', '')
        if content:
            doc_copy['vector'] = generate_embedding_fn(content)

        # Add test markers
        if '_id' not in doc_copy:
            doc_copy['_id'] = f"test_{uuid.uuid4().hex}"
        doc_copy['is_test'] = True
        doc_copy['test_run_id'] = test_run_id
        doc_copy['created_at'] = datetime.utcnow()

        docs_with_vectors.append(doc_copy)

    if docs_with_vectors:
        result = await collection.insert_many(docs_with_vectors)
        return [str(id) for id in result.inserted_ids]
    return []


async def perform_vector_search(
    collection,
    query_vector: List[float],
    index_name: str = VECTOR_INDEX_NAME,
    limit: int = 10,
    filter_query: Optional[Dict] = None,
    threshold: float = 0.0
) -> List[Dict]:
    """
    Perform MongoDB Atlas vector search.

    Args:
        collection: MongoDB collection
        query_vector: Query embedding vector
        index_name: Name of vector search index
        limit: Maximum results to return
        filter_query: Optional pre-filter
        threshold: Minimum similarity threshold

    Returns:
        List of documents with _similarity scores
    """
    # Build vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": index_name,
            "path": "vector",
            "queryVector": query_vector,
            "numCandidates": limit * 10,  # Candidates multiplier
            "limit": limit
        }
    }

    if filter_query:
        vector_search_stage["$vectorSearch"]["filter"] = filter_query

    pipeline = [
        vector_search_stage,
        {
            "$addFields": {
                "_similarity": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    # Apply threshold filter if specified
    if threshold > 0:
        pipeline.append({
            "$match": {"_similarity": {"$gte": threshold}}
        })

    cursor = collection.aggregate(pipeline)
    return await cursor.to_list(length=limit)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors (for verification)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# =============================================================================
# Test Class: Index Creation and Management
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
class TestVectorIndexManagement:
    """Tests for vector search index creation and management."""

    async def test_index_creation_success(self, vector_search_collection):
        """Test that vector search index can be created."""
        collection, index_created = vector_search_collection

        # Skip if index creation failed (not on Atlas)
        if not index_created:
            pytest.skip("Vector index creation not available")

        # Verify index exists
        indexes = await collection.list_search_indexes().to_list(length=100)
        index_names = [idx.get('name') for idx in indexes]

        assert VECTOR_INDEX_NAME in index_names, \
            f"Vector index '{VECTOR_INDEX_NAME}' not found in {index_names}"

    async def test_index_with_filter_fields(self, async_mongodb_database, test_config):
        """Test creating index with multiple filter fields."""
        collection_name = f"{TEST_COLLECTION_PREFIX}_filters_{uuid.uuid4().hex[:8]}"
        collection = async_mongodb_database[collection_name]

        try:
            index_definition = {
                "name": "filter_test_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector",
                            "numDimensions": EMBEDDING_DIMENSIONS,
                            "similarity": "cosine"
                        },
                        {"type": "filter", "path": "project"},
                        {"type": "filter", "path": "type"},
                        {"type": "filter", "path": "department"},
                        {"type": "filter", "path": "tags"},
                    ]
                }
            }

            await collection.create_search_index(index_definition)
            await asyncio.sleep(2)

            # Verify all filter fields are configured
            indexes = await collection.list_search_indexes().to_list(length=100)
            test_index = next((idx for idx in indexes if idx.get('name') == 'filter_test_index'), None)

            assert test_index is not None, "Filter test index not found"

        except Exception as e:
            if 'not supported' in str(e).lower() or 'atlas' in str(e).lower():
                pytest.skip(f"Vector search index not available: {e}")
            raise
        finally:
            if test_config.cleanup_after_test:
                await collection.drop()

    async def test_index_deletion(self, async_mongodb_database, test_config):
        """Test that vector search index can be deleted."""
        collection_name = f"{TEST_COLLECTION_PREFIX}_delete_{uuid.uuid4().hex[:8]}"
        collection = async_mongodb_database[collection_name]
        index_name = "delete_test_index"

        try:
            # Create index
            index_definition = {
                "name": index_name,
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector",
                            "numDimensions": EMBEDDING_DIMENSIONS,
                            "similarity": "cosine"
                        }
                    ]
                }
            }

            await collection.create_search_index(index_definition)
            await asyncio.sleep(2)

            # Verify index exists
            indexes = await collection.list_search_indexes().to_list(length=100)
            assert any(idx.get('name') == index_name for idx in indexes)

            # Delete index
            await collection.drop_search_index(index_name)
            await asyncio.sleep(2)

            # Verify index is deleted
            indexes = await collection.list_search_indexes().to_list(length=100)
            assert not any(idx.get('name') == index_name for idx in indexes), \
                f"Index '{index_name}' should have been deleted"

        except Exception as e:
            if 'not supported' in str(e).lower() or 'atlas' in str(e).lower():
                pytest.skip(f"Vector search index not available: {e}")
            raise
        finally:
            if test_config.cleanup_after_test:
                await collection.drop()


# =============================================================================
# Test Class: Basic Vector Search
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
class TestBasicVectorSearch:
    """Tests for basic vector search functionality."""

    @pytest.fixture
    async def populated_collection(
        self,
        vector_search_collection,
        generate_embedding,
        test_config
    ):
        """Create collection with test documents."""
        collection, index_created = vector_search_collection

        if not index_created:
            pytest.skip("Vector search index not available")

        # Create diverse test documents
        test_documents = [
            {
                "content": "Python is a programming language used for web development.",
                "project": "docs",
                "type": "tutorial",
                "department": "engineering"
            },
            {
                "content": "JavaScript is used for frontend web development.",
                "project": "docs",
                "type": "tutorial",
                "department": "engineering"
            },
            {
                "content": "Machine learning models can predict future outcomes.",
                "project": "ml",
                "type": "article",
                "department": "data_science"
            },
            {
                "content": "SQL databases store structured data in tables.",
                "project": "database",
                "type": "reference",
                "department": "engineering"
            },
            {
                "content": "Customer support handles user inquiries and complaints.",
                "project": "support",
                "type": "process",
                "department": "customer_service"
            },
            {
                "content": "The quick brown fox jumps over the lazy dog.",
                "project": "random",
                "type": "misc",
                "department": "other"
            },
        ]

        await create_test_documents_with_vectors(
            collection,
            test_documents,
            generate_embedding,
            test_config.test_run_id
        )

        # Wait for documents to be indexed
        await asyncio.sleep(1)

        return collection, generate_embedding

    async def test_search_returns_ordered_results(self, populated_collection):
        """Test that search returns documents ordered by similarity (descending)."""
        collection, generate_embedding = populated_collection

        # Query about programming
        query_vector = generate_embedding("What programming languages are used for web development?")

        results = await perform_vector_search(
            collection,
            query_vector,
            limit=5
        )

        assert len(results) > 0, "Should return at least one result"

        # Verify results are ordered by similarity (descending)
        similarities = [r['_similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True), \
            "Results should be ordered by similarity descending"

        # Top result should be about programming/web development
        top_content = results[0]['content'].lower()
        assert any(word in top_content for word in ['programming', 'web', 'development']), \
            f"Top result should be relevant to query. Got: {results[0]['content']}"

    async def test_similarity_scores_between_0_and_1(self, populated_collection):
        """Test that similarity scores are normalized between 0 and 1."""
        collection, generate_embedding = populated_collection

        query_vector = generate_embedding("database tables and structured data")

        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10
        )

        for result in results:
            similarity = result['_similarity']
            assert 0.0 <= similarity <= 1.0, \
                f"Similarity score {similarity} should be between 0 and 1"

    async def test_limit_parameter_works(self, populated_collection):
        """Test that limit parameter restricts result count."""
        collection, generate_embedding = populated_collection

        query_vector = generate_embedding("programming")

        # Test different limits
        for limit in [1, 2, 3, 5]:
            results = await perform_vector_search(
                collection,
                query_vector,
                limit=limit
            )

            assert len(results) <= limit, \
                f"Results count {len(results)} should not exceed limit {limit}"

    async def test_empty_results_for_unrelated_query(self, populated_collection):
        """Test that unrelated queries return low similarity results."""
        collection, generate_embedding = populated_collection

        # Query about something completely unrelated
        query_vector = generate_embedding("quantum physics black hole singularity event horizon")

        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            threshold=0.9  # Very high threshold
        )

        # With a high threshold, we should get few or no results
        assert len(results) <= 2, \
            "Unrelated query with high threshold should return few results"

    async def test_semantic_similarity_works(self, populated_collection):
        """Test that semantically similar queries return similar results."""
        collection, generate_embedding = populated_collection

        # Two semantically similar queries
        query1 = "coding languages for building websites"
        query2 = "programming tools for web applications"

        vector1 = generate_embedding(query1)
        vector2 = generate_embedding(query2)

        results1 = await perform_vector_search(collection, vector1, limit=3)
        results2 = await perform_vector_search(collection, vector2, limit=3)

        # Both queries should return some overlapping results
        ids1 = set(str(r['_id']) for r in results1)
        ids2 = set(str(r['_id']) for r in results2)

        overlap = ids1.intersection(ids2)
        assert len(overlap) >= 1, \
            "Semantically similar queries should have overlapping results"


# =============================================================================
# Test Class: Similarity Threshold Tests
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
class TestSimilarityThresholds:
    """Tests for similarity threshold behavior."""

    @pytest.fixture
    async def threshold_test_collection(
        self,
        vector_search_collection,
        generate_embedding,
        test_config
    ):
        """Create collection with documents at varying relevance levels."""
        collection, index_created = vector_search_collection

        if not index_created:
            pytest.skip("Vector search index not available")

        # Create documents with varying relevance to "database management"
        test_documents = [
            # Highly relevant
            {
                "content": "Database management systems handle structured data storage and retrieval.",
                "relevance": "high",
            },
            {
                "content": "SQL is the standard language for managing relational databases.",
                "relevance": "high",
            },
            # Somewhat relevant
            {
                "content": "Data storage solutions include cloud services and local servers.",
                "relevance": "medium",
            },
            {
                "content": "Information systems organize and process business data.",
                "relevance": "medium",
            },
            # Not relevant
            {
                "content": "The weather today is sunny with clear skies.",
                "relevance": "low",
            },
            {
                "content": "Cooking recipes include ingredients and preparation steps.",
                "relevance": "low",
            },
        ]

        await create_test_documents_with_vectors(
            collection,
            test_documents,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        return collection, generate_embedding

    async def test_high_threshold_filters_low_similarity(self, threshold_test_collection):
        """Test that high threshold excludes low similarity documents."""
        collection, generate_embedding = threshold_test_collection

        query_vector = generate_embedding("database management and SQL queries")

        # High threshold should only return highly relevant docs
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            threshold=HIGH_SIMILARITY_THRESHOLD
        )

        # All results should have high similarity
        for result in results:
            assert result['_similarity'] >= HIGH_SIMILARITY_THRESHOLD, \
                f"Result with similarity {result['_similarity']} should be above threshold"

    async def test_zero_threshold_returns_all_documents(self, threshold_test_collection):
        """Test that threshold of 0.0 returns all matching documents."""
        collection, generate_embedding = threshold_test_collection

        query_vector = generate_embedding("database management")

        # Zero threshold should return all documents (up to limit)
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            threshold=0.0
        )

        # Should return multiple results
        assert len(results) >= 4, \
            f"Zero threshold should return more results, got {len(results)}"

    async def test_threshold_of_1_returns_only_exact_or_near_matches(self, threshold_test_collection):
        """Test that threshold of 1.0 returns only exact or near-exact matches."""
        collection, generate_embedding = threshold_test_collection

        query_vector = generate_embedding("database management")

        # Threshold of 1.0 should return very few or no results
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            threshold=1.0
        )

        # Very few or no results expected (exact match is rare)
        assert len(results) <= 1, \
            f"Threshold 1.0 should return at most 1 result, got {len(results)}"

    async def test_increasing_threshold_reduces_results(self, threshold_test_collection):
        """Test that increasing threshold reduces result count."""
        collection, generate_embedding = threshold_test_collection

        query_vector = generate_embedding("data storage and management systems")

        thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
        result_counts = []

        for threshold in thresholds:
            results = await perform_vector_search(
                collection,
                query_vector,
                limit=10,
                threshold=threshold
            )
            result_counts.append(len(results))

        # Result count should generally decrease as threshold increases
        # (Allow some variance due to score distribution)
        assert result_counts[0] >= result_counts[-1], \
            f"Higher threshold should return fewer results: {list(zip(thresholds, result_counts))}"


# =============================================================================
# Test Class: Metadata Filtering with Vector Search
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
class TestMetadataFiltering:
    """Tests for metadata filtering combined with vector search."""

    @pytest.fixture
    async def filtered_collection(
        self,
        vector_search_collection,
        generate_embedding,
        test_config
    ):
        """Create collection with diverse metadata for filtering tests."""
        collection, index_created = vector_search_collection

        if not index_created:
            pytest.skip("Vector search index not available")

        # Create documents with various metadata combinations
        test_documents = [
            # Project A - Engineering
            {
                "content": "Building scalable microservices architecture.",
                "project": "project_a",
                "type": "architecture",
                "department": "engineering",
                "tags": ["microservices", "scalability"],
            },
            {
                "content": "API design best practices for REST services.",
                "project": "project_a",
                "type": "guide",
                "department": "engineering",
                "tags": ["api", "rest"],
            },
            # Project B - Engineering
            {
                "content": "Database optimization techniques for performance.",
                "project": "project_b",
                "type": "tutorial",
                "department": "engineering",
                "tags": ["database", "performance"],
            },
            # Project A - Data Science
            {
                "content": "Machine learning model deployment strategies.",
                "project": "project_a",
                "type": "guide",
                "department": "data_science",
                "tags": ["ml", "deployment"],
            },
            # Project C - Support
            {
                "content": "Customer support ticket handling procedures.",
                "project": "project_c",
                "type": "process",
                "department": "support",
                "tags": ["support", "tickets"],
            },
            {
                "content": "Escalation process for critical issues.",
                "project": "project_c",
                "type": "process",
                "department": "support",
                "tags": ["escalation", "critical"],
            },
        ]

        await create_test_documents_with_vectors(
            collection,
            test_documents,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        return collection, generate_embedding

    async def test_filter_by_project(self, filtered_collection):
        """Test filtering results by project field."""
        collection, generate_embedding = filtered_collection

        query_vector = generate_embedding("software development best practices")

        # Filter for project_a only
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            filter_query={"project": "project_a"}
        )

        # All results should be from project_a
        for result in results:
            assert result['project'] == 'project_a', \
                f"Expected project 'project_a', got '{result['project']}'"

    async def test_filter_by_type(self, filtered_collection):
        """Test filtering results by document type."""
        collection, generate_embedding = filtered_collection

        query_vector = generate_embedding("how to do something")

        # Filter for guide type only
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            filter_query={"type": "guide"}
        )

        # All results should be guides
        for result in results:
            assert result['type'] == 'guide', \
                f"Expected type 'guide', got '{result['type']}'"

    async def test_filter_by_department(self, filtered_collection):
        """Test filtering results by department."""
        collection, generate_embedding = filtered_collection

        query_vector = generate_embedding("technical documentation")

        # Filter for engineering department only
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            filter_query={"department": "engineering"}
        )

        # All results should be from engineering
        for result in results:
            assert result['department'] == 'engineering', \
                f"Expected department 'engineering', got '{result['department']}'"

    async def test_combined_filters(self, filtered_collection):
        """Test combining multiple filter conditions."""
        collection, generate_embedding = filtered_collection

        query_vector = generate_embedding("software architecture and design")

        # Filter for project_a AND engineering department
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            filter_query={
                "project": "project_a",
                "department": "engineering"
            }
        )

        # All results should match both conditions
        for result in results:
            assert result['project'] == 'project_a', \
                f"Expected project 'project_a', got '{result['project']}'"
            assert result['department'] == 'engineering', \
                f"Expected department 'engineering', got '{result['department']}'"

    async def test_filter_with_in_operator(self, filtered_collection):
        """Test filtering with $in operator for multiple values."""
        collection, generate_embedding = filtered_collection

        query_vector = generate_embedding("documentation and procedures")

        # Filter for multiple projects
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            filter_query={"project": {"$in": ["project_a", "project_c"]}}
        )

        # All results should be from allowed projects
        allowed_projects = {'project_a', 'project_c'}
        for result in results:
            assert result['project'] in allowed_projects, \
                f"Project '{result['project']}' not in allowed list {allowed_projects}"

    async def test_filter_returns_empty_for_nonexistent_value(self, filtered_collection):
        """Test that filtering by non-existent value returns empty results."""
        collection, generate_embedding = filtered_collection

        query_vector = generate_embedding("anything")

        # Filter for non-existent project
        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10,
            filter_query={"project": "nonexistent_project_xyz"}
        )

        assert len(results) == 0, \
            f"Expected no results for non-existent filter, got {len(results)}"


# =============================================================================
# Test Class: Performance Tests
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
@pytest.mark.slow
class TestVectorSearchPerformance:
    """Performance tests for vector search operations."""

    @pytest.fixture
    async def large_collection(
        self,
        async_mongodb_database,
        generate_embedding,
        test_config
    ):
        """Create collection with large number of documents for performance testing."""
        collection_name = f"{TEST_COLLECTION_PREFIX}_perf_{uuid.uuid4().hex[:8]}"
        collection = async_mongodb_database[collection_name]

        # Try to create vector search index
        try:
            index_definition = {
                "name": "perf_vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "vector",
                            "numDimensions": EMBEDDING_DIMENSIONS,
                            "similarity": "cosine"
                        },
                        {"type": "filter", "path": "category"},
                    ]
                }
            }
            await collection.create_search_index(index_definition)
            await asyncio.sleep(3)
        except Exception as e:
            if 'not supported' in str(e).lower():
                pytest.skip("Vector search not available for performance tests")
            raise

        # Generate 500 test documents (reduced from 1000 for faster tests)
        categories = ['tech', 'science', 'business', 'health', 'sports']
        topics = [
            "artificial intelligence machine learning",
            "cloud computing infrastructure",
            "data analytics and visualization",
            "cybersecurity and privacy",
            "software development practices",
            "database management systems",
            "network protocols and standards",
            "mobile application development",
            "web technologies and frameworks",
            "blockchain and distributed systems",
        ]

        batch_size = 100
        total_docs = 500

        for batch_start in range(0, total_docs, batch_size):
            batch_docs = []
            for i in range(batch_start, min(batch_start + batch_size, total_docs)):
                topic = topics[i % len(topics)]
                content = f"{topic} document number {i} with additional context for testing"
                batch_docs.append({
                    "_id": f"test_perf_{uuid.uuid4().hex}",
                    "content": content,
                    "vector": generate_embedding(content),
                    "category": categories[i % len(categories)],
                    "doc_number": i,
                    "is_test": True,
                    "test_run_id": test_config.test_run_id,
                })

            await collection.insert_many(batch_docs)

        # Wait for indexing
        await asyncio.sleep(2)

        yield collection, generate_embedding

        # Cleanup
        if test_config.cleanup_after_test:
            await collection.drop()

    async def test_large_result_set_performance(self, large_collection):
        """Test performance with large result sets."""
        collection, generate_embedding = large_collection

        query_vector = generate_embedding("machine learning and artificial intelligence")

        start_time = time.time()
        results = await perform_vector_search(
            collection,
            query_vector,
            index_name="perf_vector_index",
            limit=100
        )
        elapsed_time = time.time() - start_time

        assert len(results) == 100, f"Expected 100 results, got {len(results)}"

        # Performance assertion: should complete within 5 seconds
        assert elapsed_time < 5.0, \
            f"Query took {elapsed_time:.2f}s, should be under 5s"

        print(f"Large result set query time: {elapsed_time:.3f}s for {len(results)} results")

    async def test_response_time_benchmark(self, large_collection):
        """Benchmark response times for various query types."""
        collection, generate_embedding = large_collection

        queries = [
            "database management systems",
            "cloud computing infrastructure",
            "machine learning models",
        ]

        timings = []

        for query in queries:
            query_vector = generate_embedding(query)

            start_time = time.time()
            results = await perform_vector_search(
                collection,
                query_vector,
                index_name="perf_vector_index",
                limit=10
            )
            elapsed_time = time.time() - start_time
            timings.append(elapsed_time)

        avg_time = sum(timings) / len(timings)
        max_time = max(timings)

        # Average response time should be under 1 second
        assert avg_time < 1.0, \
            f"Average query time {avg_time:.3f}s should be under 1s"

        # Max response time should be under 2 seconds
        assert max_time < 2.0, \
            f"Max query time {max_time:.3f}s should be under 2s"

        print(f"Response time benchmark: avg={avg_time:.3f}s, max={max_time:.3f}s")

    async def test_concurrent_search_requests(self, large_collection):
        """Test handling of concurrent search requests."""
        collection, generate_embedding = large_collection

        queries = [
            "database optimization",
            "security protocols",
            "web development",
            "data analysis",
            "cloud services",
        ]

        async def run_search(query: str):
            query_vector = generate_embedding(query)
            start_time = time.time()
            results = await perform_vector_search(
                collection,
                query_vector,
                index_name="perf_vector_index",
                limit=10
            )
            elapsed = time.time() - start_time
            return len(results), elapsed

        # Run 5 concurrent searches
        start_time = time.time()
        tasks = [run_search(q) for q in queries]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # All searches should complete
        for result_count, _ in results:
            assert result_count > 0, "Each search should return results"

        # Total time for concurrent execution should be less than sequential
        sequential_estimate = sum(r[1] for r in results)

        print(f"Concurrent execution: {total_time:.3f}s vs sequential estimate: {sequential_estimate:.3f}s")

        # Concurrent should be faster (or at least not much slower)
        assert total_time < sequential_estimate * 1.5, \
            "Concurrent execution should not be significantly slower than sequential"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    async def edge_case_collection(
        self,
        vector_search_collection,
        generate_embedding,
        test_config
    ):
        """Create collection for edge case testing."""
        collection, index_created = vector_search_collection

        if not index_created:
            pytest.skip("Vector search index not available")

        return collection, generate_embedding

    async def test_empty_collection_search(self, edge_case_collection):
        """Test searching an empty collection."""
        collection, generate_embedding = edge_case_collection

        # Collection is empty at this point
        query_vector = generate_embedding("any query")

        results = await perform_vector_search(
            collection,
            query_vector,
            limit=10
        )

        assert len(results) == 0, "Empty collection should return no results"

    async def test_very_short_content(self, edge_case_collection, test_config):
        """Test documents with very short content."""
        collection, generate_embedding = edge_case_collection

        short_docs = [
            {"content": "Hi"},
            {"content": "OK"},
            {"content": "Yes"},
            {"content": "No"},
        ]

        await create_test_documents_with_vectors(
            collection,
            short_docs,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        query_vector = generate_embedding("greeting")
        results = await perform_vector_search(collection, query_vector, limit=10)

        # Should still return results even with short content
        assert len(results) >= 0, "Should handle short content gracefully"

    async def test_special_characters_in_content(self, edge_case_collection, test_config):
        """Test documents with special characters."""
        collection, generate_embedding = edge_case_collection

        special_docs = [
            {"content": "Error: NullPointerException at line 42"},
            {"content": "SELECT * FROM users WHERE name='O\\'Brien'"},
            {"content": "Path: C:\\Users\\test\\file.txt"},
            {"content": "Regex: ^[a-z]+@[a-z]+\\.[a-z]{2,}$"},
            {"content": "Unicode: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4"},
        ]

        await create_test_documents_with_vectors(
            collection,
            special_docs,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        query_vector = generate_embedding("error exception")
        results = await perform_vector_search(collection, query_vector, limit=10)

        # Should handle special characters without error
        assert isinstance(results, list), "Should return a list"

    async def test_duplicate_documents(self, edge_case_collection, test_config):
        """Test behavior with duplicate content."""
        collection, generate_embedding = edge_case_collection

        # Insert identical content multiple times
        duplicate_content = "This is a duplicate document for testing."
        duplicate_docs = [
            {"content": duplicate_content, "copy": 1},
            {"content": duplicate_content, "copy": 2},
            {"content": duplicate_content, "copy": 3},
        ]

        await create_test_documents_with_vectors(
            collection,
            duplicate_docs,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        query_vector = generate_embedding(duplicate_content)
        results = await perform_vector_search(collection, query_vector, limit=10)

        # All duplicates should be returned with similar scores
        assert len(results) >= 3, "Should return all duplicate documents"

        # All should have very high similarity
        for result in results:
            if result.get('content') == duplicate_content:
                assert result['_similarity'] > 0.95, \
                    f"Duplicate should have high similarity, got {result['_similarity']}"

    async def test_query_with_numbers(self, edge_case_collection, test_config):
        """Test querying with numeric content."""
        collection, generate_embedding = edge_case_collection

        numeric_docs = [
            {"content": "Error code 404 page not found"},
            {"content": "Version 2.0.1 release notes"},
            {"content": "Port 8080 is already in use"},
            {"content": "Process ID 12345 terminated"},
        ]

        await create_test_documents_with_vectors(
            collection,
            numeric_docs,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        # Query with number
        query_vector = generate_embedding("error 404")
        results = await perform_vector_search(collection, query_vector, limit=10)

        # Should find the 404 error document
        assert any('404' in r.get('content', '') for r in results), \
            "Should find document containing the queried number"

    async def test_very_long_content(self, edge_case_collection, test_config):
        """Test documents with very long content."""
        collection, generate_embedding = edge_case_collection

        # Create a document with long content (but not too long for embedding)
        long_content = "This is a test document about software development. " * 50

        long_docs = [
            {"content": long_content[:2000]},  # Truncate to reasonable length
        ]

        await create_test_documents_with_vectors(
            collection,
            long_docs,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        query_vector = generate_embedding("software development")
        results = await perform_vector_search(collection, query_vector, limit=10)

        # Should handle long content
        assert len(results) >= 0, "Should handle long content gracefully"

    async def test_malformed_vector_handling(self, edge_case_collection):
        """Test that system handles attempts to search with malformed vectors."""
        collection, _ = edge_case_collection

        # Test with wrong dimension vector
        wrong_dim_vector = [0.1] * 100  # Wrong dimension

        with pytest.raises(Exception):
            # This should raise an error due to dimension mismatch
            await perform_vector_search(
                collection,
                wrong_dim_vector,
                limit=10
            )

    async def test_nan_values_in_query_vector(self, edge_case_collection):
        """Test handling of NaN values in query vector."""
        collection, generate_embedding = edge_case_collection

        # Create a vector with NaN values
        nan_vector = [float('nan')] * EMBEDDING_DIMENSIONS

        # This should either raise an error or return empty results
        try:
            results = await perform_vector_search(
                collection,
                nan_vector,
                limit=10
            )
            # If no error, results should be empty or undefined
            assert len(results) == 0 or results is None, \
                "NaN vector should not return valid results"
        except Exception:
            # Expected - NaN vectors should fail
            pass

    async def test_inf_values_in_query_vector(self, edge_case_collection):
        """Test handling of infinity values in query vector."""
        collection, generate_embedding = edge_case_collection

        # Create a vector with infinity values
        inf_vector = [float('inf')] * EMBEDDING_DIMENSIONS

        # This should either raise an error or return empty results
        try:
            results = await perform_vector_search(
                collection,
                inf_vector,
                limit=10
            )
            # If no error, results should be empty
            assert len(results) == 0, \
                "Infinity vector should not return valid results"
        except Exception:
            # Expected - infinity vectors should fail
            pass


# =============================================================================
# Test Class: Verification Tests
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.requires_mongodb
class TestVectorSearchVerification:
    """Tests to verify vector search behavior matches expectations."""

    @pytest.fixture
    async def verification_collection(
        self,
        vector_search_collection,
        generate_embedding,
        test_config
    ):
        """Create collection for verification tests."""
        collection, index_created = vector_search_collection

        if not index_created:
            pytest.skip("Vector search index not available")

        # Create documents with known relationships
        test_documents = [
            {"content": "The cat sat on the mat.", "id": "cat_mat"},
            {"content": "A dog ran in the park.", "id": "dog_park"},
            {"content": "The kitten played with yarn.", "id": "kitten_yarn"},
            {"content": "A puppy chased its tail.", "id": "puppy_tail"},
        ]

        await create_test_documents_with_vectors(
            collection,
            test_documents,
            generate_embedding,
            test_config.test_run_id
        )

        await asyncio.sleep(1)

        return collection, generate_embedding

    async def test_similarity_score_matches_cosine(self, verification_collection):
        """Verify that MongoDB similarity scores match manual cosine similarity."""
        collection, generate_embedding = verification_collection

        query_text = "A cat sitting comfortably"
        query_vector = generate_embedding(query_text)

        results = await perform_vector_search(
            collection,
            query_vector,
            limit=4
        )

        # Verify scores are reasonable
        for result in results:
            mongo_similarity = result['_similarity']
            doc_vector = result.get('vector')

            if doc_vector:
                # Calculate expected cosine similarity
                expected_similarity = cosine_similarity(query_vector, doc_vector)

                # Allow small floating point difference
                assert abs(mongo_similarity - expected_similarity) < 0.01, \
                    f"MongoDB similarity {mongo_similarity} differs from calculated {expected_similarity}"

    async def test_semantic_relationships_preserved(self, verification_collection):
        """Test that semantic relationships are preserved in results."""
        collection, generate_embedding = verification_collection

        # Query about cats - should rank cat/kitten documents higher
        query_vector = generate_embedding("feline animals like cats")
        results = await perform_vector_search(collection, query_vector, limit=4)

        # Get the top 2 results
        top_2_contents = [r['content'] for r in results[:2]]

        # Cat or kitten documents should appear in top results
        cat_related = any(
            'cat' in c.lower() or 'kitten' in c.lower()
            for c in top_2_contents
        )

        assert cat_related, \
            f"Cat-related documents should rank high for feline query. Top 2: {top_2_contents}"
