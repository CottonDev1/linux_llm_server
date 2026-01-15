"""
MongoDB Service - Base Class with Core Infrastructure

This module contains the core MongoDB connection, vector search infrastructure,
index management, and health checking functionality.
"""
import logging
from core.log_utils import log_info, log_warning, log_error


# =============================================================================
# Custom Exceptions for Vector Search
# =============================================================================

class VectorSearchError(Exception):
    """Raised when a vector search operation fails."""
    pass


class VectorSearchUnavailableError(VectorSearchError):
    """Raised when MongoDB Atlas Vector Search is not available."""
    pass
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_REPLICA_SET,
    VECTOR_SEARCH_ENABLED,
    VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER,
    COLLECTION_DOCUMENTS,
    COLLECTION_CODE_CONTEXT,
    COLLECTION_SQL_KNOWLEDGE,
    COLLECTION_CODE_METHODS,
    COLLECTION_CODE_CLASSES,
    COLLECTION_CODE_CALLGRAPH,
    COLLECTION_CODE_EVENTHANDLERS,
    COLLECTION_CODE_DBOPERATIONS,
    COLLECTION_SQL_EXAMPLES,
    COLLECTION_SQL_FAILED_QUERIES,
    COLLECTION_SQL_SCHEMA_CONTEXT,
    COLLECTION_SQL_STORED_PROCEDURES,
    COLLECTION_SQL_CORRECTIONS,
    COLLECTION_FEEDBACK,
    COLLECTION_QUERY_SESSIONS,
    COLLECTION_TICKET_MATCH_HISTORY,
    COLLECTION_AUDIO_ANALYSIS,
    COLLECTION_PHONE_CUSTOMER_MAP,
    EMBEDDING_DIMENSIONS,
    DEFAULT_SEARCH_LIMIT,
    SIMILARITY_THRESHOLD
)
from embedding_service import get_embedding_service

if TYPE_CHECKING:
    from embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class MongoDBBase:
    """
    Base class for MongoDB service providing core infrastructure.

    This class handles:
    - MongoDB connection management
    - Vector search infrastructure (native and in-memory fallback)
    - Index creation and management
    - Health checking and statistics

    All domain-specific functionality is provided by mixin classes.
    """

    _instance: Optional['MongoDBBase'] = None

    # Mapping of collection names to their vector search index names
    VECTOR_INDEX_NAMES = {
        COLLECTION_DOCUMENTS: "documents_vector_index",
        COLLECTION_CODE_CONTEXT: "code_context_vector_index",
        COLLECTION_SQL_EXAMPLES: "sql_examples_vector_index",
        COLLECTION_SQL_SCHEMA_CONTEXT: "sql_schema_context_vector_index",
        COLLECTION_SQL_STORED_PROCEDURES: "sql_stored_procedures_vector_index",
        COLLECTION_CODE_METHODS: "code_methods_vector_index",
        COLLECTION_CODE_CLASSES: "code_classes_vector_index",
        COLLECTION_CODE_CALLGRAPH: "code_callgraph_vector_index",
        COLLECTION_CODE_EVENTHANDLERS: "code_eventhandlers_vector_index",
        COLLECTION_CODE_DBOPERATIONS: "code_dboperations_vector_index",
        COLLECTION_SQL_KNOWLEDGE: "sql_knowledge_vector_index",
        COLLECTION_SQL_FAILED_QUERIES: "sql_failed_queries_vector_index",
        COLLECTION_AUDIO_ANALYSIS: "audio_analysis_vector_index",
    }

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.embedding_service: 'EmbeddingService' = get_embedding_service()
        self.is_initialized = False
        self._vector_search_available = False
        self._mongodb_version: Optional[str] = None

    @classmethod
    def get_instance(cls) -> 'MongoDBBase':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_connection_uri(self) -> str:
        """Build MongoDB connection URI with optional replica set support"""
        uri = MONGODB_URI

        # Add replica set parameter if configured
        if MONGODB_REPLICA_SET:
            separator = "&" if "?" in uri else "?"
            uri = f"{uri}{separator}replicaSet={MONGODB_REPLICA_SET}"

        return uri

    async def initialize(self):
        """Initialize MongoDB connection and create indexes"""
        if self.is_initialized:
            return

        connection_uri = self._build_connection_uri()
        log_info("MongoDB", f"Connecting to: {connection_uri}")
        # Configure MongoDB client with proper timeouts and retry settings
        self.client = AsyncIOMotorClient(
            connection_uri,
            serverSelectionTimeoutMS=10000,  # 10 second timeout for server selection
            connectTimeoutMS=10000,          # 10 second timeout for initial connection
            socketTimeoutMS=30000,           # 30 second timeout for socket operations
            retryWrites=True,                # Automatically retry write operations
            retryReads=True,                 # Automatically retry read operations
            maxPoolSize=10,                  # Connection pool size
            minPoolSize=1,                   # Minimum connections to keep open
        )
        self.db = self.client[MONGODB_DATABASE]

        # Test connection with timeout
        await self.client.admin.command('ping')
        log_info("MongoDB", "Connected successfully")

        # Initialize embedding service
        await self.embedding_service.initialize()

        # Create standard indexes
        await self._create_indexes()

        # Check vector search support and create indexes if available
        if VECTOR_SEARCH_ENABLED:
            vector_support = await self.check_vector_search_support()
            self._vector_search_available = vector_support.get("supported", False)
            self._mongodb_version = vector_support.get("version")

            if self._vector_search_available:
                log_info("MongoDB", f"Native vector search available (MongoDB {self._mongodb_version})")
                await self.create_vector_search_indexes()
            else:
                log_warning("MongoDB", f"Native vector search not available: {vector_support.get('reason', 'unknown')}")
                log_info("MongoDB", "Using in-memory vector search fallback")
        else:
            log_info("MongoDB", "Vector search disabled by configuration, using in-memory fallback")

        self.is_initialized = True
        log_info("MongoDB", f"Initialized with database: {MONGODB_DATABASE}")

    async def _create_indexes(self):
        """Create necessary indexes for efficient queries"""
        # Documents collection indexes
        docs_collection = self.db[COLLECTION_DOCUMENTS]
        await docs_collection.create_indexes([
            IndexModel([("parent_id", ASCENDING)]),
            IndexModel([("department", ASCENDING)]),
            IndexModel([("type", ASCENDING)]),
            IndexModel([("subject", ASCENDING)]),
            IndexModel([("upload_date", DESCENDING)]),
            IndexModel([("tags", ASCENDING)]),
        ])

        # Code context collection indexes
        code_collection = self.db[COLLECTION_CODE_CONTEXT]
        await code_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("category", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
        ])

        # SQL knowledge collection indexes
        sql_collection = self.db[COLLECTION_SQL_KNOWLEDGE]
        await sql_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("knowledge_type", ASCENDING)]),
            IndexModel([("database_name", ASCENDING)]),
            IndexModel([("table_name", ASCENDING)]),
        ])

        # SQL Examples collection indexes (few-shot learning)
        examples_collection = self.db[COLLECTION_SQL_EXAMPLES]
        await examples_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("created", DESCENDING)]),
        ])

        # SQL Failed Queries collection indexes (error learning)
        failed_collection = self.db[COLLECTION_SQL_FAILED_QUERIES]
        await failed_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("error_type", ASCENDING)]),
            IndexModel([("created", DESCENDING)]),
        ])

        # SQL Schema Context collection indexes (table schemas)
        schema_collection = self.db[COLLECTION_SQL_SCHEMA_CONTEXT]
        await schema_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("table_name", ASCENDING)]),
            IndexModel([("database", ASCENDING), ("table_name", ASCENDING)], unique=True),
            # Speed up keyword searches on nested column names and keywords array
            IndexModel([("columns.name", ASCENDING)]),
            IndexModel([("keywords", ASCENDING)]),
        ])

        # SQL Stored Procedures collection indexes
        sp_collection = self.db[COLLECTION_SQL_STORED_PROCEDURES]
        await sp_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("procedure_name", ASCENDING)]),
            IndexModel([("schema", ASCENDING)]),
        ])

        # Feedback collection indexes
        feedback_collection = self.db[COLLECTION_FEEDBACK]
        await feedback_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("feedback_type", ASCENDING)]),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("query_id", ASCENDING)]),
            IndexModel([("document_ids", ASCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
        ])

        # Query sessions collection indexes
        sessions_collection = self.db[COLLECTION_QUERY_SESSIONS]
        await sessions_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("database", ASCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
        ])

        # Audio analysis collection indexes (for ticket correlation)
        audio_collection = self.db[COLLECTION_AUDIO_ANALYSIS]
        await audio_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("call_datetime", DESCENDING)]),
            IndexModel([("call_metadata.phone_number", ASCENDING)]),
            IndexModel([("call_metadata.extension", ASCENDING)]),
            IndexModel([("customer_support_staff", ASCENDING)]),
            IndexModel([("ewr_customer", ASCENDING)]),
            IndexModel([("mood", ASCENDING)]),
            IndexModel([("outcome", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            # Text index for full-text search on transcription
            IndexModel([("transcription", TEXT), ("call_content.subject", TEXT)], name="transcription_text_idx"),
        ])

        # Phone customer mapping collection indexes
        phone_mapping_collection = self.db[COLLECTION_PHONE_CUSTOMER_MAP]
        await phone_mapping_collection.create_indexes([
            IndexModel([("phone_number", ASCENDING), ("customer_id", ASCENDING)], unique=True),
            IndexModel([("phone_number", ASCENDING)]),
            IndexModel([("customer_id", ASCENDING)]),
            IndexModel([("last_seen", DESCENDING)]),
        ])

        log_info("MongoDB", "Database indexes created")

    # ========================================================================
    # Vector Search Support Detection and Index Creation
    # ========================================================================

    async def check_vector_search_support(self) -> Dict:
        """
        Check if native vector search is available (MongoDB 8.2+ or Atlas).

        Returns:
            Dict with keys:
                - supported: bool indicating if native vector search is available
                - version: MongoDB version string
                - reason: explanation if not supported
                - is_atlas: bool indicating if connected to Atlas
        """
        try:
            # Get server info to determine MongoDB version
            server_info = await self.client.admin.command('buildInfo')
            version_str = server_info.get('version', '0.0.0')
            version_parts = version_str.split('.')

            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0

            # Check if this is Atlas
            is_atlas = 'atlas' in server_info.get('modules', [])

            if is_atlas:
                return {
                    "supported": True,
                    "version": version_str,
                    "reason": "MongoDB Atlas detected - native vector search available",
                    "is_atlas": True
                }

            # MongoDB 8.2+ supports native vector search
            if major > 8 or (major == 8 and minor >= 2):
                try:
                    test_collection = self.db['__vector_search_test__']
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": "__test_index__",
                                "path": "vector",
                                "queryVector": [0.0] * EMBEDDING_DIMENSIONS,
                                "numCandidates": 1,
                                "limit": 1
                            }
                        }
                    ]
                    await test_collection.aggregate(pipeline).to_list(length=1)
                except Exception as e:
                    error_str = str(e).lower()
                    if 'searchnotenabled' in error_str or 'additional configuration' in error_str or 'connect to atlas' in error_str:
                        return {
                            "supported": False,
                            "version": version_str,
                            "reason": f"MongoDB {version_str} requires Atlas or Atlas CLI local deployment for $vectorSearch",
                            "is_atlas": False
                        }
                    if 'index' in error_str or 'not found' in error_str:
                        return {
                            "supported": True,
                            "version": version_str,
                            "reason": f"MongoDB {version_str} supports native vector search",
                            "is_atlas": False
                        }
                    elif 'unknown' in error_str or 'unrecognized' in error_str:
                        return {
                            "supported": False,
                            "version": version_str,
                            "reason": f"$vectorSearch not recognized - MongoDB {version_str} may not support it",
                            "is_atlas": False
                        }

                return {
                    "supported": True,
                    "version": version_str,
                    "reason": f"MongoDB {version_str} supports native vector search",
                    "is_atlas": False
                }

            return {
                "supported": False,
                "version": version_str,
                "reason": f"MongoDB {version_str} does not support native vector search (requires 8.2+)",
                "is_atlas": False
            }

        except Exception as e:
            return {
                "supported": False,
                "version": "unknown",
                "reason": f"Error checking vector search support: {str(e)}",
                "is_atlas": False
            }

    async def create_vector_search_indexes(self):
        """Create vector search indexes for all collections that support semantic search."""
        index_configs = [
            (COLLECTION_DOCUMENTS, "documents_vector_index", ["project", "department", "type", "subject", "tags"]),
            (COLLECTION_CODE_CONTEXT, "code_context_vector_index", ["database", "category"]),
            (COLLECTION_SQL_EXAMPLES, "sql_examples_vector_index", ["database"]),
            (COLLECTION_SQL_SCHEMA_CONTEXT, "sql_schema_context_vector_index", ["database"]),
            (COLLECTION_SQL_STORED_PROCEDURES, "sql_stored_procedures_vector_index", ["database"]),
            (COLLECTION_CODE_METHODS, "code_methods_vector_index", ["project", "namespace", "class_name"]),
            (COLLECTION_CODE_CLASSES, "code_classes_vector_index", ["project", "namespace"]),
            (COLLECTION_CODE_CALLGRAPH, "code_callgraph_vector_index", ["project", "caller_class", "callee_class"]),
            (COLLECTION_CODE_EVENTHANDLERS, "code_eventhandlers_vector_index", ["project", "event_name"]),
            (COLLECTION_CODE_DBOPERATIONS, "code_dboperations_vector_index", ["project", "operation_type"]),
            (COLLECTION_SQL_KNOWLEDGE, "sql_knowledge_vector_index", ["database_name", "knowledge_type"]),
            (COLLECTION_SQL_FAILED_QUERIES, "sql_failed_queries_vector_index", ["database", "error_type"]),
        ]

        for collection_name, index_name, filter_fields in index_configs:
            try:
                await self._create_single_vector_index(collection_name, index_name, filter_fields)
            except Exception as e:
                log_warning("MongoDB", f"Could not create vector index on {collection_name}: {e}")

    async def _create_single_vector_index(
        self,
        collection_name: str,
        index_name: str,
        filter_fields: List[str]
    ):
        """Create a single vector search index on a collection."""
        collection = self.db[collection_name]

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

        for field in filter_fields:
            index_definition["definition"]["fields"].append({
                "type": "filter",
                "path": field
            })

        try:
            existing_indexes = await collection.list_search_indexes().to_list(length=100)
            index_exists = any(idx.get('name') == index_name for idx in existing_indexes)

            if not index_exists:
                await collection.create_search_index(index_definition)
                log_info("MongoDB", f"Created vector search index: {index_name}")

        except Exception as e:
            error_str = str(e).lower()
            if 'not supported' in error_str or 'command not found' in error_str:
                log_info("MongoDB", f"Search indexes not supported for {collection_name}, skipping")
            else:
                raise

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.is_initialized = False
            log_info("MongoDB", "Connection closed")

    # ========================================================================
    # Vector Search Methods
    # ========================================================================

    async def _vector_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
        filter_query: Optional[Dict] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Perform vector similarity search using MongoDB Atlas Vector Search ONLY.

        Args:
            collection_name: Name of the MongoDB collection to search
            query_vector: The embedding vector for the query
            limit: Maximum number of results to return
            filter_query: Optional MongoDB filter to pre-filter documents
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of documents with _similarity and _distance fields added

        Raises:
            VectorSearchUnavailableError: If MongoDB Atlas Vector Search is not available
            VectorSearchError: If the vector search query fails
        """
        index_name = self.VECTOR_INDEX_NAMES.get(collection_name)

        # Require MongoDB Atlas Vector Search - no fallback to in-memory
        if not self._vector_search_available:
            raise VectorSearchUnavailableError(
                f"MongoDB Atlas Vector Search is not available. URI: {MONGODB_URI}"
            )

        if not index_name:
            raise VectorSearchError(
                f"No vector search index configured for collection: {collection_name}"
            )

        try:
            return await self._vector_search_native(
                collection_name=collection_name,
                query_vector=query_vector,
                index_name=index_name,
                limit=limit,
                filter_query=filter_query,
                threshold=threshold
            )
        except Exception as e:
            # Re-raise as VectorSearchError with details
            raise VectorSearchError(
                f"Vector search failed on {collection_name}: {str(e)}"
            ) from e

    async def _vector_search_native(
        self,
        collection_name: str,
        query_vector: List[float],
        index_name: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        filter_query: Optional[Dict] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """Vector search using MongoDB $vectorSearch aggregation (MongoDB 8.2+ or Atlas)."""
        collection = self.db[collection_name]
        num_candidates = limit * VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER

        vector_search_stage = {
            "$vectorSearch": {
                "index": index_name,
                "path": "vector",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit
            }
        }

        if filter_query:
            vector_search_stage["$vectorSearch"]["filter"] = filter_query

        pipeline = [
            vector_search_stage,
            {
                "$addFields": {
                    "_similarity": {"$meta": "vectorSearchScore"},
                    "_distance": {"$subtract": [1.0, {"$meta": "vectorSearchScore"}]}
                }
            },
            {
                "$match": {
                    "_similarity": {"$gte": threshold}
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)
        return results

    # =========================================================================
    # OBSOLETE: In-memory vector search - commented out as of 2026-01-05
    # MongoDB Atlas Vector Search is now REQUIRED - no fallback allowed.
    # This code is preserved for reference only.
    # =========================================================================
    # async def _vector_search_inmemory(
    #     self,
    #     collection_name: str,
    #     query_vector: List[float],
    #     limit: int = DEFAULT_SEARCH_LIMIT,
    #     filter_query: Optional[Dict] = None,
    #     threshold: float = SIMILARITY_THRESHOLD
    # ) -> List[Dict]:
    #     """In-memory vector similarity search using cosine similarity."""
    #     collection = self.db[collection_name]
    #     query = filter_query or {}
    #
    #     cursor = collection.find(query)
    #     documents = await cursor.to_list(length=10000)
    #
    #     if not documents:
    #         return []
    #
    #     results = []
    #     query_vec = np.array(query_vector)
    #     query_norm = np.linalg.norm(query_vec)
    #
    #     if query_norm == 0:
    #         return []
    #
    #     for doc in documents:
    #         if 'vector' not in doc:
    #             continue
    #
    #         doc_vec = np.array(doc['vector'])
    #         doc_norm = np.linalg.norm(doc_vec)
    #
    #         if doc_norm == 0:
    #             continue
    #
    #         similarity = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
    #
    #         if similarity >= threshold:
    #             doc['_similarity'] = similarity
    #             doc['_distance'] = 1.0 - similarity
    #             results.append(doc)
    #
    #     results.sort(key=lambda x: x['_similarity'], reverse=True)
    #     return results[:limit]

    # ========================================================================
    # Health Check and Statistics
    # ========================================================================

    async def health_check(self) -> Dict:
        """Check service health including vector search status"""
        try:
            if not self.is_initialized:
                await self.initialize()

            await self.client.admin.command('ping')

            collections = {}
            for coll_name in [COLLECTION_DOCUMENTS, COLLECTION_CODE_CONTEXT, COLLECTION_SQL_KNOWLEDGE]:
                count = await self.db[coll_name].count_documents({})
                collections[coll_name] = count

            # Vector search is REQUIRED - no fallback mode available
            vector_search_status = "available" if self._vector_search_available else "unavailable"

            return {
                "status": "healthy" if self._vector_search_available else "degraded",
                "mongodb_connected": True,
                "mongodb_uri": MONGODB_URI,
                "mongodb_version": self._mongodb_version,
                "embedding_model_loaded": self.embedding_service.is_initialized,
                "vector_search": {
                    "enabled": VECTOR_SEARCH_ENABLED,
                    "native_available": self._vector_search_available,
                    "status": vector_search_status,
                    "error": None if self._vector_search_available else f"MongoDB Atlas Vector Search required but not available at {MONGODB_URI}"
                },
                "collections": collections
            }
        except Exception as e:
            # Try to reconnect on failure
            log_warning("MongoDB", f"Health check failed: {e}. Attempting reconnection...")
            try:
                # Mark as not initialized to force full reconnection
                self.is_initialized = False
                if self.client:
                    self.client.close()
                await self.initialize()
                log_info("MongoDB", "Reconnection successful")
                # Retry health check after reconnection
                return await self.health_check()
            except Exception as reconnect_error:
                log_error("MongoDB", f"Reconnection failed: {reconnect_error}")

            return {
                "status": "unhealthy",
                "mongodb_connected": False,
                "mongodb_uri": MONGODB_URI,
                "embedding_model_loaded": self.embedding_service.is_initialized if self.embedding_service else False,
                "vector_search": {
                    "enabled": VECTOR_SEARCH_ENABLED,
                    "native_available": False,
                    "status": "unavailable",
                    "error": f"Unable to connect to {MONGODB_URI}: {str(e)}"
                },
                "error": str(e)
            }

    async def get_db_stats(self) -> Dict:
        """Get MongoDB database statistics including size information."""
        from .helpers import format_size

        try:
            if not self.is_initialized:
                await self.initialize()

            db_stats = await self.db.command("dbStats")

            data_size = db_stats.get("dataSize", 0)
            storage_size = db_stats.get("storageSize", 0)
            index_size = db_stats.get("indexSize", 0)
            total_size = data_size + index_size

            collections_info = {}
            collection_names = await self.db.list_collection_names()
            for coll_name in collection_names:
                coll_stats = await self.db.command("collStats", coll_name)
                collections_info[coll_name] = {
                    "count": coll_stats.get("count", 0),
                    "size": coll_stats.get("size", 0),
                    "sizeFormatted": format_size(coll_stats.get("size", 0)),
                    "avgObjSize": coll_stats.get("avgObjSize", 0),
                    "storageSize": coll_stats.get("storageSize", 0),
                    "indexSize": coll_stats.get("totalIndexSize", 0)
                }

            return {
                "success": True,
                "database": MONGODB_DATABASE,
                "dataSize": data_size,
                "dataSizeFormatted": format_size(data_size),
                "storageSize": storage_size,
                "storageSizeFormatted": format_size(storage_size),
                "indexSize": index_size,
                "indexSizeFormatted": format_size(index_size),
                "totalSize": total_size,
                "totalSizeFormatted": format_size(total_size),
                "collections": db_stats.get("collections", 0),
                "objects": db_stats.get("objects", 0),
                "collectionsInfo": collections_info
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
