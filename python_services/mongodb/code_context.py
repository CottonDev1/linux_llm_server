"""
MongoDB Service - Code Context Mixin

Handles code context storage and search operations.
"""
from datetime import datetime
from typing import Optional, List, Dict, TYPE_CHECKING

from config import COLLECTION_CODE_CONTEXT, DEFAULT_SEARCH_LIMIT
from .helpers import compute_content_hash

if TYPE_CHECKING:
    from .base import MongoDBBase


class CodeContextMixin:
    """
    Mixin providing code context operations.

    Methods:
        store_code_context: Store code context document
        search_code_context: Search code context using semantic similarity
        get_code_context_by_id: Get code context by ID
        delete_code_context: Delete code context by ID
        get_code_context_stats: Get code context statistics
    """

    async def store_code_context(
        self: 'MongoDBBase',
        document_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store code context document"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_CODE_CONTEXT]

        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(content)

        document = {
            "id": document_id,
            "document": content,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "content_hash": compute_content_hash(content),
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding,
            **(metadata or {})
        }

        # Upsert with created_at
        await collection.update_one(
            {"id": document_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

        return document_id

    async def search_code_context(
        self: 'MongoDBBase',
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        project: Optional[str] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search code context using semantic similarity"""
        if not self.is_initialized:
            await self.initialize()

        # Generate query embedding
        query_vector = await self.embedding_service.generate_embedding(query)

        # Build filter
        filter_query = filter_dict or {}
        if project:
            filter_query["database"] = project

        # Perform vector search
        results = await self._vector_search(
            COLLECTION_CODE_CONTEXT,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None
        )

        # Format results
        return [{
            "id": doc.get("id"),
            "similarity": doc.get("_similarity"),
            "distance": doc.get("_distance"),
            "content": doc.get("document"),
            "metadata": {k: v for k, v in doc.items() if k not in ["_id", "id", "document", "vector", "_similarity", "_distance", "timestamp"]},
            "timestamp": doc.get("timestamp")
        } for doc in results]

    async def get_code_context_by_id(self: 'MongoDBBase', document_id: str) -> Optional[Dict]:
        """Get code context by ID"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_CODE_CONTEXT]
        doc = await collection.find_one({"id": document_id})

        if not doc:
            return None

        return {
            "id": doc.get("id"),
            "content": doc.get("document"),
            "metadata": {k: v for k, v in doc.items() if k not in ["_id", "id", "document", "vector", "timestamp"]},
            "timestamp": doc.get("timestamp")
        }

    async def delete_code_context(self: 'MongoDBBase', document_id: str) -> bool:
        """Delete code context by ID"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_CODE_CONTEXT]
        result = await collection.delete_one({"id": document_id})
        return result.deleted_count > 0

    async def get_code_context_stats(self: 'MongoDBBase') -> Dict:
        """Get code context statistics"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_CODE_CONTEXT]
        total = await collection.count_documents({})

        return {
            "storage": "MongoDB",
            "total_documents": total,
            "collection": COLLECTION_CODE_CONTEXT
        }
