"""
MongoDB Service - Documents Mixin

Handles document storage, search, and management operations.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, TYPE_CHECKING

from pymongo import ASCENDING

from config import COLLECTION_DOCUMENTS
from .helpers import chunk_document, format_embedding_text

if TYPE_CHECKING:
    from .base import MongoDBBase


class DocumentsMixin:
    """
    Mixin providing document management operations.

    Methods:
        store_document: Store a document with chunking and embeddings
        search_documents: Search documents using semantic similarity
        get_document: Get a document by ID
        list_documents: List all documents
        update_document_metadata: Update document metadata
        delete_document: Delete a document
        get_document_stats: Get document statistics
    """

    async def store_document(
        self: 'MongoDBBase',
        title: str,
        content: str,
        department: str = "general",
        doc_type: str = "documentation",
        subject: Optional[str] = None,
        file_name: Optional[str] = None,
        file_size: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Store a document with chunking and embeddings.
        Returns document ID and chunk count.
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_DOCUMENTS]

        # Generate document ID
        doc_id = str(uuid.uuid4())
        upload_date = datetime.utcnow()

        # Normalize tags
        tags = tags or ["untagged"]
        if not tags:
            tags = ["untagged"]

        # Chunk the document
        chunks = chunk_document(content)
        print(f"Document '{title}' split into {len(chunks)} chunks")

        # Format chunks with metadata for better semantic retrieval
        formatted_chunks = [
            format_embedding_text(
                content=chunk,
                title=title,
                subject=subject,
                tags=tags
            )
            for chunk in chunks
        ]

        # Generate embeddings for formatted chunks
        embeddings = await self.embedding_service.generate_embeddings_batch(formatted_chunks)

        # Create chunk documents
        chunk_docs = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}-chunk-{i}" if len(chunks) > 1 else doc_id
            chunk_docs.append({
                "id": chunk_id,
                "parent_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "title": title,
                "content": chunk_text,
                "department": department,
                "type": doc_type,
                "subject": subject,
                "file_name": file_name,
                "file_size": file_size,
                "upload_date": upload_date,
                "tags": tags,
                "metadata": metadata or {},
                "created_at": upload_date.isoformat(),
                "updated_at": upload_date.isoformat(),
                "vector": embedding
            })

        # Insert all chunks
        await collection.insert_many(chunk_docs)

        print(f"Stored document '{title}' ({len(chunks)} chunks)")
        return {
            "success": True,
            "document_id": doc_id,
            "parent_id": doc_id,
            "chunks": len(chunks),
            "chunks_created": len(chunks),
            "message": f"Document '{title}' stored successfully"
        }

    async def search_documents(
        self: 'MongoDBBase',
        query: str,
        limit: int = 5,
        department: Optional[str] = None,
        doc_type: Optional[str] = None,
        subject: Optional[str] = None
    ) -> List[Dict]:
        """
        Search documents using semantic similarity.
        Returns unique documents (best matching chunk per document).
        """
        if not self.is_initialized:
            await self.initialize()

        # Generate query embedding
        query_vector = await self.embedding_service.generate_embedding(query)

        # Build filter
        filter_query = {}
        if department:
            filter_query["department"] = department
        if doc_type:
            filter_query["type"] = doc_type
        if subject:
            filter_query["subject"] = subject

        # Perform vector search
        results = await self._vector_search(
            COLLECTION_DOCUMENTS,
            query_vector,
            limit=limit * 2,  # Get more to deduplicate
            filter_query=filter_query if filter_query else None
        )

        # Deduplicate by parent_id (keep best match per document)
        document_map = {}
        for result in results:
            parent_id = result.get("parent_id", result.get("id"))
            if parent_id not in document_map or result["_similarity"] > document_map[parent_id]["_similarity"]:
                document_map[parent_id] = result

        # Format results
        formatted = []
        for doc in sorted(document_map.values(), key=lambda x: x["_similarity"], reverse=True)[:limit]:
            formatted.append({
                "id": doc.get("id"),
                "parent_id": doc.get("parent_id"),
                "title": doc.get("title"),
                "content": doc.get("content"),
                "department": doc.get("department"),
                "type": doc.get("type"),
                "subject": doc.get("subject"),
                "file_name": doc.get("file_name"),
                "upload_date": doc.get("upload_date"),
                "tags": doc.get("tags", []),
                "relevance_score": doc.get("_similarity"),
                "chunk_index": doc.get("chunk_index"),
                "total_chunks": doc.get("total_chunks")
            })

        return formatted

    async def get_document(self: 'MongoDBBase', document_id: str) -> Optional[Dict]:
        """Get a document by ID (returns all chunks combined)"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_DOCUMENTS]

        # Find all chunks for this document
        cursor = collection.find({
            "$or": [
                {"parent_id": document_id},
                {"id": document_id}
            ]
        }).sort("chunk_index", ASCENDING)

        chunks = await cursor.to_list(length=1000)

        if not chunks:
            return None

        # Combine chunks
        full_content = ' '.join(chunk.get("content", "") for chunk in chunks)

        first_chunk = chunks[0]
        return {
            "id": document_id,
            "title": first_chunk.get("title"),
            "content": full_content,
            "department": first_chunk.get("department"),
            "type": first_chunk.get("type"),
            "subject": first_chunk.get("subject"),
            "file_name": first_chunk.get("file_name"),
            "file_size": first_chunk.get("file_size"),
            "upload_date": first_chunk.get("upload_date"),
            "tags": first_chunk.get("tags", []),
            "metadata": first_chunk.get("metadata", {}),
            "chunks": len(chunks)
        }

    async def list_documents(self: 'MongoDBBase', limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all documents (without full content)"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_DOCUMENTS]

        # Aggregate to get unique documents
        pipeline = [
            {"$group": {
                "_id": "$parent_id",
                "title": {"$first": "$title"},
                "department": {"$first": "$department"},
                "type": {"$first": "$type"},
                "subject": {"$first": "$subject"},
                "file_name": {"$first": "$file_name"},
                "file_size": {"$first": "$file_size"},
                "upload_date": {"$first": "$upload_date"},
                "tags": {"$first": "$tags"},
                "chunks": {"$max": "$total_chunks"}
            }},
            {"$sort": {"upload_date": -1}},
            {"$skip": offset},
            {"$limit": limit}
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)

        return [{
            "id": doc.get("_id"),
            "title": doc.get("title"),
            "department": doc.get("department"),
            "type": doc.get("type"),
            "subject": doc.get("subject"),
            "file_name": doc.get("file_name"),
            "file_size": doc.get("file_size"),
            "upload_date": doc.get("upload_date"),
            "tags": doc.get("tags", []),
            "chunks": doc.get("chunks", 1)
        } for doc in results]

    async def update_document_metadata(
        self: 'MongoDBBase',
        document_id: str,
        title: Optional[str] = None,
        department: Optional[str] = None,
        doc_type: Optional[str] = None,
        subject: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Update document metadata without re-embedding"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_DOCUMENTS]

        # Build update
        update_fields = {}
        if title is not None:
            update_fields["title"] = title
        if department is not None:
            update_fields["department"] = department
        if doc_type is not None:
            update_fields["type"] = doc_type
        if subject is not None:
            update_fields["subject"] = subject
        if tags is not None:
            update_fields["tags"] = tags if tags else ["untagged"]
        if metadata is not None:
            update_fields["metadata"] = metadata

        if not update_fields:
            return {"success": False, "message": "No fields to update"}

        # Update all chunks for this document
        result = await collection.update_many(
            {"$or": [{"parent_id": document_id}, {"id": document_id}]},
            {"$set": update_fields}
        )

        return {
            "success": True,
            "document_id": document_id,
            "chunks_updated": result.modified_count,
            "message": f"Updated {result.modified_count} chunks"
        }

    async def delete_document(self: 'MongoDBBase', document_id: str) -> Dict:
        """Delete a document and all its chunks"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_DOCUMENTS]

        result = await collection.delete_many({
            "$or": [
                {"parent_id": document_id},
                {"id": document_id}
            ]
        })

        return {
            "success": result.deleted_count > 0,
            "message": f"Deleted {result.deleted_count} chunks"
        }

    async def get_document_stats(self: 'MongoDBBase') -> Dict:
        """Get document collection statistics"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_DOCUMENTS]

        # Count total chunks
        total_chunks = await collection.count_documents({})

        # Aggregate statistics
        pipeline = [
            {"$group": {
                "_id": "$parent_id",
                "department": {"$first": "$department"},
                "type": {"$first": "$type"},
                "subject": {"$first": "$subject"}
            }},
            {"$group": {
                "_id": None,
                "total_documents": {"$sum": 1},
                "departments": {"$push": "$department"},
                "types": {"$push": "$type"},
                "subjects": {"$push": "$subject"}
            }}
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)

        if not results:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "departments": {},
                "types": {},
                "subjects": {}
            }

        result = results[0]

        # Count occurrences
        def count_items(items):
            counts = {}
            for item in items:
                if item:
                    counts[item] = counts.get(item, 0) + 1
            return counts

        return {
            "total_documents": result.get("total_documents", 0),
            "total_chunks": total_chunks,
            "departments": count_items(result.get("departments", [])),
            "types": count_items(result.get("types", [])),
            "subjects": count_items(result.get("subjects", []))
        }
