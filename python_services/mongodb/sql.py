"""
MongoDB Service - SQL Operations Mixin

Handles all SQL-related storage and retrieval operations:
- SQL Knowledge (general SQL patterns and rules)
- SQL Examples (few-shot learning)
- SQL Failed Queries (error learning)
- SQL Corrections (user-provided corrections)
- SQL Schema Context (table schemas with relationships)
- SQL Stored Procedures
- Hybrid Retrieval (semantic + keyword search)
- Comprehensive Context Retrieval
"""
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, TYPE_CHECKING

from config import (
    COLLECTION_SQL_KNOWLEDGE,
    COLLECTION_SQL_EXAMPLES,
    COLLECTION_SQL_FAILED_QUERIES,
    COLLECTION_SQL_CORRECTIONS,
    COLLECTION_SQL_SCHEMA_CONTEXT,
    COLLECTION_SQL_STORED_PROCEDURES,
    DEFAULT_SEARCH_LIMIT
)
from database_name_parser import normalize_database_name
from .helpers import compute_content_hash

if TYPE_CHECKING:
    from .base import MongoDBBase

logger = logging.getLogger(__name__)


class SQLMixin:
    """
    Mixin providing SQL-related operations.

    Methods organized by category:
    - SQL Knowledge: store_sql_knowledge, search_sql_knowledge, get_sql_knowledge_stats
    - SQL Examples: store_sql_example, search_sql_examples
    - SQL Failed Queries: store_failed_query, search_failed_queries
    - SQL Corrections: store_sql_correction, search_sql_corrections, update_correction_status,
                       promote_correction_to_example, get_corrections_for_review, get_correction_stats
    - SQL Schema: store_schema_context, search_schema_context, get_schema_by_table, hybrid_schema_retrieval
    - Stored Procedures: store_stored_procedure, search_stored_procedures
    - Comprehensive: get_comprehensive_sql_context, get_sql_rag_stats
    """

    # SQL operation keywords for hybrid retrieval
    SQL_OPERATION_KEYWORDS = {
        'count', 'sum', 'average', 'avg', 'total', 'max', 'min',
        'join', 'group', 'order', 'filter', 'having', 'where',
        'between', 'like', 'distinct', 'exists', 'case',
        'union', 'except', 'intersect', 'top', 'limit'
    }

    # Stopwords to filter from keyword extraction
    SQL_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
        'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'shall', 'can', 'need', 'want', 'get', 'show', 'list', 'find',
        'how', 'what', 'when', 'where', 'who', 'which', 'why', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
        'me', 'my', 'i', 'we', 'our', 'you', 'your', 'it', 'its',
        'this', 'that', 'these', 'those', 'many', 'please', 'give',
        'select', 'table', 'column', 'database', 'query'
    }

    # ========================================================================
    # SQL Knowledge Collection
    # ========================================================================

    async def store_sql_knowledge(
        self: 'MongoDBBase',
        knowledge_id: str,
        content: str,
        knowledge_type: str,
        database_name: Optional[str] = None,
        table_name: Optional[str] = None,
        procedure_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store SQL knowledge entry"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_KNOWLEDGE]

        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(content)

        document = {
            "id": knowledge_id,
            "content": content,
            "knowledge_type": knowledge_type,
            "database_name": database_name,
            "table_name": table_name,
            "procedure_name": procedure_name,
            "description": description,
            "tags": tags or [],
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "content_hash": compute_content_hash(content),
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        # Upsert
        await collection.update_one(
            {"id": knowledge_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

        return knowledge_id

    async def search_sql_knowledge(
        self: 'MongoDBBase',
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        knowledge_type: Optional[str] = None,
        database_name: Optional[str] = None
    ) -> List[Dict]:
        """Search SQL knowledge using semantic similarity"""
        if not self.is_initialized:
            await self.initialize()

        # Generate query embedding
        query_vector = await self.embedding_service.generate_embedding(query)

        # Build filter
        filter_query = {}
        if knowledge_type:
            filter_query["knowledge_type"] = knowledge_type
        if database_name:
            filter_query["database_name"] = database_name

        # Perform vector search
        results = await self._vector_search(
            COLLECTION_SQL_KNOWLEDGE,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None
        )

        # Format results
        return [{
            "id": doc.get("id"),
            "content": doc.get("content"),
            "knowledge_type": doc.get("knowledge_type"),
            "similarity": doc.get("_similarity"),
            "database_name": doc.get("database_name"),
            "table_name": doc.get("table_name"),
            "procedure_name": doc.get("procedure_name"),
            "description": doc.get("description")
        } for doc in results]

    async def get_sql_knowledge_stats(self: 'MongoDBBase') -> Dict:
        """Get SQL knowledge statistics"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_KNOWLEDGE]

        # Count by type
        pipeline = [
            {"$group": {
                "_id": "$knowledge_type",
                "count": {"$sum": 1}
            }}
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=100)

        type_counts = {r["_id"]: r["count"] for r in results if r["_id"]}
        total = sum(type_counts.values())

        return {
            "total": total,
            "by_type": type_counts,
            "schema_contexts": type_counts.get("schema", 0),
            "stored_procedures": type_counts.get("stored_procedure", 0),
            "examples": type_counts.get("example", 0),
            "patterns": type_counts.get("pattern", 0)
        }

    # ========================================================================
    # SQL Examples Collection (Few-Shot Learning)
    # Key insight: Embed ONLY the question, store SQL in metadata
    # ========================================================================

    async def store_sql_example(
        self: 'MongoDBBase',
        database: str,
        prompt: str,
        sql: str,
        response: Optional[str] = None,
        tables_used: Optional[List[str]] = None
    ) -> str:
        """
        Store a successful SQL query example for few-shot learning.

        Following nilenso best practice: Embed only the natural language question,
        not the SQL. This improves retrieval of semantically similar questions.
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_EXAMPLES]

        # Normalize database name for consistent lookup
        normalized_db = normalize_database_name(database)

        # Generate ID based on prompt hash
        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        example_id = f"sql_example_{normalized_db}_{prompt_hash}"

        # KEY INSIGHT: Embed only the question for semantic similarity
        embedding = await self.embedding_service.generate_embedding(prompt)

        now = datetime.utcnow().isoformat()
        document = {
            "id": example_id,
            "database": normalized_db,
            "prompt": prompt,
            "sql": sql,
            "response": response,
            "tables_used": tables_used or [],
            "content_hash": compute_content_hash(prompt),
            "type": "sql_query_example",
            "updated_at": now,
            "vector": embedding
        }

        await collection.update_one(
            {"id": example_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": now}
            },
            upsert=True
        )

        return example_id

    async def search_sql_examples(
        self: 'MongoDBBase',
        query: str,
        database: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        Search for similar SQL examples using semantic similarity on questions.
        Returns question-SQL pairs for few-shot prompting.
        """
        if not self.is_initialized:
            await self.initialize()

        query_vector = await self.embedding_service.generate_embedding(query)

        filter_query = {}
        if database:
            filter_query["database"] = normalize_database_name(database)

        results = await self._vector_search(
            COLLECTION_SQL_EXAMPLES,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None,
            threshold=0.5
        )

        return [{
            "id": doc.get("id"),
            "prompt": doc.get("prompt"),
            "sql": doc.get("sql"),
            "response": doc.get("response"),
            "tables_used": doc.get("tables_used", []),
            "similarity": doc.get("_similarity"),
            "database": doc.get("database")
        } for doc in results]

    # ========================================================================
    # SQL Failed Queries Collection (Error Learning)
    # ========================================================================

    def _classify_sql_error(self, error: str) -> str:
        """Classify SQL error type for better learning"""
        error_lower = error.lower()

        if 'invalid column' in error_lower or 'invalid object name' in error_lower:
            return 'schema_error'
        if 'syntax' in error_lower:
            return 'syntax_error'
        if 'ambiguous column' in error_lower:
            return 'ambiguous_column'
        if 'join' in error_lower:
            return 'join_error'
        if 'aggregate' in error_lower:
            return 'aggregate_error'
        if 'conversion' in error_lower or 'convert' in error_lower:
            return 'type_error'

        return 'other'

    async def store_failed_query(
        self: 'MongoDBBase',
        database: str,
        prompt: str,
        sql: str,
        error: str,
        tables_involved: Optional[List[str]] = None,
        correction_id: Optional[str] = None,
        corrected_sql: Optional[str] = None
    ) -> str:
        """
        Store a failed SQL query for error-based learning.

        If correction_id and corrected_sql are provided, this failed query
        has been corrected and the link is stored for learning.
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_FAILED_QUERIES]

        normalized_db = normalize_database_name(database)

        import hashlib
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        failed_id = f"sql_failed_{normalized_db}_{prompt_hash}"

        # Create embedding text that captures the failure pattern
        if corrected_sql:
            embedding_text = f"""CORRECTED QUERY - LEARN FROM THIS MISTAKE
Question: {prompt}
Incorrect SQL: {sql}
Error: {error}
Correct SQL: {corrected_sql}
Use the correct SQL pattern, avoid the incorrect one."""
        else:
            embedding_text = f"""FAILED QUERY - AVOID THIS PATTERN
Question: {prompt}
Incorrect SQL: {sql}
Error: {error}
This query pattern should NOT be repeated."""

        embedding = await self.embedding_service.generate_embedding(embedding_text)

        now = datetime.utcnow().isoformat()
        document = {
            "id": failed_id,
            "database": normalized_db,
            "prompt": prompt,
            "sql": sql,
            "error": error,
            "error_type": self._classify_sql_error(error),
            "tables_involved": tables_involved or [],
            "content_hash": compute_content_hash(embedding_text),
            "type": "sql_failed_query",
            "updated_at": now,
            "vector": embedding,
            "has_correction": correction_id is not None,
            "correction_id": correction_id,
            "corrected_sql": corrected_sql
        }

        await collection.update_one(
            {"id": failed_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": now}
            },
            upsert=True
        )

        return failed_id

    async def search_failed_queries(
        self: 'MongoDBBase',
        query: str,
        database: Optional[str] = None,
        limit: int = 2
    ) -> List[Dict]:
        """Search for similar failed queries to avoid repeating mistakes."""
        if not self.is_initialized:
            await self.initialize()

        query_vector = await self.embedding_service.generate_embedding(query)

        filter_query = {}
        if database:
            filter_query["database"] = normalize_database_name(database)

        results = await self._vector_search(
            COLLECTION_SQL_FAILED_QUERIES,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None,
            threshold=0.6
        )

        return [{
            "id": doc.get("id"),
            "prompt": doc.get("prompt"),
            "sql": doc.get("sql"),
            "error": doc.get("error"),
            "error_type": doc.get("error_type"),
            "similarity": doc.get("_similarity"),
            "database": doc.get("database"),
            "has_correction": doc.get("has_correction", False),
            "correction_id": doc.get("correction_id"),
            "corrected_sql": doc.get("corrected_sql")
        } for doc in results]

    # ========================================================================
    # SQL Corrections Collection (User-provided corrections)
    # ========================================================================

    async def store_sql_correction(
        self: 'MongoDBBase',
        database: str,
        original_prompt: str,
        original_sql: str,
        error_message: str,
        corrected_prompt: str,
        corrected_sql: str,
        correction_notes: str = "",
        correction_type: str = "unknown",
        submitter_id: Optional[str] = None,
        tables_used: Optional[List[str]] = None
    ) -> str:
        """
        Store a user-provided SQL correction for RAG improvement.

        Following nilenso best practice: Embed only the corrected prompt.
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_CORRECTIONS]
        normalized_db = normalize_database_name(database)

        import hashlib
        prompt_hash = hashlib.sha256(corrected_prompt.encode()).hexdigest()[:16]
        correction_id = f"correction_{normalized_db}_{prompt_hash}"

        # Check for existing correction (deduplication)
        existing = await collection.find_one({"id": correction_id})
        if existing:
            await collection.update_one(
                {"id": correction_id},
                {
                    "$inc": {"validation_count": 1},
                    "$set": {"updated_at": datetime.utcnow().isoformat()}
                }
            )
            return correction_id

        # Embed the corrected prompt ONLY
        embedding = await self.embedding_service.generate_embedding(corrected_prompt)

        now = datetime.utcnow().isoformat()
        document = {
            "id": correction_id,
            "database": normalized_db,
            "original_prompt": original_prompt,
            "original_sql": original_sql,
            "error_message": error_message,
            "corrected_prompt": corrected_prompt,
            "corrected_sql": corrected_sql,
            "correction_notes": correction_notes,
            "tables_used": tables_used or [],
            "correction_type": correction_type,
            "submitter_id": submitter_id,
            "submission_source": "ui_feedback",
            "status": "pending",
            "confidence_score": 0.0,
            "validation_count": 0,
            "rejection_count": 0,
            "last_validated_at": None,
            "promoted_to_examples_at": None,
            "vector": embedding,
            "content_hash": compute_content_hash(corrected_prompt),
            "created_at": now,
            "updated_at": now
        }

        await collection.insert_one(document)
        return correction_id

    async def search_sql_corrections(
        self: 'MongoDBBase',
        query: str,
        database: Optional[str] = None,
        limit: int = 3,
        status_filter: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        Search corrections with quality filters.
        Default status filter: ['validated', 'pending'] - excludes rejected
        """
        if not self.is_initialized:
            await self.initialize()

        query_vector = await self.embedding_service.generate_embedding(query)

        filter_query = {}
        if database:
            filter_query["database"] = normalize_database_name(database)

        if status_filter:
            filter_query["status"] = {"$in": status_filter}
        else:
            filter_query["status"] = {"$in": ["validated", "pending"]}

        if min_confidence > 0:
            filter_query["confidence_score"] = {"$gte": min_confidence}

        results = await self._vector_search(
            COLLECTION_SQL_CORRECTIONS,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None,
            threshold=0.5
        )

        return [
            {
                "id": r.get("id"),
                "corrected_prompt": r.get("corrected_prompt"),
                "corrected_sql": r.get("corrected_sql"),
                "original_prompt": r.get("original_prompt"),
                "original_sql": r.get("original_sql"),
                "error_message": r.get("error_message"),
                "correction_notes": r.get("correction_notes"),
                "correction_type": r.get("correction_type"),
                "status": r.get("status"),
                "confidence_score": r.get("confidence_score", 0),
                "validation_count": r.get("validation_count", 0),
                "tables_used": r.get("tables_used", []),
                "similarity": r.get("_similarity", 0),
                "database": r.get("database")
            }
            for r in results
        ]

    async def update_correction_status(
        self: 'MongoDBBase',
        correction_id: str,
        status: str,
        reviewer_id: Optional[str] = None,
        rejection_reason: Optional[str] = None
    ) -> bool:
        """Update the status of a correction (validate, reject, promote)."""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_CORRECTIONS]

        update_fields = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }

        if status == "validated":
            update_fields["$inc"] = {"validation_count": 1}
            update_fields["last_validated_at"] = datetime.utcnow().isoformat()
        elif status == "rejected":
            update_fields["$inc"] = {"rejection_count": 1}
            if rejection_reason:
                update_fields["rejection_reason"] = rejection_reason

        if reviewer_id:
            update_fields["last_reviewer_id"] = reviewer_id

        inc_ops = update_fields.pop("$inc", None)
        update_doc = {"$set": update_fields}
        if inc_ops:
            update_doc["$inc"] = inc_ops

        result = await collection.update_one(
            {"id": correction_id},
            update_doc
        )

        return result.modified_count > 0

    async def promote_correction_to_example(
        self: 'MongoDBBase',
        correction_id: str,
        reviewer_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Promote a validated correction to the sql_examples collection.
        Returns the new example ID if successful.
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_CORRECTIONS]

        correction = await collection.find_one({"id": correction_id})
        if not correction:
            return None

        if correction.get("status") != "validated":
            return None

        example_id = await self.store_sql_example(
            database=correction.get("database"),
            prompt=correction.get("corrected_prompt"),
            sql=correction.get("corrected_sql"),
            response=f"Correction from user feedback: {correction.get('correction_notes', '')}",
            tables_used=correction.get("tables_used", [])
        )

        await collection.update_one(
            {"id": correction_id},
            {
                "$set": {
                    "status": "promoted",
                    "promoted_to_examples_at": datetime.utcnow().isoformat(),
                    "promoted_example_id": example_id,
                    "last_reviewer_id": reviewer_id,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )

        return example_id

    async def get_corrections_for_review(
        self: 'MongoDBBase',
        database: Optional[str] = None,
        status: str = "pending",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get corrections awaiting review."""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_CORRECTIONS]

        query = {"status": status}
        if database:
            query["database"] = normalize_database_name(database)

        cursor = collection.find(
            query,
            {"vector": 0}
        ).sort("created_at", -1).skip(offset).limit(limit)

        results = []
        async for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)

        return results

    async def get_correction_stats(
        self: 'MongoDBBase',
        database: Optional[str] = None
    ) -> Dict:
        """Get statistics about corrections."""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_CORRECTIONS]

        match_stage = {}
        if database:
            match_stage["database"] = normalize_database_name(database)

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"}
                }
            }
        ]

        stats = {"total": 0, "by_status": {}}
        async for doc in collection.aggregate(pipeline):
            status = doc["_id"]
            stats["by_status"][status] = {
                "count": doc["count"],
                "avg_confidence": round(doc.get("avg_confidence", 0), 3)
            }
            stats["total"] += doc["count"]

        return stats

    # ========================================================================
    # SQL Schema Context Collection
    # ========================================================================

    def _format_schema_embedding_text(
        self,
        table_name: str,
        schema_info: Dict,
        summary: Optional[str] = None,
        purpose: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        common_queries: Optional[List[str]] = None
    ) -> str:
        """Format schema for embedding following nilenso best practices."""
        parts = []

        parts.append(f"Table: {table_name}")

        if summary:
            parts.append(f"Description: {summary}")

        if purpose and purpose != summary:
            parts.append(f"Purpose: {purpose}")

        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")

        if common_queries:
            parts.append(f"Common queries: {'; '.join(common_queries)}")

        columns = schema_info.get('columns', [])
        if columns:
            col_descriptions = []
            for col in columns:
                col_desc = f"{col.get('name')} ({col.get('type')})"
                if col.get('description'):
                    col_desc += f": {col.get('description')}"
                col_descriptions.append(col_desc)
            parts.append("Columns: " + ", ".join(col_descriptions))

        primary_keys = schema_info.get('primaryKeys', [])
        if primary_keys:
            parts.append(f"Primary Key: {', '.join(primary_keys)}")

        foreign_keys = schema_info.get('foreignKeys', [])
        if foreign_keys:
            fk_parts = []
            for fk in foreign_keys:
                fk_parts.append(f"{fk.get('column')} -> {fk.get('referencedTable')}.{fk.get('referencedColumn')}")
            parts.append("Relationships: " + ", ".join(fk_parts))

        related_tables = schema_info.get('relatedTables', [])
        if related_tables:
            parts.append(f"Related Tables: {', '.join(related_tables)}")

        sample_values = schema_info.get('sampleValues', {})
        if sample_values:
            sample_parts = []
            for col, values in sample_values.items():
                if values:
                    sample_parts.append(f"{col}: {', '.join(str(v) for v in values[:5])}")
            if sample_parts:
                parts.append("Sample Values: " + "; ".join(sample_parts))

        return "\n".join(parts)

    async def store_schema_context(
        self: 'MongoDBBase',
        database: str,
        table_name: str,
        schema_info: Dict,
        summary: Optional[str] = None,
        purpose: Optional[str] = None,
        key_columns: Optional[List[str]] = None,
        relationships: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        common_queries: Optional[List[str]] = None
    ) -> str:
        """Store enhanced schema context with FK relationships and semantic metadata."""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_SCHEMA_CONTEXT]

        normalized_db = normalize_database_name(database)
        schema_id = f"schema_{normalized_db}_{table_name}"

        embedding_text = self._format_schema_embedding_text(
            table_name, schema_info, summary, purpose, keywords, common_queries
        )
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        document = {
            "id": schema_id,
            "database": normalized_db,
            "table_name": table_name,
            "schema": schema_info.get('schema', 'dbo'),
            "columns": schema_info.get('columns', []),
            "primary_keys": schema_info.get('primaryKeys', []),
            "foreign_keys": schema_info.get('foreignKeys', []),
            "related_tables": schema_info.get('relatedTables', []),
            "sample_values": schema_info.get('sampleValues', {}),
            "summary": summary,
            "purpose": purpose,
            "key_columns": key_columns or [],
            "relationships": relationships or [],
            "keywords": keywords or [],
            "common_queries": common_queries or [],
            "embedding_text": embedding_text,
            "content_hash": compute_content_hash(embedding_text),
            "type": "sql_schema_context",
            "updated": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        await collection.update_one(
            {"id": schema_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

        return schema_id

    async def search_schema_context(
        self: 'MongoDBBase',
        query: str,
        database: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for relevant schema context using semantic similarity.
        If query is empty, returns schemas filtered by database without semantic search.
        """
        if not self.is_initialized:
            await self.initialize()

        filter_query = {}
        if database:
            filter_query["database"] = normalize_database_name(database)

        if not query or not query.strip():
            collection = self.db[COLLECTION_SQL_SCHEMA_CONTEXT]
            cursor = collection.find(filter_query).limit(limit)
            results = await cursor.to_list(length=limit)
        else:
            query_vector = await self.embedding_service.generate_embedding(query)
            results = await self._vector_search(
                COLLECTION_SQL_SCHEMA_CONTEXT,
                query_vector,
                limit=limit,
                filter_query=filter_query if filter_query else None,
                threshold=0.15
            )

        return [{
            "id": doc.get("id"),
            "table_name": doc.get("table_name"),
            "schema": doc.get("schema"),
            "columns": doc.get("columns", []),
            "primary_keys": doc.get("primary_keys", []),
            "foreign_keys": doc.get("foreign_keys", []),
            "related_tables": doc.get("related_tables", []),
            "sample_values": doc.get("sample_values", {}),
            "summary": doc.get("summary"),
            "purpose": doc.get("purpose"),
            "keywords": doc.get("keywords", []),
            "is_view": doc.get("is_view", False),
            "object_type": doc.get("object_type", "TABLE"),
            "similarity": doc.get("_similarity"),
            "database": doc.get("database")
        } for doc in results]

    async def search_views(
        self: 'MongoDBBase',
        query: str,
        database: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search specifically for views (is_view=True) using semantic similarity.
        Views often contain pre-joined data with human-readable column names,
        making them ideal for SQL generation.
        """
        if not self.is_initialized:
            await self.initialize()

        filter_query = {"is_view": True}
        if database:
            filter_query["database"] = normalize_database_name(database)

        if not query or not query.strip():
            collection = self.db[COLLECTION_SQL_SCHEMA_CONTEXT]
            cursor = collection.find(filter_query).limit(limit)
            results = await cursor.to_list(length=limit)
        else:
            query_vector = await self.embedding_service.generate_embedding(query)
            results = await self._vector_search(
                COLLECTION_SQL_SCHEMA_CONTEXT,
                query_vector,
                limit=limit,
                filter_query=filter_query,
                threshold=0.2  # Lower threshold for views to capture more options
            )

        return [{
            "id": doc.get("id"),
            "table_name": doc.get("table_name"),
            "schema": doc.get("schema"),
            "columns": doc.get("columns", []),
            "primary_keys": doc.get("primary_keys", []),
            "foreign_keys": doc.get("foreign_keys", []),
            "related_tables": doc.get("related_tables", []),
            "sample_values": doc.get("sample_values", {}),
            "summary": doc.get("summary"),
            "purpose": doc.get("purpose"),
            "keywords": doc.get("keywords", []),
            "is_view": True,
            "object_type": "VIEW",
            "similarity": doc.get("_similarity"),
            "database": doc.get("database")
        } for doc in results]

    async def get_schema_by_table(
        self: 'MongoDBBase',
        database: str,
        table_name: str
    ) -> Optional[Dict]:
        """Get schema context for a specific table."""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_SCHEMA_CONTEXT]

        doc = await collection.find_one({
            "database": normalize_database_name(database),
            "table_name": table_name
        })

        if not doc:
            return None

        return {
            "id": doc.get("id"),
            "table_name": doc.get("table_name"),
            "schema": doc.get("schema"),
            "columns": doc.get("columns", []),
            "primary_keys": doc.get("primary_keys", []),
            "foreign_keys": doc.get("foreign_keys", []),
            "related_tables": doc.get("related_tables", []),
            "sample_values": doc.get("sample_values", {}),
            "summary": doc.get("summary"),
            "purpose": doc.get("purpose"),
            "keywords": doc.get("keywords", []),
            "is_view": doc.get("is_view", False),
            "object_type": doc.get("object_type", "TABLE"),
            "database": doc.get("database")
        }

    # ========================================================================
    # Hybrid Retrieval (Semantic + Keyword Search)
    # ========================================================================

    async def hybrid_schema_retrieval(
        self: 'MongoDBBase',
        query: str,
        database: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Combine semantic and keyword search for better schema retrieval.
        Uses Reciprocal Rank Fusion (RRF) to merge results.
        """
        if not self.is_initialized:
            await self.initialize()

        import asyncio

        semantic_task = self.search_schema_context(
            query=query,
            database=database,
            limit=limit * 2
        )

        keyword_task = self._keyword_search_schema(
            query=query,
            database=database,
            limit=limit * 2
        )

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )

        merged = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            k=60
        )

        return merged[:limit]

    def _extract_sql_keywords(self, query: str) -> List[str]:
        """Extract SQL-relevant keywords from a natural language query."""
        keywords = []
        query_lower = query.lower()

        # CamelCase identifiers
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', query)
        keywords.extend(camel_case)

        # snake_case identifiers
        snake_case = re.findall(r'\b[a-z]+_[a-z]+(?:_[a-z]+)*\b', query)
        keywords.extend(snake_case)

        # Quoted terms
        quoted_single = re.findall(r"'([^']+)'", query)
        quoted_double = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted_single)
        keywords.extend(quoted_double)

        # SQL operation keywords
        for term in self.SQL_OPERATION_KEYWORDS:
            if term in query_lower:
                keywords.append(term)

        # Entity names (capitalized words not at sentence start)
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and len(word) > 2:
                if word[0].isupper():
                    clean = re.sub(r'[^\w]', '', word)
                    if clean and clean.lower() not in self.SQL_STOPWORDS:
                        keywords.append(clean)

        # Table reference patterns
        table_refs = re.findall(
            r'\bthe\s+(\w+)\s+table\b|\bfrom\s+(\w+)\b|\bjoin\s+(\w+)\b',
            query,
            re.IGNORECASE
        )
        for groups in table_refs:
            for g in groups:
                if g:
                    keywords.append(g)

        # Title case sequences
        title_sequences = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        for seq in title_sequences:
            if len(seq.split()) <= 3:
                compound = seq.replace(' ', '')
                if len(compound) > 3:
                    keywords.append(compound)

        # Filter stopwords and duplicates
        filtered = []
        seen = set()
        for k in keywords:
            k_lower = k.lower()
            if (k_lower not in self.SQL_STOPWORDS and
                k_lower not in seen and
                len(k) > 2):
                filtered.append(k)
                seen.add(k_lower)

        logger.debug(f"Extracted SQL keywords from '{query[:50]}...': {filtered}")
        return filtered

    async def _keyword_search_schema(
        self: 'MongoDBBase',
        query: str,
        database: str,
        limit: int
    ) -> List[Dict]:
        """Search schema by keywords in table/column names."""
        if not self.is_initialized:
            await self.initialize()

        keywords = self._extract_sql_keywords(query)
        if not keywords:
            return []

        collection = self.db[COLLECTION_SQL_SCHEMA_CONTEXT]
        normalized_db = normalize_database_name(database)

        pattern = '|'.join(re.escape(k) for k in keywords)

        try:
            cursor = collection.find({
                "database": {"$regex": f"^{normalized_db}$", "$options": "i"},
                "$or": [
                    {"table_name": {"$regex": pattern, "$options": "i"}},
                    {"columns.name": {"$regex": pattern, "$options": "i"}},
                    {"keywords": {"$in": [k.lower() for k in keywords]}}
                ]
            }).limit(limit)

            results = await cursor.to_list(length=limit)

            return [{
                "id": doc.get("id"),
                "table_name": doc.get("table_name"),
                "schema": doc.get("schema"),
                "columns": doc.get("columns", []),
                "primary_keys": doc.get("primary_keys", []),
                "foreign_keys": doc.get("foreign_keys", []),
                "related_tables": doc.get("related_tables", []),
                "sample_values": doc.get("sample_values", {}),
                "summary": doc.get("summary"),
                "similarity": 0.8,
                "database": doc.get("database"),
                "_search_type": "keyword"
            } for doc in results]

        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        results1: List[Dict],
        results2: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """Merge ranked lists using Reciprocal Rank Fusion (RRF)."""
        scores = {}
        seen = {}

        for rank, doc in enumerate(results1):
            doc_id = doc.get("id") or doc.get("table_name", str(rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            seen[doc_id] = doc

        for rank, doc in enumerate(results2):
            doc_id = doc.get("id") or doc.get("table_name", str(rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in seen:
                seen[doc_id] = doc

        # CRITICAL: Boost views over base tables
        # Views have pre-joined data with human-readable columns (no TypeID lookups needed)
        VIEW_BONUS = 0.05  # Significant boost for views
        for doc_id, doc in seen.items():
            table_name = doc.get("table_name", "").lower()
            if ".uvw_" in table_name or table_name.startswith("uvw_"):
                scores[doc_id] += VIEW_BONUS
            elif ".vw_" in table_name or table_name.startswith("vw_"):
                scores[doc_id] += VIEW_BONUS * 0.8

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged = []
        for doc_id in sorted_ids:
            doc = seen[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            merged.append(doc)

        return merged

    # ========================================================================
    # SQL Stored Procedures Collection
    # ========================================================================

    def _format_procedure_embedding_text(
        self,
        procedure_name: str,
        procedure_info: Dict
    ) -> str:
        """Format stored procedure for embedding."""
        parts = []

        schema = procedure_info.get('schema', 'dbo')
        parts.append(f"Stored Procedure: {schema}.{procedure_name}")

        if procedure_info.get('summary'):
            parts.append(f"Purpose: {procedure_info['summary']}")

        if procedure_info.get('input_description'):
            parts.append(f"Input: {procedure_info['input_description']}")
        if procedure_info.get('output_description'):
            parts.append(f"Output: {procedure_info['output_description']}")

        keywords = procedure_info.get('keywords')
        if keywords:
            if isinstance(keywords, list):
                parts.append(f"Keywords: {', '.join(keywords)}")
            else:
                parts.append(f"Keywords: {keywords}")

        parameters = procedure_info.get('parameters', [])
        if parameters:
            param_names = [p.get('name', '') for p in parameters]
            parts.append(f"Parameters: {', '.join(param_names)}")

        if procedure_info.get('tables_affected'):
            parts.append(f"Tables: {', '.join(procedure_info['tables_affected'])}")

        if procedure_info.get('operations'):
            parts.append(f"Operations: {', '.join(procedure_info['operations'])}")

        return "\n".join(parts)

    async def store_stored_procedure(
        self: 'MongoDBBase',
        database: str,
        procedure_name: str,
        procedure_info: Dict
    ) -> str:
        """Store stored procedure information with semantic search support."""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_SQL_STORED_PROCEDURES]

        normalized_db = normalize_database_name(database)
        sp_id = f"sp_{normalized_db}_{procedure_name}"

        embedding_text = self._format_procedure_embedding_text(procedure_name, procedure_info)
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        keywords = procedure_info.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        document = {
            "id": sp_id,
            "database": normalized_db,
            "procedure_name": procedure_name,
            "schema": procedure_info.get('schema', 'dbo'),
            "parameters": procedure_info.get('parameters', []),
            "definition": procedure_info.get('definition', ''),
            "summary": procedure_info.get('summary'),
            "keywords": keywords,
            "tables_affected": procedure_info.get('tables_affected', []),
            "operations": procedure_info.get('operations', []),
            "input_description": procedure_info.get('input_description'),
            "output_description": procedure_info.get('output_description'),
            "embedding_text": embedding_text,
            "content_hash": compute_content_hash(embedding_text),
            "type": "sql_stored_procedure",
            "updated": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        await collection.update_one(
            {"id": sp_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

        return sp_id

    async def search_stored_procedures(
        self: 'MongoDBBase',
        query: str,
        database: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        Search for stored procedures by semantic similarity.
        If query is empty, returns procedures filtered by database without semantic search.
        """
        if not self.is_initialized:
            await self.initialize()

        filter_query = {}
        if database:
            filter_query["database"] = normalize_database_name(database)

        if not query or not query.strip():
            collection = self.db[COLLECTION_SQL_STORED_PROCEDURES]
            cursor = collection.find(filter_query).limit(limit)
            results = await cursor.to_list(length=limit)
        else:
            query_vector = await self.embedding_service.generate_embedding(query)
            results = await self._vector_search(
                COLLECTION_SQL_STORED_PROCEDURES,
                query_vector,
                limit=limit,
                filter_query=filter_query if filter_query else None,
                threshold=0.5
            )

        return [{
            "id": doc.get("id"),
            "procedure_name": doc.get("procedure_name"),
            "schema": doc.get("schema"),
            "parameters": doc.get("parameters", []),
            "definition": doc.get("definition"),
            "summary": doc.get("summary"),
            "keywords": doc.get("keywords"),
            "similarity": doc.get("_similarity"),
            "database": doc.get("database")
        } for doc in results]

    # ========================================================================
    # Comprehensive Context Retrieval
    # ========================================================================

    async def get_comprehensive_sql_context(
        self: 'MongoDBBase',
        query: str,
        database: str,
        schema_limit: int = 20,
        example_limit: int = 5,
        failed_limit: int = 3,
        sp_limit: int = 5,
        correction_limit: int = 3
    ) -> Dict:
        """
        Get comprehensive context for SQL generation.
        Combines schema, examples, corrections, failed queries, and stored procedures.
        """
        if not self.is_initialized:
            await self.initialize()

        import asyncio

        schema_task = self.search_schema_context(query, database, schema_limit)
        examples_task = self.search_sql_examples(query, database, example_limit)
        corrections_task = self.search_sql_corrections(
            query, database, correction_limit,
            status_filter=["validated", "pending"]
        )
        failed_task = self.search_failed_queries(query, database, failed_limit)
        sp_task = self.search_stored_procedures(query, database, sp_limit)

        schema_context, examples, corrections, failed_queries, stored_procedures = await asyncio.gather(
            schema_task, examples_task, corrections_task, failed_task, sp_task
        )

        merged_examples = self._merge_examples_and_corrections(examples, corrections)

        return {
            "schema_context": schema_context,
            "examples": merged_examples,
            "corrections": corrections,
            "failed_queries": failed_queries,
            "stored_procedures": stored_procedures,
            "database": database,
            "query": query
        }

    def _merge_examples_and_corrections(
        self,
        examples: List[Dict],
        corrections: List[Dict]
    ) -> List[Dict]:
        """Merge and deduplicate examples and corrections with weighted scoring."""
        all_items = []
        seen_prompts = set()

        # Add validated corrections first (highest priority)
        for corr in corrections:
            if corr.get("status") == "validated":
                prompt_key = (corr.get("corrected_prompt") or "").lower().strip()
                if prompt_key and prompt_key not in seen_prompts:
                    all_items.append({
                        "prompt": corr.get("corrected_prompt"),
                        "sql": corr.get("corrected_sql"),
                        "source": "validated_correction",
                        "similarity": corr.get("similarity", 0),
                        "confidence": corr.get("confidence_score", 0.9),
                        "tables_used": corr.get("tables_used", [])
                    })
                    seen_prompts.add(prompt_key)

        # Add regular examples
        for ex in examples:
            prompt_key = (ex.get("prompt") or "").lower().strip()
            if prompt_key and prompt_key not in seen_prompts:
                all_items.append({
                    "prompt": ex.get("prompt"),
                    "sql": ex.get("sql"),
                    "source": "example",
                    "similarity": ex.get("similarity", 0),
                    "confidence": 0.8,
                    "tables_used": ex.get("tables_used", [])
                })
                seen_prompts.add(prompt_key)

        # Add pending corrections (lower priority)
        for corr in corrections:
            if corr.get("status") == "pending":
                prompt_key = (corr.get("corrected_prompt") or "").lower().strip()
                if prompt_key and prompt_key not in seen_prompts:
                    all_items.append({
                        "prompt": corr.get("corrected_prompt"),
                        "sql": corr.get("corrected_sql"),
                        "source": "pending_correction",
                        "similarity": corr.get("similarity", 0),
                        "confidence": max(0.5, corr.get("confidence_score", 0.5)),
                        "tables_used": corr.get("tables_used", [])
                    })
                    seen_prompts.add(prompt_key)

        # Sort by weighted score
        for item in all_items:
            item["weighted_score"] = item.get("similarity", 0) * item.get("confidence", 0.5)

        all_items.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        return all_items

    async def get_sql_rag_stats(self: 'MongoDBBase') -> Dict:
        """Get statistics for all SQL RAG collections."""
        if not self.is_initialized:
            await self.initialize()

        stats = {}

        collections = {
            "examples": COLLECTION_SQL_EXAMPLES,
            "failed_queries": COLLECTION_SQL_FAILED_QUERIES,
            "schema_contexts": COLLECTION_SQL_SCHEMA_CONTEXT,
            "stored_procedures": COLLECTION_SQL_STORED_PROCEDURES,
            "corrections": COLLECTION_SQL_CORRECTIONS
        }

        for name, coll_name in collections.items():
            try:
                count = await self.db[coll_name].count_documents({})
                stats[name] = count
            except Exception:
                stats[name] = 0

        stats["total"] = sum(stats.values())

        try:
            correction_stats = await self.get_correction_stats()
            stats["corrections_by_status"] = correction_stats.get("by_status", {})
        except Exception:
            stats["corrections_by_status"] = {}

        return stats
