"""
Code Import Service
===================

Service for importing Roslyn analysis results to the vector database.

This service handles:
- Converting code entities to vector documents
- Generating embeddings for semantic search
- Upsert operations to MongoDB collections
- Batch processing for efficiency

The imported entities support:
- Semantic code search (find methods by description)
- Cross-project code discovery
- Impact analysis (what methods call this database operation?)
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from git_pipeline.models.pipeline_models import (
    ImportResult,
    RoslynAnalysisOutput,
    CodeEntity,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


class CodeImportService:
    """
    Service for importing code analysis results to MongoDB with vector embeddings.

    This service takes the output from Roslyn analysis and:
    1. Generates searchable text from code entities
    2. Creates vector embeddings for semantic search
    3. Upserts documents to appropriate MongoDB collections

    Collections used:
    - code_methods: Method/function definitions
    - code_classes: Class definitions
    - code_context: General code context for RAG
    - code_eventhandlers: Event handler mappings
    - code_dboperations: Database operation references

    Attributes:
        config: Pipeline configuration
        mongodb: MongoDB service instance (lazy loaded)
        embedding_service: Embedding service instance (lazy loaded)
    """

    # Batch size for embedding generation and MongoDB operations
    BATCH_SIZE = 100

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the code import service.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._mongodb = None
        self._embedding_service = None

        logger.info("CodeImportService initialized")

    async def _get_mongodb(self):
        """Lazy load MongoDB service."""
        if self._mongodb is None:
            from mongodb import MongoDBService
            self._mongodb = MongoDBService.get_instance()
            if not self._mongodb.is_initialized:
                await self._mongodb.initialize()
        return self._mongodb

    async def _get_embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService.get_instance()
            if not self._embedding_service.is_initialized:
                await self._embedding_service.initialize()
        return self._embedding_service

    async def import_analysis(
        self,
        analysis: RoslynAnalysisOutput,
        db_name: str,
    ) -> ImportResult:
        """
        Import Roslyn analysis results to MongoDB.

        This method:
        1. Processes each category of code entities
        2. Generates embeddings for semantic search
        3. Upserts documents to MongoDB
        4. Tracks import statistics

        Args:
            analysis: Roslyn analysis output with code entities
            db_name: Database/collection prefix for this repository

        Returns:
            ImportResult with import statistics
        """
        start_time = time.time()
        repository = analysis.repository

        logger.info(f"Starting import for {repository} to collection prefix {db_name}")

        try:
            mongodb = await self._get_mongodb()
            embedding_service = await self._get_embedding_service()

            total_imported = 0
            total_updated = 0
            total_deleted = 0

            # Import classes
            class_stats = await self._import_entities(
                entities=analysis.classes,
                collection_name="code_classes",
                repository=repository,
                db_name=db_name,
                mongodb=mongodb,
                embedding_service=embedding_service
            )
            total_imported += class_stats["imported"]
            total_updated += class_stats["updated"]

            # Import methods
            method_stats = await self._import_entities(
                entities=analysis.methods,
                collection_name="code_methods",
                repository=repository,
                db_name=db_name,
                mongodb=mongodb,
                embedding_service=embedding_service
            )
            total_imported += method_stats["imported"]
            total_updated += method_stats["updated"]

            # Import properties (to code_context for general search)
            property_stats = await self._import_entities(
                entities=analysis.properties,
                collection_name="code_context",
                repository=repository,
                db_name=db_name,
                mongodb=mongodb,
                embedding_service=embedding_service
            )
            total_imported += property_stats["imported"]
            total_updated += property_stats["updated"]

            # Import interfaces
            interface_stats = await self._import_entities(
                entities=analysis.interfaces,
                collection_name="code_classes",
                repository=repository,
                db_name=db_name,
                mongodb=mongodb,
                embedding_service=embedding_service
            )
            total_imported += interface_stats["imported"]
            total_updated += interface_stats["updated"]

            # Import database operations
            db_ops_stats = await self._import_db_operations(
                operations=analysis.database_operations,
                repository=repository,
                db_name=db_name,
                mongodb=mongodb,
                embedding_service=embedding_service
            )
            total_imported += db_ops_stats["imported"]

            # Import event handlers
            event_stats = await self._import_event_handlers(
                handlers=analysis.event_handlers,
                repository=repository,
                db_name=db_name,
                mongodb=mongodb,
                embedding_service=embedding_service
            )
            total_imported += event_stats["imported"]

            duration = time.time() - start_time

            logger.info(
                f"Import complete for {repository}: "
                f"{total_imported} imported, {total_updated} updated "
                f"in {duration:.2f}s"
            )

            return ImportResult(
                success=True,
                repository=repository,
                documents_imported=total_imported,
                documents_updated=total_updated,
                documents_deleted=total_deleted,
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Import failed for {repository}: {e}", exc_info=True)
            return ImportResult(
                success=False,
                repository=repository,
                error=str(e),
                duration_seconds=time.time() - start_time
            )

    async def _import_entities(
        self,
        entities: List[CodeEntity],
        collection_name: str,
        repository: str,
        db_name: str,
        mongodb,
        embedding_service,
    ) -> Dict[str, int]:
        """
        Import code entities to a MongoDB collection.

        Args:
            entities: List of CodeEntity objects to import
            collection_name: Target MongoDB collection
            repository: Repository name for metadata
            db_name: Database prefix for scoping
            mongodb: MongoDB service instance
            embedding_service: Embedding service instance

        Returns:
            Dict with import statistics
        """
        if not entities:
            return {"imported": 0, "updated": 0}

        stats = {"imported": 0, "updated": 0}
        collection = mongodb.db[collection_name]

        # Process in batches
        for batch_start in range(0, len(entities), self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, len(entities))
            batch = entities[batch_start:batch_end]

            # Generate searchable text for each entity
            texts = [self._entity_to_searchable_text(e) for e in batch]

            # Generate embeddings in batch
            embeddings = await embedding_service.generate_embeddings_batch(texts)

            # Build documents for upsert
            for entity, text, embedding in zip(batch, texts, embeddings):
                doc = self._entity_to_document(
                    entity=entity,
                    repository=repository,
                    db_name=db_name,
                    searchable_text=text,
                    embedding=embedding
                )

                # Upsert by unique key (file_path + name + type)
                filter_key = {
                    "repository": repository,
                    "file_path": entity.file_path,
                    "name": entity.name,
                    "entity_type": entity.type
                }

                result = await collection.update_one(
                    filter_key,
                    {"$set": doc},
                    upsert=True
                )

                if result.upserted_id:
                    stats["imported"] += 1
                elif result.modified_count > 0:
                    stats["updated"] += 1

        logger.debug(
            f"Imported {stats['imported']} / updated {stats['updated']} "
            f"{collection_name} entities for {repository}"
        )

        return stats

    async def _import_db_operations(
        self,
        operations: List[Dict[str, Any]],
        repository: str,
        db_name: str,
        mongodb,
        embedding_service,
    ) -> Dict[str, int]:
        """
        Import database operations to MongoDB.

        Args:
            operations: List of database operation dicts from Roslyn
            repository: Repository name
            db_name: Database prefix
            mongodb: MongoDB service instance
            embedding_service: Embedding service instance

        Returns:
            Dict with import statistics
        """
        if not operations:
            return {"imported": 0}

        stats = {"imported": 0}
        collection = mongodb.db["code_dboperations"]

        for op in operations:
            # Generate searchable text
            text = self._db_operation_to_searchable_text(op)

            # Generate embedding
            embedding = await embedding_service.generate_embedding(text)

            doc = {
                "repository": repository,
                "db_name": db_name,
                "operation_type": op.get("operationType", "unknown"),
                "sql_command": op.get("sqlCommand"),
                "stored_procedure": op.get("storedProcedure"),
                "table_name": op.get("tableName"),
                "containing_class": op.get("containingClass"),
                "containing_method": op.get("containingMethod"),
                "file_path": op.get("filePath", ""),
                "line_number": op.get("lineNumber", 0),
                "searchable_text": text,
                "vector": embedding,
                "updated_at": datetime.utcnow().isoformat()
            }

            # Upsert
            filter_key = {
                "repository": repository,
                "file_path": doc["file_path"],
                "line_number": doc["line_number"],
                "operation_type": doc["operation_type"]
            }

            result = await collection.update_one(
                filter_key,
                {"$set": doc},
                upsert=True
            )

            if result.upserted_id:
                stats["imported"] += 1

        return stats

    async def _import_event_handlers(
        self,
        handlers: List[Dict[str, Any]],
        repository: str,
        db_name: str,
        mongodb,
        embedding_service,
    ) -> Dict[str, int]:
        """
        Import event handlers to MongoDB.

        Args:
            handlers: List of event handler dicts from Roslyn
            repository: Repository name
            db_name: Database prefix
            mongodb: MongoDB service instance
            embedding_service: Embedding service instance

        Returns:
            Dict with import statistics
        """
        if not handlers:
            return {"imported": 0}

        stats = {"imported": 0}
        collection = mongodb.db["code_eventhandlers"]

        for handler in handlers:
            # Generate searchable text
            text = self._event_handler_to_searchable_text(handler)

            # Generate embedding
            embedding = await embedding_service.generate_embedding(text)

            doc = {
                "repository": repository,
                "db_name": db_name,
                "event_name": handler.get("eventName", ""),
                "handler_name": handler.get("handlerName", ""),
                "control_name": handler.get("controlName"),
                "control_type": handler.get("controlType"),
                "event_type": handler.get("eventType"),
                "containing_class": handler.get("containingClass"),
                "file_path": handler.get("filePath", ""),
                "line_number": handler.get("lineNumber", 0),
                "searchable_text": text,
                "vector": embedding,
                "updated_at": datetime.utcnow().isoformat()
            }

            # Upsert
            filter_key = {
                "repository": repository,
                "file_path": doc["file_path"],
                "handler_name": doc["handler_name"]
            }

            result = await collection.update_one(
                filter_key,
                {"$set": doc},
                upsert=True
            )

            if result.upserted_id:
                stats["imported"] += 1

        return stats

    def _entity_to_searchable_text(self, entity: CodeEntity) -> str:
        """
        Convert a code entity to searchable text for embedding.

        The text is designed to be semantically meaningful for search queries
        like "method that processes bale weights" or "class for customer management".

        Args:
            entity: CodeEntity to convert

        Returns:
            Searchable text string
        """
        parts = []

        # Type and name
        parts.append(f"{entity.type} {entity.name}")

        # Parent context
        if entity.parent_class:
            parts.append(f"in class {entity.parent_class}")

        # Namespace
        if entity.namespace:
            parts.append(f"namespace {entity.namespace}")

        # Signature (for methods)
        if entity.signature:
            parts.append(f"signature: {entity.signature}")

        # Return type
        if entity.return_type:
            parts.append(f"returns {entity.return_type}")

        # Documentation
        if entity.doc_comment:
            # Clean up XML doc comment
            doc = entity.doc_comment.replace("<summary>", "").replace("</summary>", "")
            doc = doc.replace("<param ", "").replace("</param>", "")
            doc = doc.strip()
            if doc:
                parts.append(doc)

        # Body preview (for methods)
        if entity.body_preview:
            parts.append(f"implementation: {entity.body_preview[:200]}")

        return " | ".join(parts)

    def _db_operation_to_searchable_text(self, op: Dict[str, Any]) -> str:
        """
        Convert a database operation to searchable text.

        Args:
            op: Database operation dict

        Returns:
            Searchable text string
        """
        parts = []

        op_type = op.get("operationType", "database operation")
        parts.append(op_type)

        if op.get("storedProcedure"):
            parts.append(f"stored procedure {op['storedProcedure']}")

        if op.get("tableName"):
            parts.append(f"table {op['tableName']}")

        if op.get("sqlCommand"):
            # Truncate long SQL
            sql = op["sqlCommand"][:200]
            parts.append(f"SQL: {sql}")

        if op.get("containingClass"):
            parts.append(f"in class {op['containingClass']}")

        if op.get("containingMethod"):
            parts.append(f"method {op['containingMethod']}")

        return " | ".join(parts)

    def _event_handler_to_searchable_text(self, handler: Dict[str, Any]) -> str:
        """
        Convert an event handler to searchable text.

        Args:
            handler: Event handler dict

        Returns:
            Searchable text string
        """
        parts = []

        parts.append(f"event handler {handler.get('handlerName', 'unknown')}")

        if handler.get("eventName"):
            parts.append(f"handles {handler['eventName']} event")

        if handler.get("controlName"):
            parts.append(f"on control {handler['controlName']}")

        if handler.get("controlType"):
            parts.append(f"({handler['controlType']})")

        if handler.get("containingClass"):
            parts.append(f"in class {handler['containingClass']}")

        return " | ".join(parts)

    def _entity_to_document(
        self,
        entity: CodeEntity,
        repository: str,
        db_name: str,
        searchable_text: str,
        embedding: List[float]
    ) -> Dict[str, Any]:
        """
        Convert a code entity to a MongoDB document.

        Args:
            entity: CodeEntity to convert
            repository: Repository name
            db_name: Database prefix
            searchable_text: Pre-generated searchable text
            embedding: Pre-generated embedding vector

        Returns:
            MongoDB document dict
        """
        return {
            "repository": repository,
            "db_name": db_name,
            "entity_type": entity.type,
            "name": entity.name,
            "full_name": entity.full_name,
            "file_path": entity.file_path,
            "line_number": entity.line_number,
            "namespace": entity.namespace,
            "parent_class": entity.parent_class,
            "signature": entity.signature,
            "return_type": entity.return_type,
            "parameters": entity.parameters,
            "modifiers": entity.modifiers,
            "doc_comment": entity.doc_comment,
            "body_preview": entity.body_preview,
            "searchable_text": searchable_text,
            "vector": embedding,
            "updated_at": datetime.utcnow().isoformat()
        }

    async def delete_repository_data(
        self,
        repository: str,
        collections: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Delete all data for a repository from MongoDB.

        Args:
            repository: Repository name to delete
            collections: Specific collections to clear. If None, clears all.

        Returns:
            Dict with deletion counts per collection
        """
        mongodb = await self._get_mongodb()

        default_collections = [
            "code_classes",
            "code_methods",
            "code_context",
            "code_dboperations",
            "code_eventhandlers"
        ]

        target_collections = collections or default_collections
        results = {}

        for collection_name in target_collections:
            collection = mongodb.db[collection_name]
            result = await collection.delete_many({"repository": repository})
            results[collection_name] = result.deleted_count
            logger.info(f"Deleted {result.deleted_count} documents from {collection_name}")

        return results

    def import_analysis_sync(
        self,
        analysis: RoslynAnalysisOutput,
        db_name: str
    ) -> ImportResult:
        """
        Import analysis synchronously.

        Convenience method for non-async contexts.

        Args:
            analysis: Roslyn analysis output
            db_name: Database prefix

        Returns:
            ImportResult with statistics
        """
        return asyncio.run(self.import_analysis(analysis, db_name))
