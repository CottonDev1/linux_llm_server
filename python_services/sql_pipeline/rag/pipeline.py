"""
SQL RAG Pipeline - Orchestrates the full RAG workflow

Pipeline stages:
1. EXTRACTION - Extract schema/SP data from SQL Server (sql_pipeline.extraction module)
2. ENRICHMENT - Generate LLM summaries (sql_pipeline.extraction.summarizers)
3. EMBEDDING  - Generate vector embeddings from enriched summaries (this module)
4. STORAGE    - Store in MongoDB with vectors (mongodb_service)
5. RETRIEVAL  - Semantic search (mongodb_service)

This pipeline focuses on stage 3 (embedding) and orchestrating the flow.
"""
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .embedder import SchemaEmbedder, ProcedureEmbedder
from database_name_parser import normalize_database_name


@dataclass
class PipelineStats:
    """Statistics from pipeline execution"""
    schemas_processed: int = 0
    schemas_embedded: int = 0
    schemas_failed: int = 0
    procedures_processed: int = 0
    procedures_embedded: int = 0
    procedures_failed: int = 0
    elapsed_seconds: float = 0.0


class SQLRAGPipeline:
    """
    Orchestrates the SQL RAG pipeline.

    This class handles the embedding stage of the pipeline,
    taking summarized data and generating vector embeddings
    optimized for semantic search.
    """

    def __init__(self, mongodb_service, embedding_service, verbose: bool = True):
        """
        Initialize the pipeline.

        Args:
            mongodb_service: MongoDBService instance for storage
            embedding_service: EmbeddingService for vector generation
            verbose: Whether to print progress
        """
        self.mongodb = mongodb_service
        self.embedding_service = embedding_service
        self.verbose = verbose

        # Initialize embedders
        self.schema_embedder = SchemaEmbedder(embedding_service)
        self.procedure_embedder = ProcedureEmbedder(embedding_service)

    def log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)

    async def embed_schema(
        self,
        database: str,
        table_name: str,
        columns: List[Dict],
        primary_keys: List[str] = None,
        foreign_keys: List[Dict] = None,
        related_tables: List[str] = None,
        sample_values: Dict = None,
        summary: str = None,
        purpose: str = None,
        keywords: List[str] = None
    ) -> bool:
        """
        Generate embedding for a schema and update MongoDB.

        This is the key step that creates searchable vectors from
        the LLM-generated summary.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding from summary + schema info
            result = await self.schema_embedder.generate_embedding(
                table_name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                related_tables=related_tables,
                sample_values=sample_values,
                summary=summary,
                purpose=purpose,
                keywords=keywords
            )

            # Update MongoDB document with new embedding
            collection = self.mongodb.db["sql_schema_context"]
            normalized_db = normalize_database_name(database)

            await collection.update_one(
                {"database": normalized_db, "table_name": table_name},
                {"$set": {
                    "embedding_text": result.embedding_text,
                    "vector": result.vector,
                    "embedded_at": datetime.utcnow().isoformat()
                }}
            )

            return True

        except Exception as e:
            self.log(f"    ERROR embedding schema {table_name}: {e}")
            return False

    async def embed_procedure(
        self,
        database: str,
        procedure_name: str,
        schema: str = 'dbo',
        parameters: List[Dict] = None,
        summary: str = None,
        operations: List[str] = None,
        tables_referenced: List[str] = None,
        keywords: List[str] = None,
        input_description: str = None,
        output_description: str = None
    ) -> bool:
        """
        Generate embedding for a stored procedure and update MongoDB.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding from summary + procedure info
            result = await self.procedure_embedder.generate_embedding(
                procedure_name=procedure_name,
                schema=schema,
                parameters=parameters,
                summary=summary,
                operations=operations,
                tables_referenced=tables_referenced,
                keywords=keywords,
                input_description=input_description,
                output_description=output_description
            )

            # Update MongoDB document with new embedding
            collection = self.mongodb.db["sql_stored_procedures"]
            normalized_db = normalize_database_name(database)

            await collection.update_one(
                {"database": normalized_db, "procedure_name": procedure_name},
                {"$set": {
                    "embedding_text": result.embedding_text,
                    "vector": result.vector,
                    "embedded_at": datetime.utcnow().isoformat()
                }}
            )

            return True

        except Exception as e:
            self.log(f"    ERROR embedding procedure {procedure_name}: {e}")
            return False

    async def embed_all_schemas(self, database: str) -> Tuple[int, int]:
        """
        Embed all schemas for a database that have summaries.

        Returns:
            Tuple of (success_count, failure_count)
        """
        collection = self.mongodb.db["sql_schema_context"]
        normalized_db = normalize_database_name(database)

        # Find schemas with summaries
        schemas = await collection.find({
            "database": normalized_db,
            "summary": {"$type": "string", "$ne": ""}
        }).to_list(length=1000)

        self.log(f"\n  Found {len(schemas)} schemas with summaries to embed")

        success = 0
        failed = 0

        for i, schema in enumerate(schemas):
            table_name = schema.get('table_name', 'Unknown')
            self.log(f"    [{i+1}/{len(schemas)}] Embedding {table_name}")

            result = await self.embed_schema(
                database=normalized_db,
                table_name=table_name,
                columns=schema.get('columns', []),
                primary_keys=schema.get('primary_keys', []),
                foreign_keys=schema.get('foreign_keys', []),
                related_tables=schema.get('related_tables', []),
                sample_values=schema.get('sample_values', {}),
                summary=schema.get('summary'),
                purpose=schema.get('purpose'),
                keywords=schema.get('keywords', [])
            )

            if result:
                success += 1
            else:
                failed += 1

        return success, failed

    async def embed_all_procedures(self, database: str) -> Tuple[int, int]:
        """
        Embed all stored procedures for a database that have summaries.

        Returns:
            Tuple of (success_count, failure_count)
        """
        collection = self.mongodb.db["sql_stored_procedures"]
        normalized_db = normalize_database_name(database)

        # Find procedures with summaries
        procedures = await collection.find({
            "database": normalized_db,
            "summary": {"$type": "string", "$ne": ""}
        }).to_list(length=1000)

        self.log(f"\n  Found {len(procedures)} procedures with summaries to embed")

        success = 0
        failed = 0

        for i, proc in enumerate(procedures):
            proc_name = proc.get('procedure_name', 'Unknown')
            self.log(f"    [{i+1}/{len(procedures)}] Embedding {proc_name}")

            result = await self.embed_procedure(
                database=normalized_db,
                procedure_name=proc_name,
                schema=proc.get('schema', 'dbo'),
                parameters=proc.get('parameters', []),
                summary=proc.get('summary'),
                operations=proc.get('operations', []),
                tables_referenced=proc.get('tables_referenced', []),
                keywords=proc.get('keywords', []),
                input_description=proc.get('input_description'),
                output_description=proc.get('output_description')
            )

            if result:
                success += 1
            else:
                failed += 1

        return success, failed

    async def run_embedding_stage(self, database: str) -> PipelineStats:
        """
        Run the embedding stage for all summarized data.

        This should be called AFTER summarization is complete.
        It generates vector embeddings from the LLM summaries.
        """
        import time
        start_time = time.time()

        self.log("=" * 70)
        self.log(f"RAG Pipeline - Embedding Stage for {database}")
        self.log("=" * 70)

        stats = PipelineStats()

        # Embed schemas
        self.log("\n" + "-" * 70)
        self.log("Phase 1: Embedding Table Schemas")
        self.log("-" * 70)

        schema_success, schema_failed = await self.embed_all_schemas(database)
        stats.schemas_embedded = schema_success
        stats.schemas_failed = schema_failed
        stats.schemas_processed = schema_success + schema_failed

        self.log(f"\n  Schemas: {schema_success} embedded, {schema_failed} failed")

        # Embed procedures
        self.log("\n" + "-" * 70)
        self.log("Phase 2: Embedding Stored Procedures")
        self.log("-" * 70)

        proc_success, proc_failed = await self.embed_all_procedures(database)
        stats.procedures_embedded = proc_success
        stats.procedures_failed = proc_failed
        stats.procedures_processed = proc_success + proc_failed

        self.log(f"\n  Procedures: {proc_success} embedded, {proc_failed} failed")

        stats.elapsed_seconds = time.time() - start_time

        # Summary
        self.log("\n" + "=" * 70)
        self.log("EMBEDDING COMPLETE")
        self.log("=" * 70)
        self.log(f"\n  Schemas: {stats.schemas_embedded}/{stats.schemas_processed} embedded")
        self.log(f"  Procedures: {stats.procedures_embedded}/{stats.procedures_processed} embedded")
        self.log(f"  Time: {stats.elapsed_seconds:.1f}s")

        return stats

    async def get_pipeline_status(self, database: str) -> Dict:
        """
        Get the current status of the RAG pipeline for a database.

        Returns counts of items at each stage:
        - extracted: Has raw data
        - summarized: Has LLM summary
        - embedded: Has vector embedding
        """
        normalized_db = normalize_database_name(database)

        schema_col = self.mongodb.db["sql_schema_context"]
        proc_col = self.mongodb.db["sql_stored_procedures"]

        # Schema stats
        schemas_total = await schema_col.count_documents({"database": normalized_db})
        schemas_summarized = await schema_col.count_documents({
            "database": normalized_db,
            "summary": {"$type": "string", "$ne": ""}
        })
        schemas_embedded = await schema_col.count_documents({
            "database": normalized_db,
            "embedded_at": {"$exists": True}
        })

        # Procedure stats
        procs_total = await proc_col.count_documents({"database": normalized_db})
        procs_summarized = await proc_col.count_documents({
            "database": normalized_db,
            "summary": {"$type": "string", "$ne": ""}
        })
        procs_embedded = await proc_col.count_documents({
            "database": normalized_db,
            "embedded_at": {"$exists": True}
        })

        return {
            "database": database,
            "schemas": {
                "extracted": schemas_total,
                "summarized": schemas_summarized,
                "embedded": schemas_embedded,
                "pending_summary": schemas_total - schemas_summarized,
                "pending_embed": schemas_summarized - schemas_embedded
            },
            "procedures": {
                "extracted": procs_total,
                "summarized": procs_summarized,
                "embedded": procs_embedded,
                "pending_summary": procs_total - procs_summarized,
                "pending_embed": procs_summarized - procs_embedded
            },
            "ready_for_search": schemas_embedded > 0 or procs_embedded > 0
        }
