"""
Prefect SQL RAG Pipeline

Orchestrates the SQL RAG workflow with:
1. Schema Extraction - Extract table/procedure metadata from SQL Server
2. Summarization - Generate LLM summaries
3. Embedding - Generate vector embeddings using sentence-transformers
4. Storage - Store vectors in MongoDB for semantic search

Features:
- Built-in retries with exponential backoff
- Detailed logging and artifact tracking
- Async-native execution
- Visual progress in Prefect dashboard
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class ExtractionResult:
    """Result from schema extraction task"""
    database: str
    tables_extracted: int = 0
    procedures_extracted: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


@dataclass
class SummarizationResult:
    """Result from summarization task"""
    database: str
    schemas_summarized: int = 0
    procedures_summarized: int = 0
    llm_calls: int = 0
    tokens_used: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


@dataclass
class EmbeddingResult:
    """Result from embedding generation task"""
    database: str
    schemas_embedded: int = 0
    procedures_embedded: int = 0
    vectors_generated: int = 0
    cache_hits: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


@dataclass
class StorageResult:
    """Result from vector storage task"""
    database: str
    documents_stored: int = 0
    indexes_updated: bool = False
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


@task(
    name="extract_schemas",
    description="Extract schema and stored procedure metadata from SQL Server",
    retries=2,
    retry_delay_seconds=30,
    tags=["sql-server", "extraction"]
)
async def extract_schemas_task(
    server: str,
    database: str,
    connection_config: Dict[str, Any]
) -> ExtractionResult:
    """
    Extract schema and stored procedure metadata from SQL Server.

    Args:
        server: SQL Server hostname
        database: Database name to extract
        connection_config: Connection parameters (user, password, etc.)

    Returns:
        ExtractionResult with counts and any errors
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Starting schema extraction for {database} on {server}")

    result = ExtractionResult(database=database)

    try:
        import sys
        sys.path.insert(0, '..')
        from sql_pipeline.extraction.schema_extractor import SchemaExtractor

        extractor = SchemaExtractor(
            server=server,
            database=database,
            **connection_config
        )

        extraction_data = await extractor.extract_single_database(database)
        result.tables_extracted = extraction_data.get('tables', 0)
        result.procedures_extracted = extraction_data.get('procedures', 0)

        logger.info(f"Extracted {result.tables_extracted} tables, {result.procedures_extracted} procedures")

    except Exception as e:
        error_msg = f"Extraction failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time

    # Create Prefect artifact for tracking
    await create_markdown_artifact(
        key="extraction-summary",
        markdown=f"""
## Schema Extraction Results
- **Database**: {database}
- **Server**: {server}
- **Tables Extracted**: {result.tables_extracted}
- **Procedures Extracted**: {result.procedures_extracted}
- **Duration**: {result.duration_seconds:.2f}s
- **Status**: {"Success" if result.success else "Failed"}
{"- **Errors**: " + ", ".join(result.errors) if result.errors else ""}
        """,
        description=f"Schema extraction for {database}"
    )

    return result


@task(
    name="summarize_schemas",
    description="Generate LLM summaries for extracted schemas and procedures",
    retries=3,
    retry_delay_seconds=60,
    tags=["llm", "summarization"]
)
async def summarize_schemas_task(
    extraction_result: ExtractionResult,
    llm_host: str = "http://localhost:11434",
    model: str = "llama3.2"
) -> SummarizationResult:
    """
    Generate LLM summaries for extracted schemas and procedures.

    Args:
        extraction_result: Result from extraction task
        llm_host: LLM API endpoint
        model: LLM model to use for summarization

    Returns:
        SummarizationResult with counts and metrics
    """
    logger = get_run_logger()
    start_time = time.time()

    database = extraction_result.database
    logger.info(f"Starting summarization for {database}")

    result = SummarizationResult(database=database)

    if not extraction_result.success:
        result.errors.append("Skipped: extraction failed")
        result.success = False
        result.duration_seconds = time.time() - start_time
        return result

    try:
        import sys
        sys.path.insert(0, '..')
        from sql_pipeline.extraction.schema_summarizer import SchemaSummarizer
        from sql_pipeline.extraction.procedure_summarizer import ProcedureSummarizer
        from mongodb import MongoDBService

        mongodb = MongoDBService()
        await mongodb.connect()

        schema_summarizer = SchemaSummarizer(
            mongodb_service=mongodb,
            llm_host=llm_host,
            model=model
        )

        proc_summarizer = ProcedureSummarizer(
            mongodb_service=mongodb,
            llm_host=llm_host,
            model=model
        )

        # Summarize schemas
        schema_result = await schema_summarizer.summarize_database(database)
        result.schemas_summarized = schema_result.get('summarized', 0)
        result.llm_calls += schema_result.get('llm_calls', 0)

        # Summarize procedures
        proc_result = await proc_summarizer.summarize_database(database)
        result.procedures_summarized = proc_result.get('summarized', 0)
        result.llm_calls += proc_result.get('llm_calls', 0)

        logger.info(f"Summarized {result.schemas_summarized} schemas, {result.procedures_summarized} procedures")

    except Exception as e:
        error_msg = f"Summarization failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time

    # Create Prefect artifact
    await create_markdown_artifact(
        key="summarization-summary",
        markdown=f"""
## Summarization Results
- **Database**: {database}
- **Schemas Summarized**: {result.schemas_summarized}
- **Procedures Summarized**: {result.procedures_summarized}
- **LLM Calls**: {result.llm_calls}
- **Duration**: {result.duration_seconds:.2f}s
- **Status**: {"Success" if result.success else "Failed"}
        """,
        description=f"Summarization for {database}"
    )

    return result


@task(
    name="generate_embeddings",
    description="Generate vector embeddings from summarized content",
    retries=2,
    retry_delay_seconds=30,
    tags=["embeddings", "vectors"]
)
async def generate_embeddings_task(
    summarization_result: SummarizationResult,
    model_name: str = "all-MiniLM-L6-v2"
) -> EmbeddingResult:
    """
    Generate vector embeddings from summarized content.

    Args:
        summarization_result: Result from summarization task
        model_name: Sentence transformer model to use

    Returns:
        EmbeddingResult with vector counts and cache metrics
    """
    logger = get_run_logger()
    start_time = time.time()

    database = summarization_result.database
    logger.info(f"Starting embedding generation for {database}")

    result = EmbeddingResult(database=database)

    if not summarization_result.success:
        result.errors.append("Skipped: summarization failed")
        result.success = False
        result.duration_seconds = time.time() - start_time
        return result

    try:
        import sys
        sys.path.insert(0, '..')
        from embedding_service import EmbeddingService
        from mongodb import MongoDBService
        from sql_pipeline.rag.pipeline import SQLRAGPipeline

        embedding_service = EmbeddingService(model_name=model_name)
        mongodb = MongoDBService()
        await mongodb.connect()

        pipeline = SQLRAGPipeline(
            mongodb_service=mongodb,
            embedding_service=embedding_service,
            verbose=True
        )

        stats = await pipeline.run_embedding_stage(database)

        result.schemas_embedded = stats.schemas_embedded
        result.procedures_embedded = stats.procedures_embedded
        result.vectors_generated = result.schemas_embedded + result.procedures_embedded

        # Get cache stats
        cache_stats = embedding_service.get_cache_stats()
        result.cache_hits = cache_stats.get('hits', 0)

        logger.info(f"Generated {result.vectors_generated} vectors ({result.cache_hits} cache hits)")

    except Exception as e:
        error_msg = f"Embedding generation failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time

    # Create Prefect artifact
    cache_rate = result.cache_hits / max(result.vectors_generated, 1) * 100
    await create_markdown_artifact(
        key="embedding-summary",
        markdown=f"""
## Embedding Results
- **Database**: {database}
- **Schemas Embedded**: {result.schemas_embedded}
- **Procedures Embedded**: {result.procedures_embedded}
- **Total Vectors**: {result.vectors_generated}
- **Cache Hits**: {result.cache_hits} ({cache_rate:.1f}%)
- **Duration**: {result.duration_seconds:.2f}s
- **Status**: {"Success" if result.success else "Failed"}
        """,
        description=f"Embeddings for {database}"
    )

    return result


@task(
    name="store_vectors",
    description="Store vectors in MongoDB and ensure indexes are created",
    retries=2,
    retry_delay_seconds=15,
    tags=["mongodb", "storage"]
)
async def store_vectors_task(
    embedding_result: EmbeddingResult,
    create_indexes: bool = True
) -> StorageResult:
    """
    Store vectors in MongoDB and ensure indexes are created.

    Args:
        embedding_result: Result from embedding task
        create_indexes: Whether to create/update vector indexes

    Returns:
        StorageResult with storage confirmation
    """
    logger = get_run_logger()
    start_time = time.time()

    database = embedding_result.database
    logger.info(f"Verifying vector storage for {database}")

    result = StorageResult(database=database)

    if not embedding_result.success:
        result.errors.append("Skipped: embedding failed")
        result.success = False
        result.duration_seconds = time.time() - start_time
        return result

    try:
        import sys
        sys.path.insert(0, '..')
        from mongodb import MongoDBService
        from database_name_parser import normalize_database_name

        mongodb = MongoDBService()
        await mongodb.connect()

        normalized_db = normalize_database_name(database)

        # Count stored documents with vectors
        schema_col = mongodb.db["sql_schema_context"]
        proc_col = mongodb.db["sql_stored_procedures"]

        schema_count = await schema_col.count_documents({
            "database": normalized_db,
            "vector": {"$exists": True}
        })

        proc_count = await proc_col.count_documents({
            "database": normalized_db,
            "vector": {"$exists": True}
        })

        result.documents_stored = schema_count + proc_count

        # Create indexes if requested
        if create_indexes:
            try:
                await schema_col.create_index([("vector", 1)])
                await proc_col.create_index([("vector", 1)])
                result.indexes_updated = True
            except Exception as idx_err:
                logger.warning(f"Index creation skipped: {idx_err}")

        logger.info(f"Verified {result.documents_stored} documents with vectors")

    except Exception as e:
        error_msg = f"Storage verification failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time

    # Create final Prefect artifact
    await create_markdown_artifact(
        key="storage-summary",
        markdown=f"""
## Storage Results
- **Database**: {database}
- **Documents Stored**: {result.documents_stored}
- **Indexes Updated**: {"Yes" if result.indexes_updated else "No"}
- **Duration**: {result.duration_seconds:.2f}s
- **Status**: {"Success" if result.success else "Failed"}
        """,
        description=f"Storage verification for {database}"
    )

    return result


@flow(
    name="sql-rag-pipeline",
    description="Complete SQL RAG Pipeline for database schema indexing",
    retries=1,
    retry_delay_seconds=120
)
async def sql_rag_flow(
    server: str,
    database: str,
    connection_config: Dict[str, Any],
    llm_host: str = "http://localhost:11434",
    llm_model: str = "llama3.2",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Complete SQL RAG Pipeline for database schema indexing.

    This flow:
    1. Extracts schema metadata from SQL Server
    2. Generates LLM summaries for semantic understanding
    3. Creates vector embeddings for similarity search
    4. Stores vectors in MongoDB with proper indexes

    Args:
        server: SQL Server hostname
        database: Database to process
        connection_config: SQL Server connection parameters
        llm_host: LLM API endpoint
        llm_model: Model for summarization
        embedding_model: Model for embeddings

    Returns:
        Dict with complete pipeline results
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting SQL RAG pipeline for {database} on {server}")

    # Step 1: Extract schemas
    extraction_result = await extract_schemas_task(
        server=server,
        database=database,
        connection_config=connection_config
    )

    # Step 2: Generate summaries
    summarization_result = await summarize_schemas_task(
        extraction_result=extraction_result,
        llm_host=llm_host,
        model=llm_model
    )

    # Step 3: Generate embeddings
    embedding_result = await generate_embeddings_task(
        summarization_result=summarization_result,
        model_name=embedding_model
    )

    # Step 4: Verify storage
    storage_result = await store_vectors_task(
        embedding_result=embedding_result,
        create_indexes=True
    )

    total_duration = time.time() - flow_start

    # Create final flow summary artifact
    all_errors = (
        extraction_result.errors +
        summarization_result.errors +
        embedding_result.errors +
        storage_result.errors
    )
    overall_success = storage_result.success and not all_errors

    await create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"""
# SQL RAG Pipeline Complete

## Overview
- **Database**: {database}
- **Server**: {server}
- **Total Duration**: {total_duration:.2f}s
- **Status**: {"Success" if overall_success else "Failed"}

## Stage Results
| Stage | Items | Duration | Status |
|-------|-------|----------|--------|
| Extraction | {extraction_result.tables_extracted} tables, {extraction_result.procedures_extracted} procs | {extraction_result.duration_seconds:.1f}s | {"OK" if extraction_result.success else "FAIL"} |
| Summarization | {summarization_result.schemas_summarized} schemas, {summarization_result.procedures_summarized} procs | {summarization_result.duration_seconds:.1f}s | {"OK" if summarization_result.success else "FAIL"} |
| Embedding | {embedding_result.vectors_generated} vectors | {embedding_result.duration_seconds:.1f}s | {"OK" if embedding_result.success else "FAIL"} |
| Storage | {storage_result.documents_stored} docs | {storage_result.duration_seconds:.1f}s | {"OK" if storage_result.success else "FAIL"} |

{"## Errors" + chr(10) + chr(10).join(f"- {e}" for e in all_errors) if all_errors else ""}
        """,
        description=f"SQL RAG Pipeline summary for {database}"
    )

    return {
        "success": overall_success,
        "database": database,
        "server": server,
        "total_duration_seconds": total_duration,
        "stages": {
            "extraction": {
                "tables": extraction_result.tables_extracted,
                "procedures": extraction_result.procedures_extracted,
                "duration": extraction_result.duration_seconds,
                "success": extraction_result.success
            },
            "summarization": {
                "schemas": summarization_result.schemas_summarized,
                "procedures": summarization_result.procedures_summarized,
                "llm_calls": summarization_result.llm_calls,
                "duration": summarization_result.duration_seconds,
                "success": summarization_result.success
            },
            "embedding": {
                "vectors": embedding_result.vectors_generated,
                "cache_hits": embedding_result.cache_hits,
                "duration": embedding_result.duration_seconds,
                "success": embedding_result.success
            },
            "storage": {
                "documents": storage_result.documents_stored,
                "indexes_updated": storage_result.indexes_updated,
                "duration": storage_result.duration_seconds,
                "success": storage_result.success
            }
        },
        "errors": all_errors
    }


def run_sql_rag_flow(
    server: str,
    database: str,
    user: str,
    password: str,
    llm_host: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Convenience function to run the SQL RAG flow synchronously.

    Example:
        from prefect_pipelines import run_sql_rag_flow

        result = run_sql_rag_flow(
            server="NCSQLTEST",
            database="EWRReporting",
            user="EWRUser",
            password="your_password"
        )
    """
    connection_config = {
        "user": user,
        "password": password,
        "trust_server_certificate": True,
        "encrypt": False
    }

    return asyncio.run(sql_rag_flow(
        server=server,
        database=database,
        connection_config=connection_config,
        llm_host=llm_host
    ))


if __name__ == "__main__":
    # Test run
    print("Testing SQL RAG flow...")
    result = run_sql_rag_flow(
        server="NCSQLTEST",
        database="EWRReporting",
        user="EWRUser",
        password="test"
    )
    print(f"Result: {result}")
