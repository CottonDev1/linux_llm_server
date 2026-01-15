"""Pipeline routes for SQL extraction pipeline management."""
import time
from fastapi import APIRouter

from data_models import PipelineRunRequest, PipelineSummarizeRequest, PipelineEmbedRequest
from database_name_parser import normalize_database_name
from mongodb import get_mongodb_service

router = APIRouter(prefix="/pipeline", tags=["RAG Pipeline"])


@router.get("/status/{database}")
async def get_pipeline_status(database: str):
    """
    Get the current status of the RAG pipeline for a database.

    Returns counts at each stage:
    - extracted: Has raw data in MongoDB
    - summarized: Has LLM summary
    - embedded: Has vector embedding ready for search
    """
    from embedding_service import get_embedding_service
    from rag_pipeline import SQLRAGPipeline

    mongodb = get_mongodb_service()
    embedding_service = get_embedding_service()
    await embedding_service.initialize()

    pipeline = SQLRAGPipeline(mongodb, embedding_service, verbose=False)
    status = await pipeline.get_pipeline_status(database)

    return {
        "success": True,
        **status
    }


@router.post("/run")
async def run_full_pipeline(request: PipelineRunRequest):
    """
    Run the full RAG pipeline for a database.

    Pipeline stages:
    1. EXTRACTION - Extract schema/SP data from SQL Server (optional, if config provided)
    2. SUMMARIZATION - Generate LLM summaries
    3. EMBEDDING - Generate vector embeddings from summaries

    This is the recommended way to build searchable SQL context.
    Following nilenso best practices: embed LLM summaries, not raw SQL.
    """
    from sql_extraction import ProcedureSummarizer, SchemaSummarizer
    from embedding_service import get_embedding_service
    from rag_pipeline import SQLRAGPipeline

    # Extract request parameters
    database = request.database
    llm_url = request.llm_url
    model = request.model
    skip_extraction = request.skip_extraction
    skip_summarization = request.skip_summarization
    skip_embedding = request.skip_embedding

    start_time = time.time()
    results = {
        "database": database,
        "stages": {},
        "success": True
    }

    mongodb = get_mongodb_service()
    normalized_db = normalize_database_name(database)

    # Stage 1: Check existing data
    sp_collection = mongodb.db["sql_stored_procedures"]
    schema_collection = mongodb.db["sql_schema_context"]

    total_procs = await sp_collection.count_documents({"database": normalized_db})
    total_schemas = await schema_collection.count_documents({"database": normalized_db})

    results["stages"]["extraction"] = {
        "skipped": skip_extraction,
        "procedures": total_procs,
        "schemas": total_schemas
    }

    if total_procs == 0 and total_schemas == 0:
        return {
            "success": False,
            "error": f"No data found for database '{database}'. Run extraction first.",
            "database": database
        }

    # Stage 2: Summarization
    if not skip_summarization:
        proc_summarizer = ProcedureSummarizer(llm_url=llm_url, model=model)
        schema_summarizer = SchemaSummarizer(llm_url=llm_url, model=model)

        # Check LLM availability
        if not await proc_summarizer.check_llm_available():
            return {
                "success": False,
                "error": f"LLM not available or model '{model}' not found at {llm_url}",
                "database": database
            }

        # Summarize procedures without summaries
        procs_needing_summary = await sp_collection.find({
            "database": normalized_db,
            "$or": [
                {"summary": {"$exists": False}},
                {"summary": None},
                {"summary": ""}
            ]
        }).to_list(length=1000)

        proc_summarized = 0
        proc_errors = 0

        for proc in procs_needing_summary:
            try:
                proc_name = f"{proc.get('schema', 'dbo')}.{proc['procedure_name']}"
                summary = await proc_summarizer.summarize_procedure(
                    procedure_name=proc_name,
                    procedure_info={
                        'schema': proc.get('schema', 'dbo'),
                        'parameters': proc.get('parameters', []),
                        'definition': proc.get('definition', '')
                    },
                    verbose=False
                )

                await sp_collection.update_one(
                    {"_id": proc["_id"]},
                    {"$set": {
                        "summary": summary.summary,
                        "operations": summary.operations,
                        "tables_referenced": summary.tables_referenced,
                        "keywords": summary.keywords,
                        "input_description": summary.input_description,
                        "output_description": summary.output_description
                    }}
                )
                proc_summarized += 1
            except Exception as e:
                proc_errors += 1

        # Summarize schemas without summaries
        schemas_needing_summary = await schema_collection.find({
            "database": normalized_db,
            "$or": [
                {"summary": {"$exists": False}},
                {"summary": None},
                {"summary": ""}
            ]
        }).to_list(length=1000)

        schema_summarized = 0
        schema_errors = 0

        for schema in schemas_needing_summary:
            try:
                schema_info = {
                    'columns': schema.get('columns', []),
                    'primaryKeys': schema.get('primary_keys', []),
                    'foreignKeys': schema.get('foreign_keys', []),
                    'relatedTables': schema.get('related_tables', []),
                    'sampleValues': schema.get('sample_values', {})
                }

                summary = await schema_summarizer.summarize_schema(
                    table_name=schema['table_name'],
                    schema_info=schema_info,
                    verbose=False
                )

                await schema_collection.update_one(
                    {"_id": schema["_id"]},
                    {"$set": {
                        "summary": summary.summary,
                        "purpose": summary.purpose,
                        "key_columns": summary.key_columns,
                        "relationships": summary.relationships,
                        "keywords": summary.keywords,
                        "common_queries": summary.common_queries
                    }}
                )
                schema_summarized += 1
            except Exception as e:
                schema_errors += 1

        results["stages"]["summarization"] = {
            "procedures_summarized": proc_summarized,
            "procedures_errors": proc_errors,
            "schemas_summarized": schema_summarized,
            "schemas_errors": schema_errors
        }
    else:
        results["stages"]["summarization"] = {"skipped": True}

    # Stage 3: Embedding
    if not skip_embedding:
        embedding_service = get_embedding_service()
        await embedding_service.initialize()

        pipeline = SQLRAGPipeline(mongodb, embedding_service, verbose=False)
        stats = await pipeline.run_embedding_stage(database)

        results["stages"]["embedding"] = {
            "schemas_embedded": stats.schemas_embedded,
            "schemas_failed": stats.schemas_failed,
            "procedures_embedded": stats.procedures_embedded,
            "procedures_failed": stats.procedures_failed
        }
    else:
        results["stages"]["embedding"] = {"skipped": True}

    results["elapsed_seconds"] = time.time() - start_time

    # Get final status
    embedding_service = get_embedding_service()
    await embedding_service.initialize()
    pipeline = SQLRAGPipeline(mongodb, embedding_service, verbose=False)
    final_status = await pipeline.get_pipeline_status(database)
    results["final_status"] = final_status

    return results


@router.post("/summarize")
async def run_summarization_stage(request: PipelineSummarizeRequest):
    """
    Run only the summarization stage of the pipeline.

    Generates LLM summaries for all schemas and procedures
    that don't already have summaries.
    """
    full_request = PipelineRunRequest(
        database=request.database,
        llm_url=request.llm_url,
        model=request.model,
        skip_extraction=True,
        skip_summarization=False,
        skip_embedding=True
    )
    return await run_full_pipeline(full_request)


@router.post("/embed")
async def run_embedding_stage(request: PipelineEmbedRequest):
    """
    Run only the embedding stage of the pipeline.

    Generates vector embeddings for all schemas and procedures
    that have summaries. Must run summarization first.
    """
    full_request = PipelineRunRequest(
        database=request.database,
        skip_extraction=True,
        skip_summarization=True,
        skip_embedding=False
    )
    return await run_full_pipeline(full_request)
