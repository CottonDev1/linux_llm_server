"""SQL Summarization routes for generating schema summaries with LLM."""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List

from mongodb import get_mongodb_service

router = APIRouter(prefix="/summarize", tags=["SQL Summarization"])


@router.post("/stored-procedures")
async def summarize_stored_procedures(
    database: str = Body(..., description="Database lookup key"),
    llm_url: str = Body("http://localhost:11434", description="LLM API URL"),
    model: str = Body("llama3.2", description="LLM model name"),
    limit: int = Body(10, ge=1, le=100, description="Max procedures to summarize"),
    procedure_names: Optional[List[str]] = Body(None, description="Specific procedures to summarize")
):
    """
    Generate LLM summaries for stored procedures.

    Summaries are stored back to MongoDB with the procedure record,
    improving semantic search by embedding meaningful descriptions.
    """
    from sql_extraction import ProcedureSummarizer

    try:
        summarizer = ProcedureSummarizer(llm_url=llm_url, model=model)

        # Check LLM availability
        if not await summarizer.check_llm_available():
            raise HTTPException(
                status_code=503,
                detail=f"LLM not available or model '{model}' not found at {llm_url}"
            )

        mongodb = get_mongodb_service()

        # Get procedures from MongoDB that need summarization
        collection = mongodb.db[mongodb.COLLECTION_SQL_STORED_PROCEDURES]

        query = {"database": database}
        if procedure_names:
            query["procedure_name"] = {"$in": procedure_names}

        procedures = await collection.find(query).limit(limit).to_list(length=limit)

        if not procedures:
            return {
                "success": True,
                "message": f"No procedures found for database '{database}'",
                "summarized": 0
            }

        # Summarize each procedure
        summarized = 0
        errors = []

        for proc in procedures:
            try:
                summary = await summarizer.summarize_procedure(
                    procedure_name=f"{proc.get('schema', 'dbo')}.{proc['procedure_name']}",
                    procedure_info={
                        'schema': proc.get('schema', 'dbo'),
                        'parameters': proc.get('parameters', []),
                        'definition': proc.get('definition', '')
                    },
                    verbose=True
                )

                # Update the procedure record with summary
                await collection.update_one(
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
                summarized += 1

            except Exception as e:
                errors.append(f"{proc['procedure_name']}: {str(e)}")

        return {
            "success": True,
            "database": database,
            "summarized": summarized,
            "total_found": len(procedures),
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemas")
async def summarize_schemas(
    database: str = Body(..., description="Database lookup key"),
    llm_url: str = Body("http://localhost:11434", description="LLM API URL"),
    model: str = Body("llama3.2", description="LLM model name"),
    limit: int = Body(10, ge=1, le=100, description="Max schemas to summarize"),
    table_names: Optional[List[str]] = Body(None, description="Specific tables to summarize")
):
    """
    Generate LLM summaries for table schemas.

    This is a key nilenso recommendation - embedding table descriptions
    rather than raw DDL improves text-to-SQL retrieval significantly.
    """
    from sql_extraction import SchemaSummarizer

    try:
        summarizer = SchemaSummarizer(llm_url=llm_url, model=model)

        # Check LLM availability
        if not await summarizer.check_llm_available():
            raise HTTPException(
                status_code=503,
                detail=f"LLM not available or model '{model}' not found at {llm_url}"
            )

        mongodb = get_mongodb_service()

        # Get schemas from MongoDB that need summarization
        collection = mongodb.db[mongodb.COLLECTION_SQL_SCHEMA_CONTEXT]

        query = {"database": database}
        if table_names:
            query["table_name"] = {"$in": table_names}

        schemas = await collection.find(query).limit(limit).to_list(length=limit)

        if not schemas:
            return {
                "success": True,
                "message": f"No schemas found for database '{database}'",
                "summarized": 0
            }

        # Summarize each schema
        summarized = 0
        errors = []

        for schema in schemas:
            try:
                summary = await summarizer.summarize_schema(
                    table_name=schema['table_name'],
                    schema_info=schema.get('schema_info', {}),
                    verbose=True
                )

                # Update the schema record with summary
                await collection.update_one(
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
                summarized += 1

            except Exception as e:
                errors.append(f"{schema['table_name']}: {str(e)}")

        return {
            "success": True,
            "database": database,
            "summarized": summarized,
            "total_found": len(schemas),
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def check_llm_status(
    llm_url: str = Query("http://localhost:11434", description="LLM API URL"),
    model: str = Query("llama3.2", description="Model to check")
):
    """
    Check if LLM is available and the specified model is loaded.
    """
    from sql_extraction import ProcedureSummarizer

    summarizer = ProcedureSummarizer(llm_url=llm_url, model=model)
    available = await summarizer.check_llm_available()

    return {
        "llm_url": llm_url,
        "model": model,
        "available": available,
        "message": "Ready for summarization" if available else f"LLM or model '{model}' not available"
    }
