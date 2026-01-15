"""SQL Extraction routes for extracting schema information from databases."""
from fastapi import APIRouter, HTTPException
from data_models import (
    ExtractionRequest, ExtractionFromConfigRequest,
    ExtractionStatsResponse, ExtractionResponse
)
from mongodb import get_mongodb_service

router = APIRouter(prefix="/extract", tags=["SQL Extraction"])


@router.post("/database", response_model=ExtractionResponse)
async def extract_database(request: ExtractionRequest):
    """
    Extract schema and stored procedures from a single database.

    Connects directly to SQL Server, extracts all tables and stored procedures,
    and stores them in MongoDB for RAG-based text-to-SQL generation.

    This replaces the JavaScript schema-extractor-api.js functionality.
    """
    from sql_extraction import SchemaExtractor, DatabaseConfig

    try:
        # Convert request to DatabaseConfig
        db_config = DatabaseConfig(
            name=request.config.name,
            server=request.config.server,
            database=request.config.database,
            lookup_key=request.config.lookup_key,
            user=request.config.user,
            password=request.config.password,
            port=request.config.port
        )

        # Create extractor and run extraction
        extractor = SchemaExtractor(verbose=True)
        stats = await extractor.extract_single_database(db_config)

        return ExtractionResponse(
            success=True,
            databases=[ExtractionStatsResponse(
                database=stats.database,
                tables=stats.tables,
                procedures=stats.procedures,
                errors=stats.errors,
                duration_ms=stats.duration_ms,
                error=stats.error_message
            )],
            total_stats={
                'tables': stats.tables,
                'procedures': stats.procedures,
                'errors': stats.errors
            }
        )

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"SQL Server driver not installed: {str(e)}. Install pymssql or pyodbc."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-config", response_model=ExtractionResponse)
async def extract_from_config(request: ExtractionFromConfigRequest):
    """
    Extract schemas from multiple databases defined in a configuration file.

    Supports both JSON and XML configuration formats.
    Optionally extract only a specific database using the 'only' parameter.
    """
    from sql_extraction import SchemaExtractor

    try:
        extractor = SchemaExtractor(verbose=True)
        result = await extractor.extract_from_config(
            config_path=request.config_path,
            only=request.only
        )

        return ExtractionResponse(
            success=result['success'],
            databases=[
                ExtractionStatsResponse(**db)
                for db in result['databases']
            ],
            total_stats=result['totalStats']
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stored-procedures")
async def extract_stored_procedures_only(request: ExtractionRequest):
    """
    Extract only stored procedures from a database (skip table schemas).

    Useful for incremental updates when only SPs have changed.
    """
    from sql_extraction import DatabaseConfig
    from sql_extraction.stored_procedure_extractor import extract_stored_procedures

    try:
        # Try to import SQL driver
        try:
            import pymssql
            driver = 'pymssql'
        except ImportError:
            import pyodbc
            driver = 'pyodbc'

        db_config = DatabaseConfig(
            name=request.config.name,
            server=request.config.server,
            database=request.config.database,
            lookup_key=request.config.lookup_key,
            user=request.config.user,
            password=request.config.password,
            port=request.config.port
        )

        # Create connection
        if driver == 'pymssql':
            import pymssql
            connection = pymssql.connect(
                server=db_config.server,
                database=db_config.database,
                user=db_config.user,
                password=db_config.password,
                port=str(db_config.port)
            )
        else:
            import pyodbc
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={db_config.server},{db_config.port};"
                f"DATABASE={db_config.database};"
                f"UID={db_config.user};"
                f"PWD={db_config.password};"
            )
            connection = pyodbc.connect(conn_str)

        try:
            # Extract stored procedures
            procedures = await extract_stored_procedures(
                connection,
                db_config.database,
                verbose=True
            )

            # Store each procedure
            mongodb = get_mongodb_service()
            stored_count = 0

            for sp in procedures:
                await mongodb.store_stored_procedure(
                    database=db_config.lookup_key,
                    procedure_name=sp.name,
                    procedure_info={
                        'schema': sp.schema,
                        'parameters': sp.parameters,
                        'definition': sp.definition
                    }
                )
                stored_count += 1

            return {
                "success": True,
                "database": db_config.lookup_key,
                "procedures_extracted": len(procedures),
                "procedures_stored": stored_count
            }

        finally:
            connection.close()

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"SQL Server driver not installed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_extraction_status(job_id: str):
    """
    Get status of an extraction job (placeholder for async extraction).

    For now, extractions are synchronous. This endpoint is reserved for
    future async extraction support.
    """
    return {
        "job_id": job_id,
        "status": "not_implemented",
        "message": "Async extraction not yet implemented. Extractions run synchronously."
    }
