"""SQL Validation routes for schema validation and table info."""
from fastapi import APIRouter, Request, HTTPException, Body
from schema_validator import get_schema_validator
from log_service import log_pipeline, log_error

router = APIRouter(prefix="/sql/validator", tags=["SQL Validation"])


# Note: /sql/validate and /sql/validate-and-fix endpoints are kept in main.py
# because they don't have the /sql/validator prefix - they use /sql prefix directly


@router.get("/stats")
async def get_validator_stats(request: Request):
    """
    Get statistics about the schema validator cache.

    Returns information about loaded databases, tables, and columns.
    """
    try:
        validator = await get_schema_validator()
        stats = validator.get_cache_stats()

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_validator_cache(request: Request):
    """
    Refresh the schema validator cache from MongoDB.

    Call this after adding new schema contexts to update the validation cache.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        validator = await get_schema_validator()
        await validator.refresh_cache()
        stats = validator.get_cache_stats()

        log_pipeline("SQLValidation", user_ip, "Schema cache refreshed",
                    f"{stats['tables']} tables loaded",
                    details=stats)

        return {
            "success": True,
            "message": "Schema cache refreshed",
            **stats
        }

    except Exception as e:
        log_error("SQLValidation", user_ip, "Cache refresh failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/columns/{database}/{table_name}")
async def get_valid_columns(
    database: str,
    table_name: str
):
    """
    Get all valid column names for a specific table.

    Useful for debugging and building column dropdowns.
    """
    try:
        validator = await get_schema_validator()
        columns = validator.get_valid_columns(database, table_name)

        return {
            "success": True,
            "database": database,
            "table": table_name,
            "columns": list(columns),
            "count": len(columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{database}")
async def get_valid_tables(database: str):
    """
    Get all valid table names for a database.

    Useful for debugging and building table dropdowns.
    """
    try:
        validator = await get_schema_validator()
        tables = validator.get_valid_tables(database)

        return {
            "success": True,
            "database": database,
            "tables": list(tables),
            "count": len(tables)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
