"""SQL Auth routes for SQL Server authentication and database listing."""
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

from log_service import log_pipeline, log_error

router = APIRouter(prefix="/api/sql", tags=["SQL Auth"])


class SqlAuthRequest(BaseModel):
    """Request model for SQL Server authentication"""
    server: str
    username: Optional[str] = None
    user: Optional[str] = None  # Alias for username (frontend sends 'user')
    password: Optional[str] = None
    database: Optional[str] = None


@router.post("/list-databases")
async def list_databases_sql_auth(request: Request, config: SqlAuthRequest):
    """
    List available databases using SQL Server authentication.
    """
    user_ip = request.client.host if request.client else "Unknown"
    log_pipeline("SQL_AUTH", user_ip, "Listing databases", config.server)

    try:
        import pymssql

        # Handle 'user' alias for 'username' (frontend sends 'user')
        username = config.username or config.user

        log_pipeline("SQL_AUTH", user_ip, f"Using SQL auth as {username}", config.server)
        connection = pymssql.connect(
            server=config.server,
            database='master',
            user=username,
            password=config.password,
            login_timeout=30
        )

        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sys.databases WHERE state_desc = 'ONLINE' ORDER BY name")
        databases = [row[0] for row in cursor.fetchall()]
        cursor.close()
        connection.close()

        return {
            "success": True,
            "databases": databases
        }
    except Exception as e:
        log_error("SQL_AUTH", str(e), {"server": config.server})
        return {
            "success": False,
            "error": str(e),
            "databases": []
        }
