"""Admin routes for database management and system administration."""
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import sys
import psutil
import signal
import time
from datetime import datetime, timedelta

from mongodb import get_mongodb_service
from config import (
    COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES,
    COLLECTION_CODE_CLASSES, COLLECTION_CODE_METHODS, COLLECTION_DOCUMENTS,
    GIT_ROOT, PORT
)
from git_service import GitService
from core.log_utils import log_info, log_warning, log_error

router = APIRouter(tags=["Admin"])

# Initialize git service (singleton pattern for this module)
_git_service: Optional[GitService] = None


def get_git_service() -> GitService:
    """Get or create GitService singleton."""
    global _git_service
    if _git_service is None:
        _git_service = GitService(git_root=GIT_ROOT)
    return _git_service


# ============================================================================
# Request Models
# ============================================================================

class MongoDBTestRequest(BaseModel):
    """Request model for testing MongoDB connection"""
    uri: str = Field(..., description="MongoDB connection URI to test")
    database: Optional[str] = Field(None, description="Database name (optional)")


class MongoDBReconnectRequest(BaseModel):
    """Request model for reconnecting MongoDB with new URI"""
    uri: str = Field(..., description="New MongoDB connection URI")
    database: Optional[str] = Field(None, description="Database name (optional)")


# ============================================================================
# Admin Database Stats Endpoints
# ============================================================================

@router.get("/admin/db-stats", tags=["Admin"])
async def admin_get_db_stats():
    """
    Admin: Get MongoDB database statistics including size information.
    Returns database size, collection counts, and storage details.
    """
    mongodb = get_mongodb_service()
    return await mongodb.get_db_stats()


@router.get("/admin/dashboard-stats", tags=["Admin"])
async def admin_get_dashboard_stats():
    """
    Admin: Get dashboard statistics for SQL and C# analysis.
    Returns counts for tables, procedures, projects, and classes.
    """
    mongodb = get_mongodb_service()

    # Get SQL table count (unique tables from sql_schema_context)
    schema_collection = mongodb.db[COLLECTION_SQL_SCHEMA_CONTEXT]
    sql_tables_count = await schema_collection.count_documents({})

    # Get SQL procedure count
    proc_collection = mongodb.db[COLLECTION_SQL_STORED_PROCEDURES]
    sql_procedures_count = await proc_collection.count_documents({})

    # Get C# unique projects count
    classes_collection = mongodb.db[COLLECTION_CODE_CLASSES]
    project_pipeline = [
        {"$group": {"_id": "$project"}},
        {"$count": "uniqueProjectsCount"}
    ]
    project_result = await classes_collection.aggregate(project_pipeline).to_list(length=1)
    csharp_projects_count = project_result[0]["uniqueProjectsCount"] if project_result else 0

    # Get C# total classes count
    csharp_classes_count = await classes_collection.count_documents({})

    # Get SQL database count (unique databases from sql_schema_context)
    db_pipeline = [
        {"$group": {"_id": "$database"}},
        {"$count": "databaseCount"}
    ]
    db_result = await schema_collection.aggregate(db_pipeline).to_list(length=1)
    sql_databases_count = db_result[0]["databaseCount"] if db_result else 0

    return {
        "success": True,
        "sql_tables": sql_tables_count,
        "sql_procedures": sql_procedures_count,
        "sql_databases": sql_databases_count,
        "csharp_projects": csharp_projects_count,
        "csharp_classes": csharp_classes_count
    }


@router.get("/api/projects", tags=["Admin"])
async def get_projects():
    """
    Get all unique projects from MongoDB collections.
    Returns projects from code_classes, code_methods, and documents collections.
    """
    mongodb = get_mongodb_service()
    projects = []

    try:
        # Get unique projects from code_classes collection
        classes_collection = mongodb.db[COLLECTION_CODE_CLASSES]
        project_pipeline = [
            {"$group": {
                "_id": "$project",
                "classCount": {"$sum": 1}
            }},
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"_id": 1}}
        ]
        class_projects = await classes_collection.aggregate(project_pipeline).to_list(length=100)

        # Get method counts per project from code_methods
        methods_collection = mongodb.db[COLLECTION_CODE_METHODS]
        method_pipeline = [
            {"$group": {
                "_id": "$project",
                "methodCount": {"$sum": 1}
            }},
            {"$match": {"_id": {"$ne": None}}}
        ]
        method_counts = await methods_collection.aggregate(method_pipeline).to_list(length=100)
        method_count_map = {m["_id"]: m["methodCount"] for m in method_counts}

        # Combine project data
        for proj in class_projects:
            project_name = proj["_id"]
            projects.append({
                "name": project_name,
                "documentCount": proj.get("classCount", 0) + method_count_map.get(project_name, 0),
                "classCount": proj.get("classCount", 0),
                "methodCount": method_count_map.get(project_name, 0),
                "tableCount": 1
            })

        # Also check documents collection for additional projects
        docs_collection = mongodb.db[COLLECTION_DOCUMENTS]
        doc_pipeline = [
            {"$group": {
                "_id": "$department",
                "docCount": {"$sum": 1}
            }},
            {"$match": {"_id": {"$ne": None}}}
        ]
        doc_projects = await docs_collection.aggregate(doc_pipeline).to_list(length=100)

        existing_names = {p["name"] for p in projects}
        for doc_proj in doc_projects:
            if doc_proj["_id"] and doc_proj["_id"] not in existing_names:
                projects.append({
                    "name": doc_proj["_id"],
                    "documentCount": doc_proj.get("docCount", 0),
                    "classCount": 0,
                    "methodCount": 0,
                    "tableCount": 1
                })

        return {
            "success": True,
            "projects": projects,
            "totalProjects": len(projects)
        }

    except Exception as e:
        return {
            "success": False,
            "projects": [],
            "totalProjects": 0,
            "error": str(e)
        }


# ============================================================================
# Admin Git Endpoints
# ============================================================================

@router.get("/admin/git/repositories", tags=["Admin Git"])
async def admin_list_repositories():
    """
    Admin: List all repositories with detailed status.
    """
    git_svc = get_git_service()
    repos = git_svc.scan_repositories(force_refresh=True)

    detailed_repos = []
    for repo in repos:
        info = git_svc.get_repository_info_sync(repo.path)
        detailed_repos.append(info.model_dump())

    return {
        "success": True,
        "repositories": detailed_repos
    }


@router.post("/admin/git/pull-all", tags=["Admin Git"])
async def admin_pull_all_repositories():
    """
    Admin: Pull all configured repositories.
    """
    git_svc = get_git_service()
    repos = git_svc.scan_repositories()

    results = []
    for repo in repos:
        pull_result = await git_svc.pull_repository_async(repo.path)
        results.append({
            "repo": repo.name,
            "success": pull_result.success,
            "output": pull_result.output,
            "alreadyUpToDate": pull_result.is_already_up_to_date,
            "error": pull_result.error
        })

    success_count = sum(1 for r in results if r["success"])

    return {
        "success": True,
        "message": f"Pulled {success_count}/{len(results)} repositories",
        "results": results
    }


# ============================================================================
# Admin MongoDB Management Endpoints
# ============================================================================

@router.post("/admin/test-mongodb", tags=["Admin"])
async def admin_test_mongodb_connection(request: MongoDBTestRequest):
    """
    Admin: Test MongoDB connection with provided URI.
    Does not modify the running service - only tests connectivity.
    """
    from motor.motor_asyncio import AsyncIOMotorClient

    test_uri = request.uri
    test_database = request.database or "admin"

    try:
        # Create a temporary client to test connection
        test_client = AsyncIOMotorClient(test_uri, serverSelectionTimeoutMS=5000)

        # Test the connection with a ping
        await test_client.admin.command('ping')

        # Get server info
        server_info = await test_client.server_info()
        version = server_info.get('version', 'unknown')

        # List databases to verify access
        db_list = await test_client.list_database_names()

        # Close test connection
        test_client.close()

        return {
            "success": True,
            "message": f"Successfully connected to MongoDB {version}",
            "version": version,
            "databases": db_list,
            "uri": test_uri,
            "database": test_database
        }

    except Exception as e:
        error_msg = str(e)
        return {
            "success": False,
            "message": f"Connection failed: {error_msg}",
            "error": error_msg,
            "uri": test_uri
        }


@router.post("/admin/reconnect-mongodb", tags=["Admin"])
async def admin_reconnect_mongodb(request: MongoDBReconnectRequest):
    """
    Admin: Reconnect MongoDB service with new URI.
    This updates the running service's connection to use the new URI.
    WARNING: This affects all active operations using MongoDB.
    """
    import config

    new_uri = request.uri
    new_database = request.database

    try:
        # First test the new connection
        from motor.motor_asyncio import AsyncIOMotorClient
        test_client = AsyncIOMotorClient(new_uri, serverSelectionTimeoutMS=5000)
        await test_client.admin.command('ping')
        server_info = await test_client.server_info()
        version = server_info.get('version', 'unknown')
        test_client.close()

        # Get the singleton MongoDB service
        mongodb = get_mongodb_service()

        # Close existing connection
        if mongodb.client:
            mongodb.client.close()

        # Update config module values (in-memory)
        config.MONGODB_URI = new_uri
        if new_database:
            config.MONGODB_DATABASE = new_database

        # Reinitialize with new settings
        mongodb.client = None
        mongodb.db = None
        mongodb.is_initialized = False
        mongodb._vector_search_available = False

        # Re-initialize
        await mongodb.initialize()

        return {
            "success": True,
            "message": f"Successfully reconnected to MongoDB {version}",
            "version": version,
            "uri": new_uri,
            "database": config.MONGODB_DATABASE
        }

    except Exception as e:
        error_msg = str(e)
        return {
            "success": False,
            "message": f"Reconnection failed: {error_msg}",
            "error": error_msg,
            "uri": new_uri
        }


@router.post("/admin/mongodb-reconnect", tags=["Admin"])
async def admin_mongodb_reconnect():
    """
    Admin: Force reconnection to MongoDB.
    Useful when connection is lost or needs to be refreshed.
    """
    import config

    mongodb = get_mongodb_service()

    try:
        # Close existing connection
        if mongodb.client:
            mongodb.client.close()

        # Reset state
        mongodb.client = None
        mongodb.db = None
        mongodb.is_initialized = False
        mongodb._vector_search_available = False

        # Re-initialize connection
        await mongodb.initialize()

        return {
            "success": True,
            "connected": mongodb.is_initialized,
            "message": "MongoDB reconnected successfully" if mongodb.is_initialized else "Reconnection failed"
        }
    except Exception as e:
        return {
            "success": False,
            "connected": False,
            "error": str(e)
        }


# ============================================================================
# Python Service Management Endpoints
# ============================================================================

# Global variable to track service start time
_service_start_time = datetime.now()


def _perform_graceful_restart():
    """
    Internal function to perform graceful service restart.
    Uses os.execv to replace the current process with a new one.
    """
    try:
        log_info("Python Service", "Initiating graceful restart...")

        # Get the Python executable and script path
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")

        # Sleep briefly to allow response to be sent
        time.sleep(1)

        # Replace current process with new instance
        # os.execv replaces the current process without creating a new PID
        os.execv(python_exe, [python_exe, script_path])

    except Exception as e:
        log_error("Python Service", f"Restart failed: {e}")
        # If execv fails, try SIGTERM as fallback
        os.kill(os.getpid(), signal.SIGTERM)


@router.post("/admin/service/status", tags=["Service Management"])
async def get_service_status() -> Dict[str, Any]:
    """
    Get comprehensive Python service health and status information.

    Returns:
        - healthy: Overall health status
        - uptime: Service uptime in seconds
        - memory_usage: Memory usage in MB
        - cpu_percent: CPU usage percentage
        - port: Service port
        - pid: Process ID
        - python_version: Python version string
        - threads: Number of active threads
        - connections: Number of active network connections
    """
    try:
        current_process = psutil.Process(os.getpid())
        uptime_seconds = (datetime.now() - _service_start_time).total_seconds()

        # Get memory info
        memory_info = current_process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Get CPU percent (non-blocking)
        cpu_percent = current_process.cpu_percent(interval=0.1)

        # Get thread count
        thread_count = current_process.num_threads()

        # Get connection count (on the service port)
        connections = current_process.connections()
        active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])

        # Check if MongoDB is connected
        mongodb = get_mongodb_service()
        mongodb_connected = mongodb.is_initialized

        # Get LLM service status
        llm_healthy = False
        llm_endpoints = {}
        try:
            from services.llm_service import get_llm_service
            llm_service = await get_llm_service()
            health = await llm_service.health_check()
            llm_healthy = health.get("healthy", False)
            llm_endpoints = health.get("endpoints", {})
        except Exception:
            pass

        return {
            "success": True,
            "healthy": True,
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": str(timedelta(seconds=int(uptime_seconds))),
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
            "port": PORT,
            "pid": os.getpid(),
            "python_version": sys.version,
            "threads": thread_count,
            "connections": active_connections,
            "mongodb_connected": mongodb_connected,
            "llm_healthy": llm_healthy,
            "llm_endpoints": llm_endpoints
        }

    except Exception as e:
        log_error("Service Status", f"Failed to get status: {e}")
        return {
            "success": False,
            "healthy": False,
            "error": str(e),
            "port": PORT,
            "pid": os.getpid()
        }


@router.post("/admin/service/restart", tags=["Service Management"])
async def restart_service(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Gracefully restart the Python service.

    This endpoint returns immediately and schedules the restart in the background.
    The service will:
    1. Send a success response
    2. Wait 1 second for response to be sent
    3. Replace itself with a new process (os.execv)

    Note: The process PID will remain the same when using os.execv.

    Returns:
        - success: Always true if endpoint is reached
        - message: Restart initiated message
        - pid: Current process ID
        - estimated_restart_time: Time when service should be back (seconds)
    """
    try:
        current_pid = os.getpid()
        log_info("Python Service", f"Restart requested for PID {current_pid}")

        # Schedule restart in background after response is sent
        background_tasks.add_task(_perform_graceful_restart)

        return {
            "success": True,
            "message": "Python service restart initiated",
            "pid": current_pid,
            "estimated_restart_time_seconds": 5
        }

    except Exception as e:
        log_error("Service Restart", f"Failed to initiate restart: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initiate restart"
        }


@router.get("/admin/service/info", tags=["Service Management"])
async def get_service_info() -> Dict[str, Any]:
    """
    Get detailed Python service information including loaded models and configuration.

    Returns:
        - service_version: Service version
        - python_version: Python version
        - platform: OS platform
        - cwd: Current working directory
        - port: Service port
        - pid: Process ID
        - environment: Environment variables (filtered)
        - loaded_models: Information about loaded AI models
        - mongodb_status: MongoDB connection status
        - collections: Available MongoDB collections
    """
    try:
        # Get LLM model information
        loaded_models = {}
        try:
            from services.llm_service import get_llm_service
            llm_service = await get_llm_service()
            health = await llm_service.health_check()

            if health.get("healthy"):
                endpoints = health.get("endpoints", {})
                for endpoint_name, endpoint_info in endpoints.items():
                    if endpoint_info.get("healthy"):
                        models = endpoint_info.get("models", [])
                        loaded_models[endpoint_name] = {
                            "count": len(models),
                            "models": models,
                            "url": endpoint_info.get("url")
                        }
        except Exception as e:
            loaded_models = {"error": str(e)}

        # Get MongoDB info
        mongodb = get_mongodb_service()
        mongodb_status = {
            "connected": mongodb.is_initialized,
            "database": mongodb.db.name if mongodb.is_initialized and mongodb.db else None
        }

        # Get collection names if connected
        collections = []
        if mongodb.is_initialized and mongodb.db:
            try:
                collections = await mongodb.db.list_collection_names()
            except Exception:
                pass

        # Get safe environment variables (filter out sensitive data)
        safe_env_vars = {
            k: v for k, v in os.environ.items()
            if not any(sensitive in k.upper() for sensitive in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN'])
        }

        return {
            "success": True,
            "service_version": "1.0.0",
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "port": PORT,
            "pid": os.getpid(),
            "environment": safe_env_vars,
            "loaded_models": loaded_models,
            "mongodb_status": mongodb_status,
            "collections": collections
        }

    except Exception as e:
        log_error("Service Info", f"Failed to get info: {e}")
        return {
            "success": False,
            "error": str(e)
        }
