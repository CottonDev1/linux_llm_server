"""
Python Data Services API
FastAPI server providing MongoDB-based data layer for the RAG system
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn
from core.log_utils import log_info, log_warning, log_error
from fastapi import FastAPI, HTTPException, Query, Body, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tempfile
import os
import json
import asyncio
from pathlib import Path

# Process management functions (extracted to separate module)
from process_manager import (
    write_pid_file,
    remove_pid_file,
    kill_existing_service,
    setup_worker_signal_handlers,
    setup_parent_signal_handlers,
)


# Add ffmpeg to PATH for audio processing (required for pydub)
# This must be done before any audio processing code is loaded
_ffmpeg_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "tools", "ffmpeg", "bin"
)
_ffmpeg_added = False
if os.path.exists(_ffmpeg_path) and _ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
    _ffmpeg_added = True  # Log during lifespan startup when log_utils is available

from config import (
    HOST, PORT,
    COLLECTION_SQL_EXAMPLES, COLLECTION_SQL_FAILED_QUERIES,
    COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES,
    COLLECTION_CODE_CLASSES, COLLECTION_CODE_METHODS,
    COLLECTION_CODE_CALLGRAPH, COLLECTION_DOCUMENTS
)
from data_models import (
    DocumentCreate, DocumentUpdate, DocumentResponse, DocumentSearchResult,
    StoreResponse, DeleteResponse, StatsResponse, HealthResponse,
    DatabaseConfigRequest, ExtractionRequest, ExtractionFromConfigRequest,
    ExtractionStatsResponse, ExtractionResponse,
    PipelineRunRequest, PipelineSummarizeRequest, PipelineEmbedRequest,
    FeedbackType, FeedbackCreate, FeedbackResponse, FeedbackStatsResponse, LowPerformingDocument,
    CategoryType, CategoryBase, CategoryCreate, CategoryUpdate, CategoryResponse, CategoryListResponse,
    AudioStoreRequest, AudioSearchRequest, EmotionResult, AudioEventResult, AudioMetadata,
    CallMetadata, CallContentAnalysis
)
from mongodb import get_mongodb_service
from database_name_parser import normalize_database_name
from log_service import log_pipeline, log_error, initialize_logging
from category_service import get_category_service, initialize_categories
from schema_validator import get_schema_validator, SchemaValidator

# Import API routers for LLM and SQL services
from api.llm_routes import router as llm_router
from api.sql_routes_new import router as sql_new_router
from api.document_agent_routes import router as document_agent_router
from api.agent_learning_routes import router as agent_learning_router
from api.sp_analysis_routes import router as sp_analysis_router
from api.sql_query_routes import router as sql_query_router
from api.document_pipeline_routes import router as document_pipeline_router
# TODO: Uncomment when agent module is implemented
# from api.agent_routes import router as agent_router

# SQLite-based auth and user management routes
from routes.auth_routes import router as auth_router, users_router
from routes.settings_routes import settings_router, permissions_router
from routes.tag_routes import router as tag_router
from routes.llm_monitoring_routes import router as llm_monitoring_router

# Migrated Pipeline routes (JS to Python)
from code_flow_pipeline.routes import create_code_flow_routes
from git_pipeline.routes import create_git_routes
from code_assistance_pipeline.routes import create_code_routes
from query_pipeline.routes import router as query_router

# New modular routers (from main.py refactoring)
from routes.status_routes import router as status_router
from routes.document_routes import router as document_router
from routes.code_context_routes import router as code_context_router
from routes.sql_knowledge_routes import router as sql_knowledge_router
from routes.sql_rag_routes import router as sql_rag_router
from routes.sql_validation_routes import router as sql_validation_router
from routes.sql_extraction_routes import router as sql_extraction_router
from routes.sql_summarization_routes import router as sql_summarization_router
from routes.pipeline_routes import router as pipeline_router
from routes.git_operations_routes import router as git_operations_router
from routes.admin_routes import router as admin_router
from routes.roslyn_routes import router as roslyn_router
from routes.feedback_routes import router as feedback_router
from routes.category_routes import router as category_router
from routes.audio_routes import router as audio_router
from routes.audio_metrics_routes import router as audio_metrics_router
from routes.bulk_audio_routes import router as bulk_audio_router
from routes.sql_auth_routes import router as sql_auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Initialize logging first
    initialize_logging()

    # Suppress uvicorn access logs for health endpoints
    # Health checks are now logged by the Health Monitor with descriptive messages
    import logging

    class HealthEndpointFilter(logging.Filter):
        """Filter out status endpoint access logs"""
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            # Filter out status endpoints - these are logged by Health Monitor
            if '"GET /status ' in message:
                return False
            return True

    # Apply filter to uvicorn access logger
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(HealthEndpointFilter())

    # Suppress INFO logs from third-party libraries (they use Python logging, not our log_utils)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.ERROR)  # Suppress file change warnings

    # Write WORKER PID file (this runs in the worker process, not the parent reloader)
    # This is critical for proper process management with uvicorn reload mode
    write_pid_file(PORT, is_parent=False)

    # Setup signal handlers in the worker process
    setup_worker_signal_handlers(PORT)

    # Startup
    log_info("Python Service", "Starting...")

    # Log ffmpeg status (deferred from module load)
    if _ffmpeg_added:
        log_info("Audio Pipeline", f"ffmpeg added to PATH: {_ffmpeg_path}")

    mongodb = get_mongodb_service()
    try:
        # Set a 10-second timeout for MongoDB initialization
        await asyncio.wait_for(mongodb.initialize(), timeout=10.0)
    except asyncio.TimeoutError:
        log_warning("MongoDB", "Initialization timed out after 10 seconds")
        log_warning("MongoDB", "Service will continue - some features may be unavailable")
    except Exception as mongo_error:
        log_warning("MongoDB", f"Initialization failed: {mongo_error}")
        log_warning("MongoDB", "Service will continue - some features may be unavailable")

    # Initialize category service (SQLite)
    await initialize_categories(seed=True)

    # Initialize SQLite user database
    try:
        from services.sqlite_service import get_sqlite_service
        sqlite_service = get_sqlite_service()
    except Exception as e:
        log_warning("SQLite Service", f"Initialization failed: {e}")

    # Initialize schema validator with MongoDB
    validator = await get_schema_validator(mongodb)
    stats = validator.get_cache_stats()
    log_info("Schema Validator", f"{stats['tables']} tables, {stats['columns']} columns cached")

    # Initialize LLM service
    llm_healthy = False  # Track for Audio Pipeline verification
    try:
        from services.llm_service import get_llm_service, close_llm_service
        llm_service = await get_llm_service()
        health = await llm_service.health_check()
        llm_healthy = health.get("healthy", False)
        if llm_healthy:
            # Count total models across all endpoints
            endpoints = health.get("endpoints", {})
            total_models = sum(len(ep.get("models", [])) for ep in endpoints.values() if ep.get("healthy"))
            healthy_endpoints = [name for name, ep in endpoints.items() if ep.get("healthy")]
            log_info("LLM Service", f"{total_models} models across {len(healthy_endpoints)} endpoints ({', '.join(healthy_endpoints)})")
        else:
            # Show which endpoints failed
            endpoints = health.get("endpoints", {})
            failed = [f"{name}: {ep.get('error', 'unknown')}" for name, ep in endpoints.items() if not ep.get("healthy")]
            log_warning("LLM Service", f"Some endpoints unavailable - {'; '.join(failed)}")
    except Exception as e:
        log_warning("LLM Service", f"Initialization failed: {e}")

    # Initialize SQL Query Pipeline
    try:
        from sql_pipeline import get_query_pipeline
        sql_pipeline = await get_query_pipeline()
    except Exception as e:
        log_warning("SQL Pipeline", f"Initialization failed: {e}")

    # Initialize Document Pipeline Orchestrator (CRAG-based RAG)
    try:
        from orchestrator.document_orchestrator import get_orchestrator
        orchestrator = await get_orchestrator()
    except Exception as e:
        log_warning("Document Pipeline", f"Orchestrator initialization failed: {e}")

    # Verify Audio Pipeline components
    audio_components = {"ffmpeg": False, "funasr": False, "llm": False}
    try:
        import shutil
        # Check ffmpeg
        audio_components["ffmpeg"] = shutil.which("ffmpeg") is not None
        # Check FunASR/SenseVoice model availability (used for transcription)
        try:
            from funasr import AutoModel
            audio_components["funasr"] = True
        except ImportError:
            pass
        # Check LLM service (already initialized above)
        audio_components["llm"] = llm_healthy

        missing = [k for k, v in audio_components.items() if not v]
        if missing:
            ready = [k for k, v in audio_components.items() if v]
            log_warning("Audio Pipeline", f"Missing: {', '.join(missing)} | Available: {', '.join(ready)}")
    except Exception as e:
        log_warning("Audio Pipeline", f"Component check failed: {e}")

    # Start Health Monitor (background health checks every 2 minutes)
    try:
        from services.health_monitor import get_health_monitor
        health_monitor = get_health_monitor()
        await health_monitor.start()
    except Exception as e:
        log_warning("Health Monitor", f"Failed to start: {e}")

    log_info("Python Service", "Started successfully")

    yield

    # Shutdown
    log_info("Python Service", "Shutting down...")

    # Stop Health Monitor
    try:
        from services.health_monitor import get_health_monitor
        health_monitor = get_health_monitor()
        await health_monitor.stop()
    except Exception as e:
        log_warning("Health Monitor", f"Shutdown: {e}")

    # Close Embedding service (has aiohttp session that needs proper cleanup)
    try:
        from embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        await embedding_service.close()
        log_info("Embedding Service", "Closed")
    except Exception as e:
        log_warning("Embedding Service", f"Shutdown: {e}")

    # Close LLM service
    try:
        from services.llm_service import close_llm_service
        await close_llm_service()
        log_info("LLM Service", "Closed")
    except Exception as e:
        log_warning("LLM Service", f"Shutdown: {e}")

    # Close Agent service
    try:
        from agent import close_agent_service
        await close_agent_service()
        log_info("Agent Service", "Closed")
    except Exception as e:
        log_warning("Agent Service", f"Shutdown: {e}")

    # Close Document Pipeline Orchestrator
    try:
        from orchestrator.document_orchestrator import close_orchestrator
        await close_orchestrator()
        log_info("Document Pipeline", "Orchestrator closed")
    except Exception as e:
        log_warning("Document Pipeline", f"Orchestrator shutdown: {e}")

    await mongodb.close()

    # Remove WORKER PID file on clean shutdown
    remove_pid_file(PORT, is_parent=False)

    log_info("Python Service", "Shut down complete")


app = FastAPI(
    title="RAG Python Data Services",
    description="""
    MongoDB-based data layer for RAG (Retrieval-Augmented Generation) server providing:

    - **Text-to-SQL Generation**: Natural language to SQL with LLM integration
    - **Document Processing**: Semantic search and RAG pipelines
    - **Audio Transcription**: Speech-to-text with emotion analysis
    - **Code Intelligence**: Method call graphs and semantic search
    - **Git Management**: Repository synchronization and analysis

    ## Architecture

    - **LLM Backend**: llama.cpp models (SQL, General, Code) on localhost:8080-8082
    - **Vector Database**: MongoDB with native vector search
    - **Embeddings**: Sentence transformers for semantic search
    - **Audio Processing**: FunASR/SenseVoice for transcription

    ## Documentation

    - **Interactive API Docs**: /docs (Swagger UI)
    - **Alternative Docs**: /redoc (ReDoc)
    - **OpenAPI Spec**: /openapi.json
    - **YAML Spec**: See `docs/api/python_service_openapi.yaml`
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "EWR Development Team",
        "email": "dev@ewr.com"
    },
    license_info={
        "name": "Proprietary"
    },
    openapi_tags=[
        {"name": "Status", "description": "Health checks and system status"},
        {"name": "Authentication", "description": "User authentication and session management"},
        {"name": "Users", "description": "User management and settings"},
        {"name": "LLM", "description": "Large Language Model text generation"},
        {"name": "SQL Query", "description": "Natural language to SQL conversion and execution"},
        {"name": "SQL", "description": "SQL generation and validation (legacy endpoints)"},
        {"name": "Document Agent", "description": "Document processing with agent framework"},
        {"name": "Documents", "description": "Document storage and search"},
        {"name": "Audio Analysis", "description": "Audio transcription and analysis"},
        {"name": "Code Flow", "description": "Code context and call graph analysis"},
        {"name": "Code Assistance", "description": "AI-powered code assistance"},
        {"name": "Query", "description": "General RAG query endpoints"},
        {"name": "Git", "description": "Git repository management"},
        {"name": "Admin", "description": "Administrative operations"},
        {"name": "Categories", "description": "Document categorization"},
        {"name": "Feedback", "description": "User feedback collection"},
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Register API Routers
# ==============================================================================

# LLM routes: /llm/generate, /llm/generate-stream, /llm/health, /llm/cache/*
app.include_router(llm_router)

# SQL generation routes: /sql/query-stream, /sql/generate, /sql/validate, /sql/validate-and-fix
app.include_router(sql_new_router)

# Document Agent routes: /document-agent/health, /document-agent/process, /document-agent/upload, /document-agent/search
app.include_router(document_agent_router)

# Agent Learning routes: /api/agent-learning/sql/check-cache, /api/agent-learning/sql/validate, /api/agent-learning/feedback
app.include_router(agent_learning_router)

# SP Analysis routes: /api/sp-analysis/health, /api/sp-analysis/analyze/{sp_id}, /api/sp-analysis/analyze-batch, /api/sp-analysis/training-data
app.include_router(sp_analysis_router)

# SQL Query Pipeline routes: /api/sql/query, /api/sql/query-stream, /api/sql/execute, /api/sql/test-connection, /api/sql/databases, /api/sql/feedback, /api/sql/save-example, /api/sql/settings/{user_id}, /api/sql/rules/{database}
app.include_router(sql_query_router)

# Agent routes: /agent/status, /agent/health, /agent/shell, /agent/powershell, /agent/git/sync, /agent/sql/chain
# TODO: Uncomment when agent module is implemented
# app.include_router(agent_router)

# SQLite-based authentication routes: /api/auth/login, /api/auth/logout, /api/auth/refresh, /api/auth/me
app.include_router(auth_router)

# User management routes: /api/users, /api/users/{id}, /api/users/{id}/settings
app.include_router(users_router)

# System settings routes: /api/settings, /api/settings/{section}
app.include_router(settings_router)

# Permission routes: /api/permissions/roles, /api/permissions/users/{id}
app.include_router(permissions_router)

# Tag routes: /api/tags, /api/tags/{id}, /api/tags/documents/{doc_id}
app.include_router(tag_router)

# Document Pipeline routes: /api/documents/search, /api/documents/query, /api/documents/query-stream, /api/documents/feedback, /api/documents/projects, /api/documents/health
# Uses KnowledgeBaseOrchestrator for CRAG-based RAG pipeline
app.include_router(document_pipeline_router)

# ==============================================================================
# Migrated Pipeline Routes (from JavaScript to Python)
# ==============================================================================

# Code Flow routes: /api/code-flow, /api/method-lookup, /api/call-chain
code_flow_router = create_code_flow_routes()
app.include_router(code_flow_router, prefix="/api", tags=["Code Flow"])

# Git Sync routes: /api/admin/git/pull, /api/admin/git/repositories, etc.
git_router = create_git_routes()
app.include_router(git_router, prefix="/api/admin/git", tags=["Git"])

# Code Assistance routes: /api/code/query, /api/code/query/stream, /api/code/feedback, /api/code/stats
code_assistance_router = create_code_routes()
app.include_router(code_assistance_router, prefix="/api/code", tags=["Code Assistance"])

# Query/RAG routes: /api/query, /api/query/search, /api/query/stream, /api/query/projects
app.include_router(query_router, prefix="/api", tags=["Query"])

# LLM Monitoring routes: /api/llm-monitoring/*
app.include_router(llm_monitoring_router)

# ==============================================================================
# Modular Routes (Refactored from main.py)
# ==============================================================================

# Status routes: /status, /status/refresh, /
app.include_router(status_router)

# Document routes: /documents/*
app.include_router(document_router)

# Code context routes: /code-context/*
app.include_router(code_context_router)

# SQL knowledge routes: /sql-knowledge/*
app.include_router(sql_knowledge_router)

# SQL RAG routes: /sql/*
app.include_router(sql_rag_router)

# SQL validation routes: /sql/validator/*
app.include_router(sql_validation_router)

# SQL extraction routes: /extract/*
app.include_router(sql_extraction_router)

# SQL summarization routes: /summarize/*
app.include_router(sql_summarization_router)

# Pipeline routes: /pipeline/*
app.include_router(pipeline_router)

# Git operations routes: /git/*
app.include_router(git_operations_router)

# Admin routes: /admin/*
app.include_router(admin_router)

# Roslyn routes: /roslyn/*
app.include_router(roslyn_router)

# Feedback routes: /feedback/*
app.include_router(feedback_router)

# Category routes: /categories/*
app.include_router(category_router)

# Audio routes: /audio/*
app.include_router(audio_router)

# Audio metrics routes: /audio/staff-metrics/*
app.include_router(audio_metrics_router)

# Bulk audio routes: /audio/bulk/*
app.include_router(bulk_audio_router)

# SQL auth routes: /sql/auth/*
app.include_router(sql_auth_router)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Setup PARENT signal handlers for graceful shutdown
    # This handles the parent/reloader process when reload=True
    setup_parent_signal_handlers(PORT)

    # Kill any existing service on the port before starting
    # This will kill both parent and worker processes from previous runs
    kill_existing_service(PORT)

    # Write PARENT PID file before spawning uvicorn
    # The worker PID is written in the lifespan function
    write_pid_file(PORT, is_parent=True)

    # Use reload only in development, disable in production
    # Note: When reload=True, uvicorn creates parent (reloader) + child (worker)
    # We track BOTH PIDs to ensure complete cleanup
    use_reload = os.environ.get("PYTHON_SERVICE_RELOAD", "true").lower() == "true"

    from core.log_utils import log_info as startup_log
    startup_log("Python Service", f"Starting uvicorn on {HOST}:{PORT} (reload={use_reload})")

    try:
        uvicorn.run(
            "main:app",
            host=HOST,
            port=PORT,
            reload=use_reload,
            log_level="warning"
        )
    finally:
        # Remove parent PID file on exit (worker PID removed in lifespan)
        remove_pid_file(PORT, is_parent=True)
