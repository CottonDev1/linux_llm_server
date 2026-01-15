"""
Application Factory

Creates and configures the FastAPI application instance with all
routes, middleware, and lifespan handlers.
"""
import os
from fastapi import FastAPI

from .middleware import configure_cors
from .lifespan import lifespan


def _setup_ffmpeg_path():
    """Add ffmpeg to PATH for audio processing (required for pydub)."""
    ffmpeg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "tools", "ffmpeg", "bin"
    )
    if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
        print(f"Added ffmpeg to PATH: {ffmpeg_path}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    # Setup ffmpeg PATH before importing audio modules
    _setup_ffmpeg_path()

    # Create app with lifespan handler
    app = FastAPI(
        title="RAG Python Data Services",
        description="MongoDB-based data layer for RAG server",
        version="1.0.0",
        lifespan=lifespan
    )

    # Configure CORS
    configure_cors(app)

    # Register all routers
    _register_routers(app)

    return app


def _register_routers(app: FastAPI):
    """Register all API routers with the application."""

    # Import API routers
    from api.llm_routes import router as llm_router
    from api.sql_routes_new import router as sql_new_router
    from api.document_agent_routes import router as document_agent_router
    from api.agent_learning_routes import router as agent_learning_router
    from api.sp_analysis_routes import router as sp_analysis_router
    from api.sql_query_routes import router as sql_query_router
    from api.document_pipeline_routes import router as document_pipeline_router

    # SQLite-based auth and user management routes
    from routes.auth_routes import router as auth_router, users_router
    from routes.settings_routes import settings_router, permissions_router
    from routes.tag_routes import router as tag_router
    from routes.llm_monitoring_routes import router as llm_monitoring_router
    from routes.health_routes import router as health_router

    # Migrated Pipeline routes (JS to Python)
    from code_flow_pipeline.routes import create_code_flow_routes
    from git_pipeline.routes import create_git_routes
    from code_assistance_pipeline.routes import create_code_routes
    from query_pipeline.routes import router as query_router

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

    # Health routes: / and /health
    app.include_router(health_router)
