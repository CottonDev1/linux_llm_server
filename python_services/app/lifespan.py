"""
Application Lifespan Management

Handles startup and shutdown events for the FastAPI application.
All service initialization and cleanup happens here.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import PORT
from core.process_manager import (
    write_pid_file,
    remove_pid_file,
    setup_worker_signal_handlers,
)
from mongodb import get_mongodb_service
from log_service import initialize_logging
from category_service import initialize_categories
from schema_validator import get_schema_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the application.

    Startup:
    - Initialize logging
    - Write worker PID file
    - Setup signal handlers
    - Initialize MongoDB connection
    - Initialize category service
    - Initialize SQLite user database
    - Initialize schema validator
    - Initialize LLM service
    - Initialize SQL generator service
    - Initialize Document Pipeline Orchestrator

    Shutdown:
    - Close LLM service
    - Close Agent service
    - Close Document Pipeline Orchestrator
    - Close MongoDB connection
    - Remove PID file
    """
    # Initialize logging first
    initialize_logging()

    # Write WORKER PID file (this runs in the worker process, not the parent reloader)
    # This is critical for proper process management with uvicorn reload mode
    write_pid_file(PORT, is_parent=False)

    # Setup signal handlers in the worker process
    setup_worker_signal_handlers(PORT)

    # Startup
    print("Starting Python Data Services...")
    mongodb = get_mongodb_service()
    await mongodb.initialize()

    # Initialize category service (SQLite)
    print("Initializing category service...")
    await initialize_categories(seed=True)
    print("Category service initialized")

    # Initialize SQLite user database
    print("Initializing SQLite user database...")
    try:
        from services.sqlite_service import get_sqlite_service
        sqlite_service = get_sqlite_service()
        print("SQLite user database initialized")
    except Exception as e:
        print(f"SQLite initialization warning: {e}")

    # Initialize schema validator with MongoDB
    print("Initializing schema validator...")
    validator = await get_schema_validator(mongodb)
    stats = validator.get_cache_stats()
    print(f"Schema validator initialized: {stats['tables']} tables, {stats['columns']} columns")

    # Initialize LLM service
    print("Initializing LLM service...")
    try:
        from services.llm_service import get_llm_service, close_llm_service
        llm_service = await get_llm_service()
        health = await llm_service.health_check()
        if health.get("healthy"):
            # Count total models across all endpoints
            endpoints = health.get("endpoints", {})
            total_models = sum(len(ep.get("models", [])) for ep in endpoints.values() if ep.get("healthy"))
            healthy_endpoints = [name for name, ep in endpoints.items() if ep.get("healthy")]
            print(f"LLM service initialized: {total_models} models across {len(healthy_endpoints)} endpoints ({', '.join(healthy_endpoints)})")
        else:
            # Show which endpoints failed
            endpoints = health.get("endpoints", {})
            failed = [f"{name}: {ep.get('error', 'unknown')}" for name, ep in endpoints.items() if not ep.get("healthy")]
            print(f"LLM service warning: Some endpoints unavailable - {'; '.join(failed)}")
    except Exception as e:
        print(f"LLM service initialization warning: {e}")

    # Initialize SQL Query Pipeline
    print("Initializing SQL Query Pipeline...")
    try:
        from sql_pipeline import get_query_pipeline
        sql_pipeline = await get_query_pipeline()
        print("SQL Query Pipeline initialized")
    except Exception as e:
        print(f"SQL Query Pipeline initialization warning: {e}")

    # Initialize Document Pipeline Orchestrator (CRAG-based RAG)
    print("Initializing Document Pipeline Orchestrator...")
    try:
        from orchestrator.document_orchestrator import get_orchestrator
        orchestrator = await get_orchestrator()
        print("Document Pipeline Orchestrator initialized (CRAG pattern ready)")
    except Exception as e:
        print(f"Document Pipeline Orchestrator initialization warning: {e}")

    print("Python Data Services started successfully")

    yield

    # Shutdown
    print("Shutting down Python Data Services...")

    # Close Embedding service
    try:
        from embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        await embedding_service.close()
        print("Embedding service closed")
    except Exception as e:
        print(f"Embedding service shutdown warning: {e}")

    # Close LLM service
    try:
        from services.llm_service import close_llm_service
        await close_llm_service()
        print("LLM service closed")
    except Exception as e:
        print(f"LLM service shutdown warning: {e}")

    # Close Agent service
    try:
        from agent import close_agent_service
        await close_agent_service()
        print("Agent service closed")
    except Exception as e:
        print(f"Agent service shutdown warning: {e}")

    # Close Document Pipeline Orchestrator
    try:
        from orchestrator.document_orchestrator import close_orchestrator
        await close_orchestrator()
        print("Document Pipeline Orchestrator closed")
    except Exception as e:
        print(f"Document Pipeline Orchestrator shutdown warning: {e}")

    await mongodb.close()

    # Remove WORKER PID file on clean shutdown
    remove_pid_file(PORT, is_parent=False)

    print("Python Data Services shut down")
