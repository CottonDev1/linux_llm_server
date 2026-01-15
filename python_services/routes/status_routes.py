"""Status routes for health checks and service status."""
from fastapi import APIRouter

router = APIRouter(tags=["Status"])


@router.get("/status", tags=["Status"])
async def get_system_status():
    """
    Get complete system status (cached results from background checks).

    Health checks run in the background every 2 minutes.
    Returns status of: MongoDB, Vector Search, Embeddings, LLM backends.
    """
    from services.health_monitor import get_health_monitor
    health_monitor = get_health_monitor()

    python_health = health_monitor.get_python_health()
    llm_health = health_monitor.get_llm_health()

    return {
        "mongodb": {
            "connected": python_health.get("mongodb_connected", False),
            "version": python_health.get("mongodb_version"),
            "uri": python_health.get("mongodb_uri"),
        },
        "vector_search": python_health.get("vector_search", {}),
        "embeddings": {
            "loaded": python_health.get("embedding_model_loaded", False),
        },
        "llm": {
            "healthy": llm_health.get("healthy", False),
            "endpoints": llm_health.get("endpoints", {}),
            "error": llm_health.get("error"),
        },
        "collections": python_health.get("collections", {}),
        "status": python_health.get("status", "unknown"),
        "checked_at": python_health.get("checked_at"),
        "next_check_in": python_health.get("next_check_in"),
    }


@router.post("/status/refresh", tags=["Status"])
async def refresh_system_status():
    """
    Force an immediate status refresh.

    Use this when you suspect the cached status is stale.
    """
    from services.health_monitor import get_health_monitor
    health_monitor = get_health_monitor()
    await health_monitor.force_refresh("all")
    return await get_system_status()


@router.get("/", tags=["Status"])
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Python Data Services",
        "version": "1.0.0",
        "status": "running"
    }
