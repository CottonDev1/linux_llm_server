"""
Routes package for EWR AI Python API.

Exports all route modules for easy import.
"""

from .auth_routes import router as auth_router, users_router
from .git_routes import router as git_router
from .settings_routes import settings_router, permissions_router
from .tag_routes import router as tag_router
from .health_routes import router as health_router
from .llm_monitoring_routes import router as llm_monitoring_router
from .sql_extraction_routes import router as sql_extraction_router

__all__ = [
    "auth_router",
    "users_router",
    "git_router",
    "settings_router",
    "permissions_router",
    "tag_router",
    "health_router",
    "llm_monitoring_router",
    "sql_extraction_router",
]
