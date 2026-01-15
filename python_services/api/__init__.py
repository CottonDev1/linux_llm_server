"""
API Routes module for Python RAG Server.

This module provides FastAPI routers for:
- LLM operations (generate, generate-stream)
- SQL generation and execution
"""

from .llm_routes import router as llm_router
from .sql_routes_new import router as sql_router

__all__ = [
    "llm_router",
    "sql_router",
]
