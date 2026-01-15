"""
App Factory Module

Provides the FastAPI application factory pattern for creating
configured application instances with all middleware and routes.
"""
from .factory import create_app

__all__ = ["create_app"]
