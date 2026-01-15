"""
Middleware configuration for the FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_cors(app: FastAPI, origins: list = None) -> None:
    """
    Configure CORS middleware for the application.

    Args:
        app: FastAPI application instance
        origins: List of allowed origins. Defaults to ["*"] for development.
    """
    if origins is None:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
