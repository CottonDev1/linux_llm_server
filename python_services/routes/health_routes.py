"""
Health check endpoints for the API.
"""
from fastapi import APIRouter
from data_models import HealthResponse
from mongodb import get_mongodb_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health"""
    mongodb = get_mongodb_service()
    return await mongodb.health_check()


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Python Data Services",
        "version": "1.0.0",
        "status": "running"
    }
