"""
Embedding API Wrapper for llama.cpp

This FastAPI service wraps llama.cpp's embedding endpoint to match
the API expected by the Python embedding service.

API Translation:
- POST /embed {"text": "..."} -> llama.cpp /embedding {"content": "..."}
- POST /embed/batch {"texts": [...]} -> Multiple llama.cpp calls
- GET /health -> Health check with model info

Run:
    uvicorn embedding_wrapper:app --host 0.0.0.0 --port 8084
"""

import asyncio
import logging
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration
LLAMACPP_URL = "http://localhost:8083"  # llama.cpp embedding server
TIMEOUT = 30.0
MODEL_NAME = "nomic-embed-text-v1.5"
DIMENSIONS = 768

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Embedding API Wrapper",
    description="Wrapper for llama.cpp embedding service",
    version="1.0.0"
)


class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embedding: List[float]


class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]


class HealthResponse(BaseModel):
    status: str
    model: str
    dimensions: int


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLAMACPP_URL}/health")
            if response.status_code == 200:
                return HealthResponse(
                    status="ok",
                    model=MODEL_NAME,
                    dimensions=DIMENSIONS
                )
            else:
                raise HTTPException(status_code=503, detail="llama.cpp server unhealthy")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"llama.cpp server unavailable: {e}")


@app.post("/embed", response_model=EmbedResponse)
async def embed_single(request: EmbedRequest):
    """Generate embedding for a single text."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # llama.cpp expects {"content": "..."} format
            response = await client.post(
                f"{LLAMACPP_URL}/embedding",
                json={"content": request.text}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"llama.cpp error: {response.text}"
                )

            data = response.json()
            embedding = data.get("embedding", [])

            if not embedding:
                raise HTTPException(status_code=500, detail="No embedding returned")

            return EmbedResponse(embedding=embedding)

    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=503, detail=f"llama.cpp unavailable: {e}")


@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    """Generate embeddings for multiple texts."""
    if not request.texts:
        return EmbedBatchResponse(embeddings=[])

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Process texts concurrently (up to 10 at a time)
            semaphore = asyncio.Semaphore(10)

            async def embed_one(text: str) -> List[float]:
                async with semaphore:
                    response = await client.post(
                        f"{LLAMACPP_URL}/embedding",
                        json={"content": text}
                    )
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"llama.cpp error: {response.text}"
                        )
                    data = response.json()
                    return data.get("embedding", [])

            embeddings = await asyncio.gather(*[embed_one(text) for text in request.texts])
            return EmbedBatchResponse(embeddings=list(embeddings))

    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=503, detail=f"llama.cpp unavailable: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
