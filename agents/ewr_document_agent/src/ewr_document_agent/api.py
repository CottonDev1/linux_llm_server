"""
EWR Document Agent - FastAPI Service Layer
==========================================

Provides REST API endpoints for document processing, enabling both:
- Standalone operation (direct API calls)
- Pipeline integration (called by main RAG server)

Endpoints:
- POST /process - Process a document by file path
- POST /upload - Upload and process a document file
- POST /search - Semantic search across documents
- GET /jobs/{job_id} - Get job status
- GET /jobs/{job_id}/stream - SSE stream for job progress
- DELETE /documents/{document_id} - Delete a document
- GET /health - Health check
- GET /stats - Agent statistics
"""

import asyncio
import os
import tempfile
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import DocumentAgent
from .models import (
    ChunkingConfig,
    ChunkingStrategy,
    ProcessingResult,
    ProcessingStatus,
    VectorStoreConfig,
    SearchResult,
)

# ============================================================================
# Configuration
# ============================================================================

# Default MongoDB configuration (can be overridden via environment)
MONGODB_CONNECTION = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "EWRAI")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "document_chunks")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


# ============================================================================
# Request/Response Models
# ============================================================================

class ProcessRequest(BaseModel):
    """Request to process a document by file path."""
    file_path: str = Field(..., description="Path to the document file")
    chunking_strategy: str = Field("recursive", description="Chunking strategy")
    chunk_size: int = Field(500, description="Target chunk size in characters")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ProcessResponse(BaseModel):
    """Response from document processing."""
    job_id: str
    status: str
    document_id: Optional[str] = None
    total_chunks: int = 0
    chunks_embedded: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, description="Number of results to return")
    min_score: float = Field(0.0, description="Minimum similarity score (0-1)")
    filter_document_ids: Optional[List[str]] = Field(None, description="Filter to specific documents")


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time_ms: int


class JobStatus(BaseModel):
    """Job status tracking."""
    job_id: str
    status: str
    progress: float = 0.0
    stage: str = "pending"
    current_item: int = 0
    total_items: int = 0
    result: Optional[ProcessResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agent_running: bool
    vector_store_connected: bool
    llm_available: bool
    cached_documents: int
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    """Agent statistics response."""
    cached_documents: int
    cached_chunks: int
    vector_store: str
    vector_store_connected: bool
    chunking_strategy: str
    chunk_size: int
    total_documents_processed: int = 0


# ============================================================================
# Global State
# ============================================================================

# Agent instance (lazy initialized)
_agent: Optional[DocumentAgent] = None
_agent_lock = asyncio.Lock()

# Job tracking
_jobs: Dict[str, JobStatus] = {}
_jobs_lock = asyncio.Lock()

# Processing statistics
_stats = {
    "total_documents_processed": 0,
    "total_chunks_created": 0,
    "total_searches": 0,
}


async def get_agent() -> DocumentAgent:
    """Get or create the document agent singleton."""
    global _agent
    async with _agent_lock:
        if _agent is None:
            # Configure vector store
            vector_config = VectorStoreConfig(
                store_type="mongodb",
                connection_string=MONGODB_CONNECTION,
                database_name=MONGODB_DATABASE,
                collection_name=MONGODB_COLLECTION,
            )

            # Configure chunking
            chunking_config = ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=500,
                chunk_overlap=50,
            )

            # Create agent
            _agent = DocumentAgent(
                vector_config=vector_config,
                chunking_config=chunking_config,
            )

            # Configure LLM backend for embeddings
            from ewr_agent_core.llm_backends import OpenAIBackend
            _agent.llm = OpenAIBackend(
                base_url=LLM_BASE_URL,
                model=EMBEDDING_MODEL,
            )

            # Start agent
            await _agent.start()

        return _agent


async def create_job() -> JobStatus:
    """Create a new job for tracking."""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    job = JobStatus(
        job_id=job_id,
        status="pending",
        stage="queued",
        created_at=now,
        updated_at=now,
    )
    async with _jobs_lock:
        _jobs[job_id] = job
    return job


async def update_job(
    job_id: str,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    current_item: Optional[int] = None,
    total_items: Optional[int] = None,
    result: Optional[ProcessResponse] = None,
    error: Optional[str] = None,
):
    """Update job status."""
    async with _jobs_lock:
        if job_id in _jobs:
            job = _jobs[job_id]
            if status:
                job.status = status
            if stage:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            if current_item is not None:
                job.current_item = current_item
            if total_items is not None:
                job.total_items = total_items
            if result:
                job.result = result
            if error:
                job.error = error
            job.updated_at = datetime.utcnow()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="EWR Document Agent API",
    description="Document processing, chunking, embedding, and retrieval service",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    print("Document Agent API starting...")
    try:
        agent = await get_agent()
        stats = await agent.get_stats()
        print(f"Document Agent initialized: {stats}")
    except Exception as e:
        print(f"Warning: Agent initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _agent
    if _agent:
        await _agent.stop()
        _agent = None
    print("Document Agent API shut down")


# ============================================================================
# Health and Stats Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check agent health status."""
    try:
        agent = await get_agent()
        stats = await agent.get_stats()
        return HealthResponse(
            status="healthy",
            agent_running=True,
            vector_store_connected=stats.get("vector_store_connected", False),
            llm_available=agent.llm is not None,
            cached_documents=stats.get("cached_documents", 0),
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            agent_running=False,
            vector_store_connected=False,
            llm_available=False,
            cached_documents=0,
        )


@app.get("/stats", response_model=StatsResponse, tags=["Health"])
async def get_stats():
    """Get agent statistics."""
    agent = await get_agent()
    stats = await agent.get_stats()
    return StatsResponse(
        cached_documents=stats.get("cached_documents", 0),
        cached_chunks=stats.get("cached_chunks", 0),
        vector_store=stats.get("vector_store", "unknown"),
        vector_store_connected=stats.get("vector_store_connected", False),
        chunking_strategy=stats.get("chunking_strategy", "recursive"),
        chunk_size=stats.get("chunk_size", 500),
        total_documents_processed=_stats["total_documents_processed"],
    )


# ============================================================================
# Document Processing Endpoints
# ============================================================================

@app.post("/process", response_model=ProcessResponse, tags=["Documents"])
async def process_document(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Process a document by file path.

    For small files (<1MB), processes synchronously.
    For larger files, returns immediately with job_id for status tracking.
    """
    # Check file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    # Check file size
    file_size = os.path.getsize(request.file_path)

    # Create chunking config from request
    try:
        strategy = ChunkingStrategy(request.chunking_strategy)
    except ValueError:
        strategy = ChunkingStrategy.RECURSIVE

    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    # Small files: process synchronously
    if file_size < 1_000_000:  # 1MB
        start_time = datetime.utcnow()
        agent = await get_agent()
        result = await agent.process_document(
            file_path=request.file_path,
            chunking_config=config,
        )

        _stats["total_documents_processed"] += 1

        return ProcessResponse(
            job_id="sync",
            status=result.status.value,
            document_id=result.document_id,
            total_chunks=result.total_chunks,
            chunks_embedded=result.chunks_embedded,
            processing_time_ms=result.processing_time_ms,
            error=result.error,
        )

    # Large files: process in background
    job = await create_job()
    background_tasks.add_task(
        _process_document_background,
        job.job_id,
        request.file_path,
        config,
    )

    return ProcessResponse(
        job_id=job.job_id,
        status="pending",
    )


async def _process_document_background(
    job_id: str,
    file_path: str,
    config: ChunkingConfig,
):
    """Background task for document processing."""
    try:
        await update_job(job_id, status="processing", stage="parsing")

        agent = await get_agent()
        result = await agent.process_document(
            file_path=file_path,
            chunking_config=config,
        )

        _stats["total_documents_processed"] += 1

        await update_job(
            job_id,
            status=result.status.value,
            stage="completed",
            progress=1.0,
            result=ProcessResponse(
                job_id=job_id,
                status=result.status.value,
                document_id=result.document_id,
                total_chunks=result.total_chunks,
                chunks_embedded=result.chunks_embedded,
                processing_time_ms=result.processing_time_ms,
                error=result.error,
            ),
        )

    except Exception as e:
        await update_job(
            job_id,
            status="failed",
            stage="error",
            error=str(e),
        )


@app.post("/upload", response_model=ProcessResponse, tags=["Documents"])
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    chunking_strategy: str = Form("recursive", description="Chunking strategy"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(50, description="Chunk overlap"),
    department: Optional[str] = Form(None, description="Department category"),
    doc_type: Optional[str] = Form(None, description="Document type"),
    user_id: Optional[str] = Form(None, description="User ID for tracking"),
):
    """
    Upload a file, save it temporarily, and process it.

    Supports: PDF, DOCX, DOC, HTML, Markdown, TXT, JSON, XML, CSV
    """
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "")[1] or ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Create chunking config
        try:
            strategy = ChunkingStrategy(chunking_strategy)
        except ValueError:
            strategy = ChunkingStrategy.RECURSIVE

        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Small files: process synchronously
        if len(content) < 1_000_000:  # 1MB
            agent = await get_agent()
            result = await agent.process_document(
                file_path=tmp_path,
                chunking_config=config,
            )

            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            _stats["total_documents_processed"] += 1

            return ProcessResponse(
                job_id="sync",
                status=result.status.value,
                document_id=result.document_id,
                total_chunks=result.total_chunks,
                chunks_embedded=result.chunks_embedded,
                processing_time_ms=result.processing_time_ms,
                error=result.error,
            )

        # Large files: process in background
        job = await create_job()
        background_tasks.add_task(
            _process_uploaded_file_background,
            job.job_id,
            tmp_path,
            config,
        )

        return ProcessResponse(
            job_id=job.job_id,
            status="pending",
        )

    except Exception as e:
        # Cleanup temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


async def _process_uploaded_file_background(
    job_id: str,
    tmp_path: str,
    config: ChunkingConfig,
):
    """Background task for uploaded file processing."""
    try:
        await _process_document_background(job_id, tmp_path, config)
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================================
# Job Status Endpoints
# ============================================================================

@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    async with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _jobs[job_id]


@app.get("/jobs/{job_id}/stream", tags=["Jobs"])
async def stream_job_status(job_id: str):
    """Stream job status updates via Server-Sent Events."""

    async def generate():
        last_status = ""
        while True:
            async with _jobs_lock:
                if job_id not in _jobs:
                    yield f"data: {{\"error\": \"Job not found\"}}\n\n"
                    break

                job = _jobs[job_id]
                status_json = job.model_dump_json()

                # Only send if status changed
                if status_json != last_status:
                    last_status = status_json
                    yield f"data: {status_json}\n\n"

                # Stop if job is complete
                if job.status in ["completed", "failed"]:
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Search for relevant document chunks using semantic similarity.
    """
    start_time = datetime.utcnow()

    agent = await get_agent()
    results = await agent.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        filter_document_ids=request.filter_document_ids,
    )

    _stats["total_searches"] += 1

    # Convert results to response format
    result_dicts = []
    for r in results:
        result_dicts.append({
            "chunk_id": r.chunk.id,
            "document_id": r.chunk.document_id,
            "content": r.chunk.content,
            "score": r.score,
            "document_path": r.document_path,
            "chunk_index": r.chunk.chunk_index,
            "section": r.chunk.section,
        })

    end_time = datetime.utcnow()
    processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

    return SearchResponse(
        query=request.query,
        results=result_dicts,
        total_results=len(result_dicts),
        processing_time_ms=processing_time_ms,
    )


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    agent = await get_agent()
    success = await agent.delete_document(document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"success": True, "document_id": document_id}


@app.get("/documents/{document_id}", tags=["Documents"])
async def get_document(document_id: str):
    """Get a document by ID."""
    agent = await get_agent()
    document = await agent.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": document.id,
        "file_path": document.file_path,
        "document_type": document.document_type.value,
        "metadata": document.metadata.model_dump() if hasattr(document.metadata, 'model_dump') else document.metadata,
        "word_count": document.metadata.word_count if document.metadata else 0,
    }


@app.get("/documents/{document_id}/chunks", tags=["Documents"])
async def get_document_chunks(
    document_id: str,
    limit: int = Query(100, description="Maximum chunks to return"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """Get chunks for a document."""
    agent = await get_agent()
    chunks = await agent.get_chunks(document_id)

    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no chunks")

    # Paginate
    paginated = chunks[offset:offset + limit]

    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "returned_chunks": len(paginated),
        "offset": offset,
        "chunks": [
            {
                "id": c.id,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "section": c.section,
                "has_embedding": bool(c.embedding),
            }
            for c in paginated
        ],
    }


# ============================================================================
# Standalone Entry Point
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8002):
    """Run the API server standalone."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
