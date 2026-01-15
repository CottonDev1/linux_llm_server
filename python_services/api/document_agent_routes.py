"""
Document Agent Integration Routes
=================================

FastAPI router for integrating ewr_document_agent into the main RAG server.
Provides endpoints for document processing using the agent framework.

These routes complement the existing /documents endpoints in main.py by
providing agent-based processing with advanced features:
- Multiple chunking strategies
- Background processing for large files
- Progress tracking via SSE
- Semantic search via embeddings

Usage in main.py:
    from api.document_agent_routes import router as document_agent_router
    app.include_router(document_agent_router)
"""

import os
import tempfile
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import MongoDB service for storing processed documents
from mongodb import get_mongodb_service
from log_service import log_pipeline, log_error
from config import MONGODB_URI, MONGODB_DATABASE, EMBEDDING_MODEL

# ============================================================================
# Configuration
# ============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")


# ============================================================================
# Request/Response Models
# ============================================================================

class AgentProcessRequest(BaseModel):
    """Request to process a document using the document agent."""
    file_path: str = Field(..., description="Path to the document file")
    chunking_strategy: str = Field("recursive", description="Chunking strategy: fixed_size, sentence, paragraph, recursive, markdown, code")
    chunk_size: int = Field(500, description="Target chunk size in characters")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    department: Optional[str] = Field(None, description="Department category (Tier 1)")
    doc_type: Optional[str] = Field(None, description="Document type (Tier 2)")
    subject: Optional[str] = Field(None, description="Subject/Product (Tier 3)")
    tags: Optional[List[str]] = Field(None, description="Tags for the document")


class AgentProcessResponse(BaseModel):
    """Response from document agent processing."""
    success: bool
    job_id: str
    status: str
    document_id: Optional[str] = None
    total_chunks: int = 0
    chunks_embedded: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None
    message: str = ""


class AgentSearchRequest(BaseModel):
    """Search request using document agent."""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(5, description="Number of results to return")
    min_score: float = Field(0.3, description="Minimum similarity score (0-1)")
    filter_document_ids: Optional[List[str]] = Field(None, description="Filter to specific documents")
    filter_departments: Optional[List[str]] = Field(None, description="Filter by departments")


class AgentSearchResponse(BaseModel):
    """Search response from document agent."""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time_ms: int


class AgentHealthResponse(BaseModel):
    """Health check response for document agent."""
    status: str
    agent_available: bool
    vector_store_connected: bool
    llm_available: bool
    details: Dict[str, Any] = {}


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float = 0.0
    stage: str = "pending"
    result: Optional[AgentProcessResponse] = None
    error: Optional[str] = None


# ============================================================================
# Global State
# ============================================================================

# Document agent instance (lazy initialized)
_document_agent = None
_agent_lock = asyncio.Lock()

# Job tracking
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


async def get_document_agent():
    """Get or create the document agent singleton."""
    global _document_agent

    async with _agent_lock:
        if _document_agent is None:
            try:
                from ewr_document_agent import DocumentAgent
                from ewr_document_agent.models import VectorStoreConfig, ChunkingConfig, ChunkingStrategy
                from ewr_agent_core.llm_backends import LlamaCppBackend

                # Configure vector store
                vector_config = VectorStoreConfig(
                    store_type="mongodb",
                    connection_string=MONGODB_URI,
                    database_name=MONGODB_DATABASE,
                    collection_name="document_agent_chunks",
                )

                # Configure chunking
                chunking_config = ChunkingConfig(
                    strategy=ChunkingStrategy.RECURSIVE,
                    chunk_size=500,
                    chunk_overlap=50,
                )

                # Create agent
                _document_agent = DocumentAgent(
                    vector_config=vector_config,
                    chunking_config=chunking_config,
                )

                # Configure LLM backend for embeddings
                _document_agent.set_llm(LlamaCppBackend(
                    base_url=LLM_BASE_URL,
                    model=EMBEDDING_MODEL,
                ))

                # Start agent
                await _document_agent.start()
                print("Document Agent initialized successfully")

            except ImportError as e:
                print(f"Document Agent not available: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Document Agent not installed. Run: pip install -e agents/ewr_document_agent[api]"
                )
            except Exception as e:
                print(f"Failed to initialize Document Agent: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Document Agent initialization failed: {str(e)}"
                )

        return _document_agent


async def create_job() -> str:
    """Create a new job for tracking."""
    job_id = str(uuid.uuid4())
    async with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "stage": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
    return job_id


async def update_job(job_id: str, **kwargs):
    """Update job status."""
    async with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/document-agent", tags=["Document Agent"])


# ============================================================================
# Health Endpoints
# ============================================================================

@router.get("/health", response_model=AgentHealthResponse)
async def health_check():
    """Check document agent health status."""
    try:
        agent = await get_document_agent()
        stats = await agent.get_stats()

        return AgentHealthResponse(
            status="healthy",
            agent_available=True,
            vector_store_connected=stats.get("vector_store_connected", False),
            llm_available=agent.llm is not None,
            details=stats,
        )
    except HTTPException:
        raise
    except Exception as e:
        return AgentHealthResponse(
            status="unhealthy",
            agent_available=False,
            vector_store_connected=False,
            llm_available=False,
            details={"error": str(e)},
        )


@router.get("/stats")
async def get_stats():
    """Get document agent statistics."""
    agent = await get_document_agent()
    stats = await agent.get_stats()
    return {"success": True, "stats": stats}


# ============================================================================
# Document Processing Endpoints
# ============================================================================

@router.post("/process", response_model=AgentProcessResponse)
async def process_document(
    request: AgentProcessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Process a document using the document agent.

    Supports multiple chunking strategies:
    - fixed_size: Split by character count
    - sentence: Split by sentences
    - paragraph: Split by paragraphs
    - recursive: Recursive text splitter (recommended)
    - markdown: Markdown-aware splitting
    - code: Code-aware splitting

    For small files (<1MB), processes synchronously.
    For larger files, returns job_id for status tracking.
    """
    user_ip = "api-call"

    log_pipeline("DOCUMENT_AGENT", user_ip, "Processing document via agent",
                 request.file_path,
                 details={"chunking_strategy": request.chunking_strategy})

    # Check file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    try:
        agent = await get_document_agent()
        from ewr_document_agent.models import ChunkingConfig, ChunkingStrategy

        # Create chunking config
        try:
            strategy = ChunkingStrategy(request.chunking_strategy)
        except ValueError:
            strategy = ChunkingStrategy.RECURSIVE

        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # Check file size
        file_size = os.path.getsize(request.file_path)

        # Small files: process synchronously
        if file_size < 1_000_000:  # 1MB
            result = await agent.process_document(
                file_path=request.file_path,
                chunking_config=config,
            )

            log_pipeline("DOCUMENT_AGENT", user_ip, "Document processed",
                         details={
                             "document_id": result.document_id,
                             "chunks": result.total_chunks,
                             "time_ms": result.processing_time_ms
                         })

            return AgentProcessResponse(
                success=result.status.value == "completed",
                job_id="sync",
                status=result.status.value,
                document_id=result.document_id,
                total_chunks=result.total_chunks,
                chunks_embedded=result.chunks_embedded,
                processing_time_ms=result.processing_time_ms,
                error=result.error,
                message=f"Processed {result.total_chunks} chunks" if not result.error else result.error,
            )

        # Large files: process in background
        job_id = await create_job()
        background_tasks.add_task(
            _process_document_background,
            job_id,
            request.file_path,
            config,
        )

        return AgentProcessResponse(
            success=True,
            job_id=job_id,
            status="pending",
            message="Document queued for processing",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("DOCUMENT_AGENT", user_ip, f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_document_background(job_id: str, file_path: str, config):
    """Background task for document processing."""
    try:
        await update_job(job_id, status="processing", stage="parsing")

        agent = await get_document_agent()
        result = await agent.process_document(
            file_path=file_path,
            chunking_config=config,
        )

        await update_job(
            job_id,
            status=result.status.value,
            stage="completed",
            progress=1.0,
            result={
                "success": result.status.value == "completed",
                "job_id": job_id,
                "status": result.status.value,
                "document_id": result.document_id,
                "total_chunks": result.total_chunks,
                "chunks_embedded": result.chunks_embedded,
                "processing_time_ms": result.processing_time_ms,
                "error": result.error,
            },
        )

    except Exception as e:
        await update_job(
            job_id,
            status="failed",
            stage="error",
            error=str(e),
        )


@router.post("/upload", response_model=AgentProcessResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    chunking_strategy: str = Form("recursive", description="Chunking strategy"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(50, description="Chunk overlap"),
    department: Optional[str] = Form(None, description="Department (Tier 1)"),
    doc_type: Optional[str] = Form(None, description="Document type (Tier 2)"),
    subject: Optional[str] = Form(None, description="Subject (Tier 3)"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
):
    """
    Upload and process a document using the document agent.

    Supported formats: PDF, DOCX, DOC, HTML, Markdown, TXT, JSON, XML, CSV
    """
    user_ip = "api-upload"

    log_pipeline("DOCUMENT_AGENT", user_ip, "File upload started",
                 file.filename or "unknown",
                 details={"chunking_strategy": chunking_strategy})

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "")[1] or ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        agent = await get_document_agent()
        from ewr_document_agent.models import ChunkingConfig, ChunkingStrategy

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
            result = await agent.process_document(
                file_path=tmp_path,
                chunking_config=config,
            )

            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            log_pipeline("DOCUMENT_AGENT", user_ip, "File upload complete",
                         file.filename or "unknown",
                         details={
                             "document_id": result.document_id,
                             "chunks": result.total_chunks,
                         })

            return AgentProcessResponse(
                success=result.status.value == "completed",
                job_id="sync",
                status=result.status.value,
                document_id=result.document_id,
                total_chunks=result.total_chunks,
                chunks_embedded=result.chunks_embedded,
                processing_time_ms=result.processing_time_ms,
                error=result.error,
                message=f"Uploaded {file.filename}: {result.total_chunks} chunks" if not result.error else result.error,
            )

        # Large files: process in background
        job_id = await create_job()
        background_tasks.add_task(
            _process_uploaded_file_background,
            job_id,
            tmp_path,
            config,
        )

        return AgentProcessResponse(
            success=True,
            job_id=job_id,
            status="pending",
            message=f"File {file.filename} queued for processing",
        )

    except HTTPException:
        # Cleanup on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        log_error("DOCUMENT_AGENT", user_ip, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_uploaded_file_background(job_id: str, tmp_path: str, config):
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

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    async with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = _jobs[job_id]
        return JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=job.get("progress", 0.0),
            stage=job.get("stage", "pending"),
            result=job.get("result"),
            error=job.get("error"),
        )


@router.get("/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Stream job status updates via Server-Sent Events."""

    async def generate():
        import json
        last_status = ""

        while True:
            async with _jobs_lock:
                if job_id not in _jobs:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                job = _jobs[job_id]
                status_json = json.dumps(job)

                # Only send if status changed
                if status_json != last_status:
                    last_status = status_json
                    yield f"data: {status_json}\n\n"

                # Stop if job is complete
                if job["status"] in ["completed", "failed"]:
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

@router.post("/search", response_model=AgentSearchResponse)
async def search_documents(request: AgentSearchRequest):
    """
    Search for relevant document chunks using semantic similarity.

    Uses the document agent's embedding-based search.
    """
    start_time = datetime.utcnow()

    try:
        agent = await get_document_agent()
        results = await agent.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            filter_document_ids=request.filter_document_ids,
        )

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

        return AgentSearchResponse(
            success=True,
            query=request.query,
            results=result_dicts,
            total_results=len(result_dicts),
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Document Management Endpoints
# ============================================================================

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks from the agent's vector store."""
    try:
        agent = await get_document_agent()
        success = await agent.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"success": True, "document_id": document_id, "message": "Document deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get a document by ID from the agent's cache."""
    try:
        agent = await get_document_agent()
        document = await agent.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found in agent cache")

        return {
            "success": True,
            "document": {
                "id": document.id,
                "file_path": document.file_path,
                "document_type": document.document_type.value,
                "word_count": document.metadata.word_count if document.metadata else 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    limit: int = Query(100, description="Maximum chunks to return"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """Get chunks for a document."""
    try:
        agent = await get_document_agent()
        chunks = await agent.get_chunks(document_id)

        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found or has no chunks")

        # Paginate
        paginated = chunks[offset:offset + limit]

        return {
            "success": True,
            "document_id": document_id,
            "total_chunks": len(chunks),
            "returned_chunks": len(paginated),
            "offset": offset,
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content[:500] + "..." if len(c.content) > 500 else c.content,
                    "chunk_index": c.chunk_index,
                    "section": c.section,
                    "has_embedding": bool(c.embedding),
                }
                for c in paginated
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
