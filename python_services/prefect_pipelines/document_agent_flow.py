"""
Prefect Document Agent Pipeline

Orchestrates document processing through the EWR Document Agent:
1. Agent Health Check - Verify agent availability
2. Document Upload - Upload and process via agent API
3. Embedding Verification - Confirm embeddings generated
4. Search Validation - Validate semantic search works

Features:
- Document Agent API integration
- Health monitoring and alerting
- Batch processing support
- Progress tracking via Prefect UI
"""

import asyncio
import time
import os
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# Configuration
DOCUMENT_AGENT_BASE_URL = os.environ.get("DOCUMENT_AGENT_URL", "http://localhost:8001")


@dataclass
class AgentHealthResult:
    """Result from agent health check"""
    is_healthy: bool = False
    agent_available: bool = False
    vector_store_connected: bool = False
    llm_available: bool = False
    cached_documents: int = 0
    cached_chunks: int = 0
    error: str = ""


@dataclass
class DocumentUploadResult:
    """Result from document upload"""
    success: bool = False
    document_id: str = ""
    total_chunks: int = 0
    chunks_embedded: int = 0
    processing_time_ms: int = 0
    job_id: str = ""
    status: str = ""
    error: str = ""


@dataclass
class SearchValidationResult:
    """Result from search validation"""
    success: bool = False
    query: str = ""
    total_results: int = 0
    processing_time_ms: int = 0
    top_score: float = 0.0
    error: str = ""


@dataclass
class DocumentAgentStats:
    """Document agent statistics"""
    total_documents: int = 0
    total_chunks: int = 0
    vector_store: str = ""
    vector_store_connected: bool = False
    chunking_strategy: str = ""
    chunk_size: int = 0


@task(
    name="check_agent_health",
    description="Verify Document Agent availability and health",
    retries=3,
    retry_delay_seconds=10,
    tags=["document-agent", "health"]
)
async def check_agent_health_task() -> AgentHealthResult:
    """
    Check Document Agent health status.

    Verifies:
    - Agent is running and responding
    - Vector store (MongoDB) is connected
    - LLM service is available for embeddings

    Returns:
        AgentHealthResult with health status details
    """
    logger = get_run_logger()
    logger.info("Checking Document Agent health...")

    result = AgentHealthResult()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DOCUMENT_AGENT_BASE_URL}/document-agent/health")
            response.raise_for_status()

            data = response.json()
            result.is_healthy = data.get("status") == "healthy"
            result.agent_available = data.get("agent_available", False)
            result.vector_store_connected = data.get("vector_store_connected", False)
            result.llm_available = data.get("llm_available", False)

            details = data.get("details", {})
            result.cached_documents = details.get("cached_documents", 0)
            result.cached_chunks = details.get("cached_chunks", 0)

            logger.info(f"Agent healthy: {result.is_healthy}, Vector store: {result.vector_store_connected}, LLM: {result.llm_available}")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Health check failed: {e}")

    return result


@task(
    name="get_agent_stats",
    description="Get Document Agent statistics",
    retries=2,
    retry_delay_seconds=5,
    tags=["document-agent", "stats"]
)
async def get_agent_stats_task() -> DocumentAgentStats:
    """
    Get current Document Agent statistics.

    Returns:
        DocumentAgentStats with current statistics
    """
    logger = get_run_logger()
    logger.info("Getting Document Agent statistics...")

    result = DocumentAgentStats()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DOCUMENT_AGENT_BASE_URL}/document-agent/stats")
            response.raise_for_status()

            data = response.json()
            result.total_documents = data.get("cached_documents", 0)
            result.total_chunks = data.get("cached_chunks", 0)
            result.vector_store = data.get("vector_store", "")
            result.vector_store_connected = data.get("vector_store_connected", False)
            result.chunking_strategy = data.get("chunking_strategy", "")
            result.chunk_size = data.get("chunk_size", 0)

            logger.info(f"Stats: {result.total_documents} docs, {result.total_chunks} chunks")

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")

    return result


@task(
    name="upload_document",
    description="Upload and process document via Document Agent",
    retries=2,
    retry_delay_seconds=30,
    tags=["document-agent", "upload"]
)
async def upload_document_task(
    file_path: str,
    chunking_strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    department: str = "",
    doc_type: str = "",
    subject: str = ""
) -> DocumentUploadResult:
    """
    Upload document to Document Agent for processing.

    Args:
        file_path: Path to the document file
        chunking_strategy: Chunking strategy (recursive, sentence, paragraph, etc.)
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        department: Optional department classification
        doc_type: Optional document type
        subject: Optional subject classification

    Returns:
        DocumentUploadResult with upload status
    """
    logger = get_run_logger()
    logger.info(f"Uploading document: {file_path}")

    result = DocumentUploadResult()

    if not os.path.exists(file_path):
        result.error = f"File not found: {file_path}"
        logger.error(result.error)
        return result

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare multipart form data
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                data = {
                    "chunking_strategy": chunking_strategy,
                    "chunk_size": str(chunk_size),
                    "chunk_overlap": str(chunk_overlap),
                }

                if department:
                    data["department"] = department
                if doc_type:
                    data["doc_type"] = doc_type
                if subject:
                    data["subject"] = subject

                response = await client.post(
                    f"{DOCUMENT_AGENT_BASE_URL}/document-agent/upload",
                    files=files,
                    data=data
                )
                response.raise_for_status()

            upload_data = response.json()
            result.success = upload_data.get("success", False)
            result.document_id = upload_data.get("document_id", "")
            result.total_chunks = upload_data.get("total_chunks", 0)
            result.chunks_embedded = upload_data.get("chunks_embedded", 0)
            result.processing_time_ms = upload_data.get("processing_time_ms", 0)
            result.job_id = upload_data.get("job_id", "")
            result.status = upload_data.get("status", "")
            result.error = upload_data.get("error", "")

            # If background job, poll for completion
            if result.job_id and result.job_id != "sync" and result.status == "pending":
                result = await poll_job_completion(result.job_id, file_path)

            logger.info(f"Upload result: {result.total_chunks} chunks, ID: {result.document_id}")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Upload failed: {e}")

    return result


async def poll_job_completion(job_id: str, file_path: str, max_attempts: int = 60) -> DocumentUploadResult:
    """Poll for background job completion."""
    logger = get_run_logger()
    result = DocumentUploadResult(job_id=job_id)

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{DOCUMENT_AGENT_BASE_URL}/document-agent/jobs/{job_id}")
                response.raise_for_status()

                job_data = response.json()
                status = job_data.get("status", "")

                if status == "completed":
                    result.success = True
                    result.document_id = job_data.get("document_id", "")
                    result.total_chunks = job_data.get("total_chunks", 0)
                    result.chunks_embedded = job_data.get("chunks_embedded", 0)
                    result.status = "completed"
                    logger.info(f"Job {job_id} completed")
                    return result

                elif status == "failed":
                    result.error = job_data.get("error", "Unknown error")
                    result.status = "failed"
                    logger.error(f"Job {job_id} failed: {result.error}")
                    return result

                logger.info(f"Job {job_id} progress: {job_data.get('progress', 0)*100:.0f}%")

        except Exception as e:
            logger.warning(f"Polling error: {e}")

        await asyncio.sleep(2)

    result.error = "Job timed out"
    result.status = "timeout"
    return result


@task(
    name="validate_search",
    description="Validate semantic search works for uploaded document",
    retries=2,
    retry_delay_seconds=10,
    tags=["document-agent", "search", "validation"]
)
async def validate_search_task(
    document_id: str,
    query: str = "document content",
    top_k: int = 3,
    min_score: float = 0.0
) -> SearchValidationResult:
    """
    Validate that semantic search returns results for the document.

    Args:
        document_id: ID of uploaded document
        query: Search query to test
        top_k: Number of results to request
        min_score: Minimum similarity score threshold

    Returns:
        SearchValidationResult with search validation status
    """
    logger = get_run_logger()
    logger.info(f"Validating search for document: {document_id}")

    result = SearchValidationResult(query=query)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{DOCUMENT_AGENT_BASE_URL}/document-agent/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "min_score": min_score,
                    "filter_document_ids": [document_id]
                }
            )
            response.raise_for_status()

            search_data = response.json()
            result.success = search_data.get("success", False)
            result.total_results = search_data.get("total_results", 0)
            result.processing_time_ms = search_data.get("processing_time_ms", 0)

            results = search_data.get("results", [])
            if results:
                result.top_score = results[0].get("score", 0.0)

            logger.info(f"Search returned {result.total_results} results, top score: {result.top_score:.3f}")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Search validation failed: {e}")

    return result


@task(
    name="delete_document",
    description="Delete document from Document Agent",
    retries=2,
    retry_delay_seconds=10,
    tags=["document-agent", "delete"]
)
async def delete_document_task(document_id: str) -> bool:
    """
    Delete document from Document Agent.

    Args:
        document_id: ID of document to delete

    Returns:
        True if deletion successful
    """
    logger = get_run_logger()
    logger.info(f"Deleting document: {document_id}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{DOCUMENT_AGENT_BASE_URL}/document-agent/documents/{document_id}"
            )
            response.raise_for_status()

            data = response.json()
            success = data.get("success", False)
            logger.info(f"Document deletion: {'success' if success else 'failed'}")
            return success

    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        return False


@flow(
    name="document-agent-health-check",
    description="Document Agent Health Monitoring Flow",
    retries=0
)
async def health_check_flow() -> Dict[str, Any]:
    """
    Health check flow for Document Agent monitoring.

    Checks:
    - Agent availability
    - Vector store connection
    - LLM availability
    - Current statistics

    Returns:
        Dict with health status and statistics
    """
    logger = get_run_logger()
    logger.info("Starting Document Agent health check...")

    # Check health
    health = await check_agent_health_task()

    # Get stats
    stats = await get_agent_stats_task()

    # Create artifact
    status_emoji = "✅" if health.is_healthy else "❌"
    await create_markdown_artifact(
        key="health-check",
        markdown=f"""
# Document Agent Health Check {status_emoji}

## Status
- **Overall Health**: {"Healthy" if health.is_healthy else "Unhealthy"}
- **Agent Available**: {"Yes" if health.agent_available else "No"}
- **Vector Store**: {"Connected" if health.vector_store_connected else "Disconnected"}
- **LLM Service**: {"Available" if health.llm_available else "Unavailable"}

## Statistics
- **Cached Documents**: {stats.total_documents}
- **Cached Chunks**: {stats.total_chunks}
- **Vector Store**: {stats.vector_store}
- **Chunking Strategy**: {stats.chunking_strategy}
- **Chunk Size**: {stats.chunk_size}

## Timestamp
{datetime.utcnow().isoformat()}Z

{f"## Error\n{health.error}" if health.error else ""}
        """,
        description="Document Agent health check results"
    )

    return {
        "healthy": health.is_healthy,
        "health": {
            "agent_available": health.agent_available,
            "vector_store_connected": health.vector_store_connected,
            "llm_available": health.llm_available,
            "error": health.error
        },
        "stats": {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "vector_store": stats.vector_store,
            "chunking_strategy": stats.chunking_strategy,
            "chunk_size": stats.chunk_size
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@flow(
    name="document-agent-process",
    description="Process Document via Document Agent",
    retries=1,
    retry_delay_seconds=60
)
async def process_document_flow(
    file_path: str,
    chunking_strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    department: str = "",
    doc_type: str = "",
    subject: str = "",
    validate_search: bool = True,
    search_query: str = "document content"
) -> Dict[str, Any]:
    """
    Complete Document Processing Flow via Document Agent.

    This flow:
    1. Checks agent health
    2. Uploads and processes document
    3. Optionally validates search functionality

    Args:
        file_path: Path to document to process
        chunking_strategy: Chunking strategy to use
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        department: Optional department classification
        doc_type: Optional document type
        subject: Optional subject
        validate_search: Whether to validate search after upload
        search_query: Query to use for search validation

    Returns:
        Dict with complete processing results
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting document processing for: {file_path}")

    # Step 1: Health check
    health = await check_agent_health_task()

    if not health.is_healthy:
        logger.error("Document Agent is not healthy, aborting")
        return {
            "success": False,
            "error": f"Agent unhealthy: {health.error}",
            "file_path": file_path
        }

    # Step 2: Upload document
    upload_result = await upload_document_task(
        file_path=file_path,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        department=department,
        doc_type=doc_type,
        subject=subject
    )

    search_result = None
    if validate_search and upload_result.success and upload_result.document_id:
        # Step 3: Validate search
        search_result = await validate_search_task(
            document_id=upload_result.document_id,
            query=search_query
        )

    total_duration = time.time() - flow_start

    # Create summary artifact
    await create_markdown_artifact(
        key="processing-summary",
        markdown=f"""
# Document Processing Complete

## File Information
- **File**: {os.path.basename(file_path)}
- **Path**: {file_path}
- **Strategy**: {chunking_strategy}
- **Chunk Size**: {chunk_size}

## Upload Results
- **Success**: {"Yes" if upload_result.success else "No"}
- **Document ID**: {upload_result.document_id}
- **Total Chunks**: {upload_result.total_chunks}
- **Chunks Embedded**: {upload_result.chunks_embedded}
- **Processing Time**: {upload_result.processing_time_ms}ms
{f"- **Error**: {upload_result.error}" if upload_result.error else ""}

{f'''## Search Validation
- **Success**: {"Yes" if search_result.success else "No"}
- **Query**: {search_result.query}
- **Results Found**: {search_result.total_results}
- **Top Score**: {search_result.top_score:.3f}
- **Search Time**: {search_result.processing_time_ms}ms
''' if search_result else ""}

## Duration
- **Total**: {total_duration:.2f}s
- **Timestamp**: {datetime.utcnow().isoformat()}Z
        """,
        description=f"Processing summary for {os.path.basename(file_path)}"
    )

    return {
        "success": upload_result.success,
        "file_path": file_path,
        "document_id": upload_result.document_id,
        "total_duration_seconds": total_duration,
        "upload": {
            "success": upload_result.success,
            "total_chunks": upload_result.total_chunks,
            "chunks_embedded": upload_result.chunks_embedded,
            "processing_time_ms": upload_result.processing_time_ms,
            "error": upload_result.error
        },
        "search_validation": {
            "success": search_result.success if search_result else None,
            "total_results": search_result.total_results if search_result else None,
            "top_score": search_result.top_score if search_result else None
        } if search_result else None
    }


@flow(
    name="document-agent-batch-process",
    description="Batch Process Multiple Documents via Document Agent",
    retries=0
)
async def batch_process_flow(
    file_paths: List[str],
    chunking_strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    continue_on_error: bool = True
) -> Dict[str, Any]:
    """
    Batch process multiple documents via Document Agent.

    Args:
        file_paths: List of file paths to process
        chunking_strategy: Chunking strategy for all documents
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        continue_on_error: Whether to continue if a document fails

    Returns:
        Dict with batch processing results
    """
    logger = get_run_logger()
    batch_start = time.time()

    logger.info(f"Starting batch processing for {len(file_paths)} documents")

    # Check health first
    health = await check_agent_health_task()

    if not health.is_healthy:
        return {
            "success": False,
            "error": f"Agent unhealthy: {health.error}",
            "processed": 0,
            "failed": len(file_paths)
        }

    results = []
    successful = 0
    failed = 0

    for file_path in file_paths:
        try:
            upload_result = await upload_document_task(
                file_path=file_path,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            results.append({
                "file_path": file_path,
                "success": upload_result.success,
                "document_id": upload_result.document_id,
                "total_chunks": upload_result.total_chunks,
                "error": upload_result.error
            })

            if upload_result.success:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1
            results.append({
                "file_path": file_path,
                "success": False,
                "error": str(e)
            })

            if not continue_on_error:
                break

    total_duration = time.time() - batch_start

    # Create batch summary
    await create_markdown_artifact(
        key="batch-summary",
        markdown=f"""
# Batch Processing Complete

## Summary
- **Total Files**: {len(file_paths)}
- **Successful**: {successful}
- **Failed**: {failed}
- **Duration**: {total_duration:.2f}s

## Results
| File | Status | Document ID | Chunks |
|------|--------|-------------|--------|
{chr(10).join(f"| {os.path.basename(r['file_path'])} | {'✅' if r['success'] else '❌'} | {r.get('document_id', 'N/A')} | {r.get('total_chunks', 0)} |" for r in results)}

## Timestamp
{datetime.utcnow().isoformat()}Z
        """,
        description="Batch processing summary"
    )

    return {
        "success": failed == 0,
        "total_files": len(file_paths),
        "successful": successful,
        "failed": failed,
        "total_duration_seconds": total_duration,
        "results": results
    }


def run_health_check() -> Dict[str, Any]:
    """Run health check flow synchronously."""
    return asyncio.run(health_check_flow())


def run_process_document(file_path: str, **kwargs) -> Dict[str, Any]:
    """Run document processing flow synchronously."""
    return asyncio.run(process_document_flow(file_path=file_path, **kwargs))


def run_batch_process(file_paths: List[str], **kwargs) -> Dict[str, Any]:
    """Run batch processing flow synchronously."""
    return asyncio.run(batch_process_flow(file_paths=file_paths, **kwargs))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python document_agent_flow.py health")
        print("  python document_agent_flow.py process <file_path>")
        print("  python document_agent_flow.py batch <file1> <file2> ...")
        sys.exit(1)

    command = sys.argv[1]

    if command == "health":
        result = run_health_check()
        print(f"Health: {'OK' if result['healthy'] else 'FAILED'}")
        print(f"Stats: {result['stats']}")

    elif command == "process" and len(sys.argv) > 2:
        result = run_process_document(sys.argv[2])
        print(f"Result: {result}")

    elif command == "batch" and len(sys.argv) > 2:
        result = run_batch_process(sys.argv[2:])
        print(f"Batch Result: {result}")

    else:
        print(f"Unknown command: {command}")
