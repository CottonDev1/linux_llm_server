"""
Prefect Document Processing Pipeline

A thin wrapper around the actual DocumentProcessor and MongoDBService that provides
Prefect tracking, artifacts, and observability without duplicating any pipeline logic.

IMPORTANT: This flow uses the ACTUAL document processing services.
All document processing logic lives in:
- document_processor.py - Content extraction
- mongodb/helpers.py - Chunking
- mongodb/documents.py - Storage with embeddings

This ensures that testing the Prefect flow tests the real production code.
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class DocumentPipelineResult:
    """Result from the actual document pipeline."""
    success: bool = False
    file_path: str = ""
    filename: str = ""
    file_type: str = "unknown"
    file_size_bytes: int = 0
    document_id: str = ""
    content_length: int = 0
    chunks_created: int = 0
    pages: int = 0
    tables_found: int = 0
    error: Optional[str] = None
    processing_time_ms: float = 0.0


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_artifact_key(text: str) -> str:
    """Sanitize text for use in artifact keys."""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '-', text.lower())[:50]


# =============================================================================
# Main Pipeline Task - Calls the ACTUAL Services
# =============================================================================

@task(
    name="execute_document_pipeline",
    description="Execute the actual document processing pipeline (not a copy)",
    retries=2,
    retry_delay_seconds=30,
    tags=["document", "pipeline", "production"]
)
async def execute_document_pipeline_task(
    file_path: str,
    filename: Optional[str] = None,
    collection_name: str = "knowledge_base",
    department: str = "general",
    doc_type: str = "documentation",
    subject: Optional[str] = None,
    tags: Optional[list] = None,
) -> DocumentPipelineResult:
    """
    Execute the actual document pipeline - NOT a reimplementation.

    This task is a thin wrapper that calls:
    1. DocumentProcessor.process_file() for content extraction
    2. MongoDBService.store_document() for chunking, embedding, and storage

    All document processing logic lives in the actual services.
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Calling actual DocumentProcessor and MongoDBService")
    logger.info(f"File: {file_path}, Collection: {collection_name}")

    result = DocumentPipelineResult(
        file_path=file_path,
        filename=filename or os.path.basename(file_path)
    )

    try:
        # Import and use the ACTUAL services
        from document_processor import get_document_processor
        from mongodb import get_mongodb_service

        # Step 1: Get file info and validate
        if not os.path.exists(file_path):
            result.error = f"File not found: {file_path}"
            logger.error(result.error)
            return result

        result.file_size_bytes = os.path.getsize(file_path)

        # Step 2: Extract content using actual DocumentProcessor
        processor = get_document_processor()
        processed = await processor.process_file(file_path, result.filename)

        if not processed.success:
            result.error = processed.error or "Content extraction failed"
            logger.error(f"Extraction failed: {result.error}")
            return result

        result.file_type = processed.content_type
        result.content_length = len(processed.content)
        result.pages = processed.metadata.page_count
        result.tables_found = processed.metadata.table_count

        logger.info(f"Extracted {result.content_length} chars, {result.pages} pages, {result.tables_found} tables")

        # Step 3: Store with chunking and embedding using actual MongoDBService
        mongo = get_mongodb_service()
        await mongo.initialize()

        store_result = await mongo.store_document(
            title=result.filename,
            content=processed.content,
            department=department,
            doc_type=doc_type,
            subject=subject,
            file_name=result.filename,
            file_size=result.file_size_bytes,
            tags=tags,
            metadata={
                "pages": result.pages,
                "tables": result.tables_found,
                "author": processed.metadata.author,
                "created_date": processed.metadata.created_date,
                "file_hash": processed.metadata.file_hash,
            }
        )

        result.success = store_result.get("success", False)
        result.document_id = store_result.get("document_id", "")
        result.chunks_created = store_result.get("chunks_created", 0)

        if not result.success:
            result.error = store_result.get("message", "Storage failed")
            logger.error(f"Storage failed: {result.error}")
            return result

        logger.info(
            f"Pipeline completed: {result.chunks_created} chunks, "
            f"doc_id={result.document_id}"
        )

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"Pipeline error: {e}")

    result.processing_time_ms = (time.time() - start_time) * 1000

    return result


# =============================================================================
# Main Flow
# =============================================================================

@flow(
    name="document-processing-pipeline",
    description="Document Processing Pipeline - Wrapper around actual services for Prefect tracking",
    retries=1,
    retry_delay_seconds=60
)
async def document_flow(
    file_path: str,
    filename: Optional[str] = None,
    collection_name: str = "knowledge_base",
    department: str = "general",
    doc_type: str = "documentation",
    subject: Optional[str] = None,
    tags: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Document Processing Pipeline Flow - Tests the ACTUAL production pipeline.

    This flow is a thin wrapper that:
    1. Calls the real DocumentProcessor.process_file() method
    2. Calls the real MongoDBService.store_document() method
    3. Creates Prefect artifacts for observability
    4. Returns results in a consistent format

    All document processing logic lives in the actual services.
    This ensures that Prefect tests verify the actual production code.

    Args:
        file_path: Path to document file
        filename: Optional filename override
        collection_name: MongoDB collection for storage
        department: Document department/category
        doc_type: Document type (documentation, manual, etc.)
        subject: Document subject
        tags: List of tags for the document

    Returns:
        Dict with document ID, chunk count, and metadata
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting Document Processing Pipeline (using actual services)")
    logger.info(f"File: {file_path}")

    # Execute the actual pipeline
    result = await execute_document_pipeline_task(
        file_path=file_path,
        filename=filename,
        collection_name=collection_name,
        department=department,
        doc_type=doc_type,
        subject=subject,
        tags=tags,
    )

    total_time_ms = (time.time() - flow_start) * 1000

    # Create artifact for Prefect UI
    artifact_content = f"""
## Document Processing Pipeline Result

**File**: {result.filename}
**Path**: {result.file_path}
**Success**: {result.success}
**Processing Time**: {total_time_ms:.1f}ms

### Document Details
- **Type**: {result.file_type}
- **Size**: {result.file_size_bytes / 1024:.1f} KB
- **Content Length**: {result.content_length:,} chars
- **Pages**: {result.pages}
- **Tables Found**: {result.tables_found}

### Storage
- **Document ID**: {result.document_id or 'N/A'}
- **Chunks Created**: {result.chunks_created}
- **Collection**: {collection_name}

{f"### Error\\n{result.error}" if result.error else ""}
"""

    await create_markdown_artifact(
        key=f"doc-pipeline-{sanitize_artifact_key(result.filename)[:20]}-{int(time.time())}",
        markdown=artifact_content,
        description=f"Document processing for: {result.filename}"
    )

    # Return result dict
    return {
        "success": result.success,
        "file_path": result.file_path,
        "filename": result.filename,
        "file_type": result.file_type,
        "file_size_bytes": result.file_size_bytes,
        "document_id": result.document_id,
        "content_length": result.content_length,
        "chunks_created": result.chunks_created,
        "pages": result.pages,
        "tables_found": result.tables_found,
        "error": result.error,
        "collection": collection_name,
        "department": department,
        "doc_type": doc_type,
        "subject": subject,
        "tags": tags,
        "timing": {
            "total_ms": total_time_ms,
            "pipeline_ms": result.processing_time_ms,
        },
        "processing_time_ms": total_time_ms,
    }


# Backwards compatibility alias
def run_document_flow(
    file_path: str,
    collection_name: str = "knowledge_base"
) -> Dict[str, Any]:
    """Synchronous wrapper for backwards compatibility."""
    return asyncio.run(document_flow(
        file_path=file_path,
        collection_name=collection_name
    ))


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Document Processing Pipeline")
    parser.add_argument("--file", required=True, help="Path to document file")
    parser.add_argument("--collection", default="knowledge_base", help="MongoDB collection")
    parser.add_argument("--department", default="general", help="Document department")
    parser.add_argument("--type", default="documentation", help="Document type")
    parser.add_argument("--subject", help="Document subject")

    args = parser.parse_args()

    result = asyncio.run(document_flow(
        file_path=args.file,
        collection_name=args.collection,
        department=args.department,
        doc_type=args.type,
        subject=args.subject,
    ))

    print(f"\nResult: {result}")
