"""
Prefect Document Processing Pipeline

Orchestrates document ingestion workflow:
1. Document Type Detection - Identify file format
2. Content Extraction - Extract text and metadata
3. Chunking - Split into searchable segments
4. Embedding - Generate vectors for similarity search

Features:
- Automatic file type detection
- Smart chunking with overlap
- MongoDB vector storage
- Built-in retries for resilience
"""

import asyncio
import time
import os
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class DocumentTypeResult:
    """Result from document type detection"""
    file_path: str
    file_type: str = "unknown"
    file_size_bytes: int = 0
    mime_type: str = "application/octet-stream"
    is_supported: bool = False


@dataclass
class ContentExtractionResult:
    """Result from content extraction"""
    file_path: str
    file_type: str = ""
    content: str = ""
    content_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables_found: int = 0
    pages: int = 0
    extraction_method: str = "none"
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


@dataclass
class ChunkingResult:
    """Result from document chunking"""
    file_path: str
    total_chunks: int = 0
    avg_chunk_size: int = 0
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


@dataclass
class DocumentEmbeddingResult:
    """Result from document embedding"""
    file_path: str
    chunks_embedded: int = 0
    vectors_generated: int = 0
    stored_in_mongodb: bool = False
    document_id: str = ""
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True
    embedding_model: str = ""
    embedding_dim: int = 0
    use_local_embeddings: bool = False


@task(
    name="detect_document_type",
    description="Detect document type and validate support",
    retries=1,
    retry_delay_seconds=5,
    tags=["document", "detection"]
)
async def detect_type_task(file_path: str) -> DocumentTypeResult:
    """
    Detect document type and validate support.

    Supported formats: PDF, DOCX, XLSX, TXT, MD, CSV, JSON

    Args:
        file_path: Path to the document file

    Returns:
        DocumentTypeResult with file type information
    """
    import mimetypes

    logger = get_run_logger()
    logger.info(f"Detecting document type for: {file_path}")

    path = Path(file_path)
    result = DocumentTypeResult(file_path=file_path)

    # Get file info
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return result

    result.file_size_bytes = os.path.getsize(file_path)
    extension = path.suffix.lower()

    # Determine mime type
    mime_type, _ = mimetypes.guess_type(file_path)
    result.mime_type = mime_type or "application/octet-stream"

    # Map extensions to types
    type_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        '.xlsx': 'xlsx',
        '.xls': 'xls',
        '.txt': 'text',
        '.md': 'markdown',
        '.csv': 'csv',
        '.json': 'json'
    }

    result.file_type = type_map.get(extension, 'unknown')
    result.is_supported = result.file_type != 'unknown'

    logger.info(f"Detected: {result.file_type}, {result.file_size_bytes/1024:.1f} KB, supported={result.is_supported}")

    return result


@task(
    name="extract_content",
    description="Extract content and metadata from document",
    retries=2,
    retry_delay_seconds=15,
    tags=["document", "extraction"]
)
async def extract_content_task(type_result: DocumentTypeResult) -> ContentExtractionResult:
    """
    Extract content and metadata from document.

    Uses appropriate extractor based on file type:
    - PDF: PyMuPDF with table extraction
    - DOCX: python-docx
    - XLSX: openpyxl
    - Text/MD/CSV/JSON: Direct text reading

    Args:
        type_result: Result from type detection task

    Returns:
        ContentExtractionResult with extracted content
    """
    logger = get_run_logger()
    start_time = time.time()

    file_path = type_result.file_path
    file_type = type_result.file_type

    logger.info(f"Extracting content from {file_type} file: {file_path}")

    result = ContentExtractionResult(file_path=file_path, file_type=file_type)

    if not type_result.is_supported:
        result.errors.append(f"Unsupported file type: {file_type}")
        result.success = False
        result.duration_seconds = time.time() - start_time
        return result

    try:
        import sys
        sys.path.insert(0, '..')
        from document_processor import DocumentProcessor

        processor = DocumentProcessor()
        extraction_data = await processor.process_file(file_path)

        result.content = extraction_data.get('content', '')
        result.content_length = len(result.content)
        result.metadata = extraction_data.get('metadata', {})
        result.tables_found = len(extraction_data.get('tables', []))
        result.pages = result.metadata.get('page_count', 1)
        result.extraction_method = f"{file_type}_processor"

        logger.info(f"Extracted {result.content_length} characters, {result.tables_found} tables, {result.pages} pages")

    except Exception as e:
        error_msg = f"Content extraction failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time

    # Create Prefect artifact
    await create_markdown_artifact(
        key="extraction-result",
        markdown=f"""
## Content Extraction Results
- **File**: {os.path.basename(file_path)}
- **Type**: {file_type}
- **Content Length**: {result.content_length:,} chars
- **Pages**: {result.pages}
- **Tables Found**: {result.tables_found}
- **Duration**: {result.duration_seconds:.2f}s
- **Status**: {"Success" if result.success else "Failed"}
        """,
        description=f"Extraction for {os.path.basename(file_path)}"
    )

    return result


@task(
    name="chunk_document",
    description="Split document content into overlapping chunks",
    retries=1,
    retry_delay_seconds=5,
    tags=["document", "chunking"]
)
async def chunk_document_task(
    extraction_result: ContentExtractionResult,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> ChunkingResult:
    """
    Split document content into overlapping chunks for embedding.

    Args:
        extraction_result: Result from content extraction
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        ChunkingResult with chunk information
    """
    logger = get_run_logger()
    start_time = time.time()

    file_path = extraction_result.file_path
    content = extraction_result.content

    logger.info(f"Chunking document: {file_path}")

    result = ChunkingResult(file_path=file_path)

    if not content or not extraction_result.success:
        result.duration_seconds = time.time() - start_time
        result.success = extraction_result.success
        return result

    # Smart chunking with sentence boundary detection
    start = 0
    chunk_index = 0
    chunks = []

    while start < len(content):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(content):
            search_start = max(end - 100, start)
            for punct in ['. ', '! ', '? ', '\n\n']:
                pos = content.rfind(punct, search_start, end)
                if pos > search_start:
                    end = pos + len(punct)
                    break

        chunk_text = content[start:end].strip()

        if chunk_text:
            chunks.append({
                "index": chunk_index,
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "length": len(chunk_text)
            })
            chunk_index += 1

        start = end - chunk_overlap

    result.chunks = chunks
    result.total_chunks = len(chunks)
    result.avg_chunk_size = sum(c["length"] for c in chunks) // max(len(chunks), 1)
    result.duration_seconds = time.time() - start_time

    logger.info(f"Created {result.total_chunks} chunks, avg size: {result.avg_chunk_size} chars")

    return result


@task(
    name="embed_document",
    description="Generate embeddings and store in MongoDB",
    retries=2,
    retry_delay_seconds=30,
    tags=["embeddings", "mongodb", "storage"]
)
async def embed_document_task(
    chunking_result: ChunkingResult,
    extraction_result: ContentExtractionResult,
    collection_name: str = "knowledge_base",
    model_name: str = "BAAI/bge-base-en-v1.5",
    use_local_embeddings: bool = True
) -> DocumentEmbeddingResult:
    """
    Generate embeddings for document chunks and store in MongoDB.

    Uses the new LocalEmbeddingService for faster, higher quality embeddings
    when use_local_embeddings=True (default).

    Args:
        chunking_result: Result from chunking task
        extraction_result: Original extraction result for metadata
        collection_name: MongoDB collection to store documents
        model_name: Embedding model to use (default: BAAI/bge-base-en-v1.5)
        use_local_embeddings: Use local sentence-transformers (faster, better quality)

    Returns:
        DocumentEmbeddingResult with storage confirmation
    """
    logger = get_run_logger()
    start_time = time.time()

    file_path = chunking_result.file_path
    chunks = chunking_result.chunks

    logger.info(f"Embedding {len(chunks)} chunks from: {file_path}")

    result = DocumentEmbeddingResult(file_path=file_path)

    if not chunks or not chunking_result.success:
        result.errors.append("No chunks to embed")
        result.success = False
        result.duration_seconds = time.time() - start_time
        return result

    try:
        import sys
        sys.path.insert(0, '..')

        # Generate document ID from file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        result.document_id = f"doc_{file_hash}"

        # Generate embeddings for all chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = []

        if use_local_embeddings:
            # Use new LocalEmbeddingService (faster, better quality)
            try:
                from services.document_embedder import LocalEmbeddingService
                embedder = LocalEmbeddingService(model_name=model_name)
                embeddings = await embedder.embed_batch(chunk_texts)
                result.embedding_model = embedder.model_name
                result.embedding_dim = embedder.embedding_dimension
                result.use_local_embeddings = True
                logger.info(f"Using local embeddings: {result.embedding_model} (dim={result.embedding_dim})")
            except ImportError as e:
                logger.warning(f"LocalEmbeddingService not available: {e}, falling back to EmbeddingService")
                use_local_embeddings = False

        if not use_local_embeddings:
            # Fallback to original EmbeddingService
            from embedding_service import EmbeddingService
            embedding_service = EmbeddingService(model_name=model_name)
            embeddings = embedding_service.generate_embeddings_batch(chunk_texts)
            result.embedding_model = model_name
            result.use_local_embeddings = False

        result.vectors_generated = len(embeddings)

        # Store in MongoDB
        from mongodb import MongoDBService
        mongodb = MongoDBService()
        await mongodb.connect()

        # Prepare documents for MongoDB
        collection = mongodb.db[collection_name]

        # Delete existing chunks for this document
        await collection.delete_many({"document_id": result.document_id})

        # Insert new chunks with embeddings
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Handle both list and numpy array embeddings
            vector = embedding if isinstance(embedding, list) else embedding.tolist()
            doc = {
                "document_id": result.document_id,
                "file_path": file_path,
                "file_type": extraction_result.file_type,
                "chunk_index": i,
                "text": chunk["text"],
                "vector": vector,
                "metadata": extraction_result.metadata,
                "embedding_model": result.embedding_model,
                "created_at": datetime.utcnow().isoformat()
            }
            documents.append(doc)

        if documents:
            await collection.insert_many(documents)
            result.chunks_embedded = len(documents)
            result.stored_in_mongodb = True

        logger.info(f"Stored {result.chunks_embedded} chunks with vectors in MongoDB")

    except Exception as e:
        error_msg = f"Embedding/storage failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    result.duration_seconds = time.time() - start_time

    # Create Prefect artifact with enhanced reporting
    await create_markdown_artifact(
        key="embedding-result",
        markdown=f"""
## Document Embedding Results
- **File**: {os.path.basename(file_path)}
- **Document ID**: {result.document_id}
- **Chunks Embedded**: {result.chunks_embedded}
- **Vectors Generated**: {result.vectors_generated}
- **Stored in MongoDB**: {"Yes" if result.stored_in_mongodb else "No"}
- **Duration**: {result.duration_seconds:.2f}s
- **Status**: {"Success" if result.success else "Failed"}

### Embedding Details
- **Model**: {result.embedding_model}
- **Dimension**: {result.embedding_dim if result.embedding_dim else "N/A"}
- **Local Embeddings**: {"Yes (sentence-transformers)" if result.use_local_embeddings else "No (external API)"}
        """,
        description=f"Embedding for {os.path.basename(file_path)}"
    )

    return result


@flow(
    name="document-processing-pipeline",
    description="Complete Document Processing Pipeline with Local Embeddings",
    retries=1,
    retry_delay_seconds=60
)
async def document_flow(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "knowledge_base",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    use_local_embeddings: bool = True
) -> Dict[str, Any]:
    """
    Complete Document Processing Pipeline with improved embeddings.

    This flow:
    1. Detects document type and validates support
    2. Extracts content and metadata
    3. Chunks content for optimal embedding
    4. Generates vectors using local sentence-transformers (faster, better quality)
    5. Stores in MongoDB

    Args:
        file_path: Path to document to process
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        collection_name: MongoDB collection for storage
        embedding_model: Model for vector generation (default: BAAI/bge-base-en-v1.5)
        use_local_embeddings: Use local sentence-transformers (default: True)

    Returns:
        Dict with complete pipeline results
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting document processing for: {file_path}")

    # Step 1: Detect document type
    type_result = await detect_type_task(file_path=file_path)

    # Step 2: Extract content
    extraction_result = await extract_content_task(type_result=type_result)

    # Step 3: Chunk document
    chunking_result = await chunk_document_task(
        extraction_result=extraction_result,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Step 4: Embed and store (with local embeddings by default)
    embedding_result = await embed_document_task(
        chunking_result=chunking_result,
        extraction_result=extraction_result,
        collection_name=collection_name,
        model_name=embedding_model,
        use_local_embeddings=use_local_embeddings
    )

    total_duration = time.time() - flow_start

    all_errors = extraction_result.errors + embedding_result.errors
    overall_success = embedding_result.success and embedding_result.stored_in_mongodb

    # Create final flow summary with embedding details
    await create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"""
# Document Processing Complete

## Overview
- **File**: {os.path.basename(file_path)}
- **Type**: {type_result.file_type}
- **Size**: {type_result.file_size_bytes/1024:.1f} KB
- **Total Duration**: {total_duration:.2f}s
- **Status**: {"Success" if overall_success else "Failed"}

## Processing Results
| Stage | Result | Duration |
|-------|--------|----------|
| Detection | {type_result.file_type} | <1s |
| Extraction | {extraction_result.content_length:,} chars | {extraction_result.duration_seconds:.1f}s |
| Chunking | {chunking_result.total_chunks} chunks | {chunking_result.duration_seconds:.1f}s |
| Embedding | {embedding_result.vectors_generated} vectors | {embedding_result.duration_seconds:.1f}s |

## Embedding Details
- **Model**: {embedding_result.embedding_model}
- **Dimension**: {embedding_result.embedding_dim if embedding_result.embedding_dim else "N/A"}
- **Local Embeddings**: {"Yes (sentence-transformers)" if embedding_result.use_local_embeddings else "No (external API)"}

## Storage
- **Document ID**: {embedding_result.document_id}
- **Collection**: {collection_name}
- **Chunks Stored**: {embedding_result.chunks_embedded}

{"## Errors" + chr(10) + chr(10).join(f"- {e}" for e in all_errors) if all_errors else ""}
        """,
        description=f"Document pipeline summary for {os.path.basename(file_path)}"
    )

    return {
        "success": overall_success,
        "file_path": file_path,
        "file_type": type_result.file_type,
        "document_id": embedding_result.document_id,
        "total_duration_seconds": total_duration,
        "stages": {
            "detection": {
                "file_type": type_result.file_type,
                "file_size_bytes": type_result.file_size_bytes,
                "is_supported": type_result.is_supported
            },
            "extraction": {
                "content_length": extraction_result.content_length,
                "pages": extraction_result.pages,
                "tables_found": extraction_result.tables_found,
                "duration": extraction_result.duration_seconds,
                "success": extraction_result.success
            },
            "chunking": {
                "total_chunks": chunking_result.total_chunks,
                "avg_chunk_size": chunking_result.avg_chunk_size,
                "duration": chunking_result.duration_seconds
            },
            "embedding": {
                "vectors_generated": embedding_result.vectors_generated,
                "chunks_embedded": embedding_result.chunks_embedded,
                "stored_in_mongodb": embedding_result.stored_in_mongodb,
                "duration": embedding_result.duration_seconds,
                "success": embedding_result.success,
                "model": embedding_result.embedding_model,
                "dimension": embedding_result.embedding_dim,
                "use_local_embeddings": embedding_result.use_local_embeddings
            }
        },
        "errors": all_errors
    }


def run_document_flow(
    file_path: str,
    collection_name: str = "knowledge_base"
) -> Dict[str, Any]:
    """
    Convenience function to run the document flow synchronously.

    Example:
        from prefect_pipelines import run_document_flow

        result = run_document_flow(
            file_path="/path/to/document.pdf",
            collection_name="my_docs"
        )
    """
    return asyncio.run(document_flow(
        file_path=file_path,
        collection_name=collection_name
    ))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = run_document_flow(sys.argv[1])
        print(f"Result: {result}")
    else:
        print("Usage: python document_flow.py <file_path>")
