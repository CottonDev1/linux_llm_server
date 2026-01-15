"""
EWR Document Agent
==================

Document processing, chunking, embedding, and retrieval agent.

This agent handles document ETL pipelines:
1. Ingests documents from various sources (PDF, DOCX, HTML, TXT, etc.)
2. Chunks documents using configurable strategies
3. Generates embeddings for semantic search
4. Stores in MongoDB with vector search support
5. Retrieves relevant chunks for RAG queries

Capabilities:
- Multi-format document parsing (PDF, DOCX, HTML, Markdown, TXT)
- Configurable chunking (size, overlap, semantic boundaries)
- Batch embedding generation
- Vector store integration
- Metadata extraction and management
- Document versioning and updates

Usage:
    from ewr_document_agent import DocumentAgent

    agent = DocumentAgent(name="doc-processor")
    await agent.start()

    # Process a document
    result = await agent.process_document("/path/to/document.pdf")

    # Search for relevant chunks
    chunks = await agent.search("query text", top_k=5)

    # Process entire folder
    results = await agent.process_folder("/path/to/docs")
"""

from .agent import DocumentAgent
from .models import (
    Document,
    DocumentChunk,
    ChunkingConfig,
    ChunkingStrategy,
    ProcessingResult,
    ProcessingStatus,
    SearchResult,
    DocumentMetadata,
    EmbeddingBatch,
    VectorStoreConfig,
)
from .retry import (
    RetryConfig,
    with_retry,
    with_retry_sync,
    retry_async,
    retry_sync,
    EMBEDDING_RETRY,
    STORAGE_RETRY,
    FILE_RETRY,
    HTTP_RETRY,
)

__version__ = "1.0.0"

__all__ = [
    # Agent
    "DocumentAgent",
    # Models
    "Document",
    "DocumentChunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "ProcessingResult",
    "ProcessingStatus",
    "SearchResult",
    "DocumentMetadata",
    "EmbeddingBatch",
    "VectorStoreConfig",
    # Retry utilities
    "RetryConfig",
    "with_retry",
    "with_retry_sync",
    "retry_async",
    "retry_sync",
    "EMBEDDING_RETRY",
    "STORAGE_RETRY",
    "FILE_RETRY",
    "HTTP_RETRY",
]
