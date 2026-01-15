"""
EWR Document Agent Models
=========================

Pydantic models for document processing and retrieval.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    CODE = "code"
    AUDIO = "audio"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"          # Fixed character/token count
    SENTENCE = "sentence"               # Split by sentences
    PARAGRAPH = "paragraph"             # Split by paragraphs
    SEMANTIC = "semantic"               # Use LLM to find semantic breaks
    RECURSIVE = "recursive"             # Recursive text splitter
    CODE = "code"                       # Code-aware splitting
    MARKDOWN = "markdown"               # Markdown header/section aware
    CUSTOM = "custom"                   # Custom splitting logic


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    file_size: int = 0
    page_count: int = 0
    word_count: int = 0
    language: Optional[str] = None
    source_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A document to be processed."""
    id: str
    file_path: str
    document_type: DocumentType = DocumentType.UNKNOWN
    content: str = ""
    raw_content: bytes = b""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    version: int = 1

    class Config:
        arbitrary_types_allowed = True


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 500              # Target chunk size in characters
    chunk_overlap: int = 50            # Overlap between chunks
    min_chunk_size: int = 100          # Minimum chunk size
    max_chunk_size: int = 2000         # Maximum chunk size
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    preserve_metadata: bool = True     # Include source metadata in chunks
    include_headers: bool = True       # Include section headers
    code_languages: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A chunk of a document."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    page_number: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)
    embedding_model: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EmbeddingBatch(BaseModel):
    """Batch of chunks for embedding."""
    batch_id: str
    chunks: List[DocumentChunk] = Field(default_factory=list)
    model: str = "nomic-embed-text"
    dimensions: int = 768
    status: ProcessingStatus = ProcessingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of document processing."""
    document_id: str
    file_path: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    document_type: DocumentType = DocumentType.UNKNOWN
    total_chunks: int = 0
    chunks_embedded: int = 0
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    processing_time_ms: int = 0
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Result from vector search."""
    chunk: DocumentChunk
    score: float
    document_path: str
    highlights: List[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    """Query for document search."""
    query: str
    top_k: int = 5
    min_score: float = 0.0
    filter_document_ids: Optional[List[str]] = None
    filter_document_types: Optional[List[DocumentType]] = None
    filter_tags: Optional[List[str]] = None
    include_metadata: bool = True
    rerank: bool = False


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    store_type: str = "mongodb"        # mongodb (default), chromadb, pinecone, etc.
    connection_string: Optional[str] = "mongodb://localhost:27017"
    database_name: str = "EWRAI"
    collection_name: str = "documents"
    embedding_dimensions: int = 768
    distance_metric: str = "cosine"    # cosine, euclidean, dot
    batch_size: int = 100              # Batch size for inserts


class DocumentBatch(BaseModel):
    """Batch of documents for processing."""
    batch_id: str
    documents: List[str] = Field(default_factory=list)  # File paths
    config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    results: List[ProcessingResult] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DocumentUpdate(BaseModel):
    """Update to an existing document."""
    document_id: str
    action: str = "update"  # update, delete, reprocess
    new_path: Optional[str] = None
    new_metadata: Optional[Dict[str, Any]] = None
    reprocess_embeddings: bool = False


class ExtractionConfig(BaseModel):
    """Configuration for content extraction."""
    extract_tables: bool = True
    extract_images: bool = False
    extract_links: bool = True
    extract_code_blocks: bool = True
    ocr_enabled: bool = False
    ocr_language: str = "eng"
    max_image_size: int = 10_000_000   # 10MB
    preserve_formatting: bool = False
