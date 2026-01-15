"""
MongoDB Service - Shared Helper Functions

Common utilities used across all MongoDB service mixins.
"""
import hashlib
from typing import Optional, List


def chunk_document(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Chunk large documents into smaller pieces with overlap.

    Args:
        text: The text to chunk
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    sentences = text.split('. ')

    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk + sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Add overlap
            words = current_chunk.split(' ')
            overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
            current_chunk = ' '.join(overlap_words) + ' ' + sentence
        else:
            current_chunk += ('. ' if current_chunk else '') + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def format_embedding_text(
    content: str,
    title: Optional[str] = None,
    subject: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> str:
    """
    Format text for embedding with semantic metadata to improve retrieval.

    This method prepends metadata (title, subject, tags) to the content before
    embedding. This technique improves retrieval by:
    1. Adding semantic context that may not be in the content itself
    2. Allowing queries about metadata to match relevant documents
    3. Boosting relevance when both content and metadata match

    Research shows that including structured metadata in embeddings can
    improve retrieval accuracy by 5-10% for document search tasks.

    Args:
        content: The main text content to embed
        title: Optional document title
        subject: Optional document subject/category
        tags: Optional list of tags/keywords

    Returns:
        Formatted text string ready for embedding
    """
    parts = []

    # Add title as semantic anchor (high importance for retrieval)
    if title:
        parts.append(f"Title: {title}")

    # Add subject for categorical context
    if subject:
        parts.append(f"Subject: {subject}")

    # Add tags as keywords (helps with specific term matching)
    if tags and len(tags) > 0:
        # Filter out generic tags like "untagged"
        meaningful_tags = [t for t in tags if t and t.lower() != "untagged"]
        if meaningful_tags:
            parts.append(f"Keywords: {', '.join(meaningful_tags)}")

    # Add separator if we have metadata
    if parts:
        parts.append("")  # Empty line as separator

    # Add the main content
    parts.append(content)

    return "\n".join(parts)


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of content for change detection and deduplication.

    Args:
        content: The content to hash

    Returns:
        First 16 characters of the SHA256 hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string (e.g., "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
