"""
MongoDB Service - Shared Helper Functions

Common utilities used across all MongoDB service mixins.
"""
import hashlib
from typing import Optional, List

from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_METADATA_OVERHEAD,
    CHARS_PER_TOKEN
)


def calculate_max_chunk_chars(
    max_tokens: int = CHUNK_MAX_TOKENS,
    metadata_overhead: int = CHUNK_METADATA_OVERHEAD,
    chars_per_token: float = CHARS_PER_TOKEN
) -> int:
    """Calculate maximum chunk size in characters to stay under token limit."""
    available_tokens = max_tokens - metadata_overhead
    return int(available_tokens * chars_per_token)


def _split_long_text(text: str, max_size: int) -> List[str]:
    """Force-split text that exceeds max_size by splitting on newlines, then words."""
    if len(text) <= max_size:
        return [text]

    pieces = []
    lines = text.split('\n')
    current = ''
    for line in lines:
        if len(current) + len(line) + 1 <= max_size:
            current = current + '\n' + line if current else line
        else:
            if current:
                pieces.append(current)
            if len(line) > max_size:
                words = line.split(' ')
                current = ''
                for word in words:
                    if len(current) + len(word) + 1 <= max_size:
                        current = current + ' ' + word if current else word
                    else:
                        if current:
                            pieces.append(current)
                        current = word
            else:
                current = line
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    metadata_overhead: int = CHUNK_METADATA_OVERHEAD,
    chars_per_token: float = CHARS_PER_TOKEN
) -> List[str]:
    """
    Chunk documents with token-aware sizing, force-splitting long sections.

    Args:
        text: The text to chunk
        max_tokens: Target maximum tokens per chunk (default 400)
        overlap_tokens: Tokens to overlap between chunks (default 50)
        metadata_overhead: Token overhead for metadata (default 50)
        chars_per_token: Average characters per token (default 4.0)

    Returns:
        List of text chunks, each under the token limit
    """
    max_chunk_size = calculate_max_chunk_chars(max_tokens, metadata_overhead, chars_per_token)
    overlap = int(overlap_tokens * chars_per_token)

    chunks = []
    sentences = text.replace('\n\n', '. ').split('. ')

    current_chunk = ''
    for sentence in sentences:
        # Force-split very long sentences
        if len(sentence) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ''
            for piece in _split_long_text(sentence, max_chunk_size):
                chunks.append(piece.strip())
            continue

        if len(current_chunk + sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
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
