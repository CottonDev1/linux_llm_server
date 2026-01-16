# Token-Aware Document Chunking Implementation Plan

## Problem Statement
Document uploads fail when chunks exceed the embedding service's 512 token limit. The current chunking uses character-based limits (1000 chars) without accounting for:
- Actual token counts
- Metadata overhead (title, subject, tags)
- The embedding model's token limit

## Research Findings

### Current Implementation
- **Chunking**: Character-based (1000 chars max, 200 char overlap) in `mongodb/helpers.py`
- **Storage**: Chunks stored with `parent_id`, `chunk_index`, `total_chunks` in MongoDB
- **Retrieval**: `get_document()` already reconstructs full documents by joining chunks sorted by `chunk_index`

### Token Estimation
- Average: ~4 characters per token for English text
- With metadata overhead: ~30-45 tokens per chunk
- Safe chunk size for 512 token limit: ~400 tokens content = ~1600 characters
- With safety margin (15%): ~350 tokens content = ~1400 characters

### Nomic Embed Text v1.5 Specs
- Max context: 8192 tokens
- Current server limit: 512 tokens (ubatch-size)
- Recommended production: 400-512 tokens per chunk

## Implementation Plan

### Phase 1: Add Token Counting Utility
**File**: `python_services/mongodb/helpers.py`

Add a token estimation function that:
1. Estimates tokens from character count (4 chars/token baseline)
2. Accounts for metadata overhead
3. Provides a safety margin

```python
def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length."""
    return int(len(text) / chars_per_token)

def calculate_max_chunk_chars(
    max_tokens: int = 450,  # Target under 512 with margin
    metadata_tokens: int = 50,  # Title, subject, tags overhead
    chars_per_token: float = 4.0
) -> int:
    """Calculate max chunk size in characters to stay under token limit."""
    available_tokens = max_tokens - metadata_tokens
    return int(available_tokens * chars_per_token)
```

### Phase 2: Update chunk_document Function
**File**: `python_services/mongodb/helpers.py`

Modify `chunk_document()` to:
1. Accept a `max_tokens` parameter (default 450)
2. Calculate character limit dynamically based on token budget
3. Split on sentence boundaries within token limits
4. Maintain overlap for context preservation

```python
def chunk_document(
    text: str,
    max_tokens: int = 450,  # Stay under 512 with metadata
    overlap_tokens: int = 50,
    metadata_tokens: int = 50
) -> List[str]:
    """
    Chunk documents with token-aware sizing.
    Ensures each chunk + metadata stays under embedding model's token limit.
    """
```

### Phase 3: Verify Document Retrieval
**File**: `python_services/routes/document_routes.py`

Verify these endpoints return complete documents:
1. `GET /documents/{document_id}` - uses `get_document()` which already joins chunks
2. `GET /documents/by-title` - already reconstructs from chunks
3. Search endpoints - return chunk matches with parent_id for full doc retrieval

### Phase 4: Configuration
**File**: `python_services/config.py`

Add configurable chunking parameters:
```python
# Chunking Configuration (token-aware)
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
CHUNK_METADATA_OVERHEAD = int(os.getenv("CHUNK_METADATA_OVERHEAD", "50"))
```

## Files to Modify

| File | Changes |
|------|---------|
| `mongodb/helpers.py` | Add token estimation, update `chunk_document()` |
| `config.py` | Add chunking configuration variables |
| `mongodb/documents.py` | Use new chunking config (minor) |

## Testing Plan

1. **Unit Test**: Verify chunks stay under 450 tokens
2. **Integration Test**: Upload the PDF that previously failed
3. **Retrieval Test**: Verify `get_document()` returns complete content
4. **Search Test**: Verify search returns relevant chunks with parent references

## Rollback Plan

If issues arise:
- Chunking config is environment-based, can revert via env vars
- Increase embedding server batch size as fallback (already tested working)

## Success Criteria

1. PDF uploads succeed without "input too large" errors
2. Each chunk stays under 512 tokens (including metadata)
3. Full document reconstruction returns complete content
4. Search returns relevant results with ability to fetch full document
