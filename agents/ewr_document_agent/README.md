# EWR Document Agent

Document processing agent for ETL pipelines in the EWR Multi-Agent Framework.

## Overview

The EWR Document Agent provides comprehensive document processing capabilities including:

- **Document Parsing**: PDF, DOCX, DOC, HTML, Markdown, TXT, JSON, XML, CSV, and code files
- **Intelligent Chunking**: Multiple strategies (recursive, sentence, paragraph, fixed-size, markdown, code-aware)
- **Embedding Generation**: Integration with llama.cpp (nomic-embed-text model)
- **Vector Storage**: MongoDB with vector search support
- **Semantic Search**: Find relevant document chunks using embeddings

## Installation

```bash
# Basic installation
pip install -e agents/ewr_document_agent

# With API support (FastAPI endpoints)
pip install -e agents/ewr_document_agent[api]

# With all optional features
pip install -e agents/ewr_document_agent[all]
```

**Dependencies**: Requires `ewr-agent-core` to be installed first:
```bash
pip install -e agents/ewr_agent_core
```

## Capabilities

| Capability | Description |
|------------|-------------|
| `DOC_SEARCH` | Semantic search across document chunks |
| `DOC_SUMMARIZE` | Document summarization support |
| `MULTI_DOC` | Process multiple documents in batch |

## Chunking Strategies

| Strategy | Use Case |
|----------|----------|
| `recursive` | Default - splits by logical separators |
| `sentence` | Natural language documents |
| `paragraph` | Well-structured documents |
| `fixed_size` | When consistent chunk size needed |
| `markdown` | Markdown files (header-aware) |
| `code` | Source code (function/class-aware) |

## Usage

### Standalone Usage

```python
import asyncio
from ewr_document_agent import DocumentAgent
from ewr_document_agent.models import VectorStoreConfig, ChunkingConfig, ChunkingStrategy
from ewr_agent_core.llm_backends import LlamaCppBackend

async def main():
    # Configure vector store
    vector_config = VectorStoreConfig(
        store_type="mongodb",
        connection_string="mongodb://localhost:27017",
        database_name="EWRAI",
        collection_name="document_agent_chunks",
    )

    # Configure chunking
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=500,
        chunk_overlap=50,
    )

    # Create agent
    agent = DocumentAgent(
        vector_config=vector_config,
        chunking_config=chunking_config,
    )

    # Set LLM backend for embeddings
    agent.set_llm(LlamaCppBackend(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
    ))

    # Start agent
    await agent.start()

    # Process a document
    result = await agent.process_document("path/to/document.pdf")
    print(f"Processed {result.total_chunks} chunks")

    # Search for content
    results = await agent.search(
        query="What is the main topic?",
        top_k=5,
        min_score=0.3,
    )

    for r in results:
        print(f"Score: {r.score:.2f} - {r.chunk.content[:100]}...")

    # Clean up
    await agent.stop()

asyncio.run(main())
```

### CLI Usage

```bash
# Process a document
ewr-document process /path/to/document.pdf --strategy recursive --chunk-size 500

# Search documents
ewr-document search "your query" --top-k 5 --min-score 0.3

# Run the API server
ewr-document-api --host 0.0.0.0 --port 8002
```

## RAG Server Integration

The document agent is integrated into the main RAG server via FastAPI routes.

### API Endpoints

All endpoints are prefixed with `/document-agent`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Agent statistics |
| `/process` | POST | Process document from file path |
| `/upload` | POST | Upload and process document |
| `/search` | POST | Semantic search |
| `/jobs/{job_id}` | GET | Get job status |
| `/jobs/{job_id}/stream` | GET | Stream job updates (SSE) |
| `/documents/{id}` | GET | Get document info |
| `/documents/{id}` | DELETE | Delete document |
| `/documents/{id}/chunks` | GET | Get document chunks |

### Example API Calls

```bash
# Health check
curl http://localhost:8001/document-agent/health

# Upload a document
curl -X POST http://localhost:8001/document-agent/upload \
  -F "file=@document.pdf" \
  -F "chunking_strategy=recursive" \
  -F "chunk_size=500"

# Search
curl -X POST http://localhost:8001/document-agent/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 5}'

# Delete a document
curl -X DELETE http://localhost:8001/document-agent/documents/{document_id}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `EWRAI` | Database name |
| `LLM_BASE_URL` | `http://localhost:11434` | llama.cpp server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |

### ChunkingConfig Options

```python
ChunkingConfig(
    strategy=ChunkingStrategy.RECURSIVE,  # Chunking strategy
    chunk_size=500,                        # Target chunk size (chars)
    chunk_overlap=50,                      # Overlap between chunks
    min_chunk_size=100,                    # Minimum chunk size
    max_chunk_size=2000,                   # Maximum chunk size
    separators=["\n\n", "\n", ". ", " "], # Separators for recursive
)
```

### VectorStoreConfig Options

```python
VectorStoreConfig(
    store_type="mongodb",                              # Vector store type
    connection_string="mongodb://localhost:27017",     # Connection string
    database_name="EWRAI",                             # Database name
    collection_name="document_agent_chunks",           # Collection name
)
```

## Architecture

```
Document → Parse → Chunk → Embed → Store → Search
   │         │        │       │       │       │
   ▼         ▼        ▼       ▼       ▼       ▼
  PDF     Extract  Split   LLM    MongoDB  Vector
  DOCX    content  text    embed   insert  similarity
  HTML    meta     by              index   search
  ...     data   strategy
```

## File Structure

```
ewr_document_agent/
├── src/
│   └── ewr_document_agent/
│       ├── __init__.py       # Package exports
│       ├── agent.py          # Main DocumentAgent class
│       ├── models.py         # Data models (Document, Chunk, etc.)
│       ├── api.py            # Standalone FastAPI server
│       ├── retry.py          # Retry utilities
│       └── cli.py            # CLI interface
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Testing

Run the E2E tests:

```bash
cd /path/to/llm_website
npx playwright test tests/e2e/document-agent.spec.js
```

## Inter-Agent Communication

The document agent can participate in the multi-agent framework:

```python
# Request help from document agent via message broker
response = await code_agent.request_help(
    capability="DOC_SEARCH",
    task_description="Find documentation about authentication",
    task_params={"query": "authentication flow", "top_k": 3},
)
```

## Supported Document Types

| Extension | Type | Parser |
|-----------|------|--------|
| `.pdf` | PDF | pypdf |
| `.docx` | Word | python-docx |
| `.doc` | Word (legacy) | python-docx |
| `.html`, `.htm` | HTML | BeautifulSoup |
| `.md`, `.markdown` | Markdown | Built-in |
| `.txt` | Plain text | Built-in |
| `.json` | JSON | Built-in |
| `.xml` | XML | Built-in |
| `.csv` | CSV | Built-in |
| `.py`, `.js`, `.ts`, etc. | Code | Built-in |

## Performance

- Small files (<1MB): Processed synchronously
- Large files (>1MB): Background processing with job tracking
- Batch processing: Use `process_folder()` for multiple files

## Troubleshooting

### No search results

1. Check if llama.cpp is running: `curl http://localhost:11434/health`
2. Verify chunks have embeddings: Check `has_embedding: true` in `/documents/{id}/chunks`
3. Try lowering `min_score` parameter

### MongoDB connection issues

1. Verify MongoDB is running
2. Check connection string in environment variables
3. Ensure `motor` package is installed

### Import errors

Ensure both packages are installed:
```bash
pip install -e agents/ewr_agent_core
pip install -e agents/ewr_document_agent[api]
```
