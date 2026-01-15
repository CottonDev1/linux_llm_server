"""
RAG Pipeline for SQL Schema Context

This module implements a proper RAG pipeline following nilenso best practices:
1. EXTRACTION - Extract schema/SP data from SQL Server
2. ENRICHMENT - Generate LLM summaries for human-readable descriptions
3. EMBEDDING  - Generate vector embeddings from enriched summaries
4. RETRIEVAL  - Semantic search to find relevant context
5. GENERATION - Use context to generate SQL (handled by main RAG endpoint)

Key insight: Embed the LLM-generated summaries, NOT the raw schema/SQL code.
This improves retrieval accuracy by ~7% according to nilenso research.

API Endpoints (via FastAPI main.py):
------------------------------------
GET  /pipeline/status/{database}  - Get pipeline status for a database
POST /pipeline/run                - Run full pipeline (summarize + embed)
POST /pipeline/summarize          - Run summarization stage only
POST /pipeline/embed              - Run embedding stage only

Node.js Proxy Routes (via sqlRoutes.js):
----------------------------------------
GET  /api/sql/pipeline/status/:database  - Proxy to Python status
POST /api/sql/pipeline/run               - Proxy to Python run
POST /api/sql/pipeline/summarize         - Proxy to Python summarize
POST /api/sql/pipeline/embed             - Proxy to Python embed
POST /api/sql/pipeline/extract           - Proxy to Python extract

Usage:
------
1. Extract data from SQL Server (stores in MongoDB)
2. Run summarization to generate LLM descriptions
3. Run embedding to create searchable vectors
4. Query via /sql/comprehensive-context endpoint

Example:
    # Check status
    curl http://localhost:3000/api/sql/pipeline/status/ewrcentral

    # Run full pipeline
    curl -X POST http://localhost:3000/api/sql/pipeline/run \\
         -H "Content-Type: application/json" \\
         -d '{"database": "ewrcentral", "model": "llama3.2:3b"}'
"""

from .pipeline import SQLRAGPipeline
from .embedder import SchemaEmbedder, ProcedureEmbedder

__all__ = [
    'SQLRAGPipeline',
    'SchemaEmbedder',
    'ProcedureEmbedder'
]
