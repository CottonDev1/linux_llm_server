"""
Services module for Python RAG Server.

This module provides core business logic services:
- LLMService: Async wrapper for llama.cpp API with streaming support
- SQLGeneratorService: DEPRECATED - Use sql_pipeline.QueryPipeline instead
- DocumentRerankerService: Cross-encoder re-ranking for document retrieval
- LocalEmbeddingService: Local embedding generation with sentence-transformers

MIGRATION NOTICE:
-----------------
SQLGeneratorService is deprecated. For new code, use:

    from sql_pipeline import get_query_pipeline
    pipeline = await get_query_pipeline()
    result = await pipeline.generate(question=..., database=..., server=...)

The sql_pipeline.QueryPipeline provides the same functionality with better
modular architecture and improved maintainability.
"""

from .llm_service import LLMService, get_llm_service
from .sql_generator import SQLGeneratorService, get_sql_generator_service  # Deprecated
from .document_reranker import DocumentRerankerService, get_reranker_service
from .document_embedder import LocalEmbeddingService, get_embedding_service

__all__ = [
    "LLMService",
    "get_llm_service",
    # SQLGeneratorService is deprecated - use sql_pipeline.QueryPipeline instead
    "SQLGeneratorService",
    "get_sql_generator_service",
    "DocumentRerankerService",
    "get_reranker_service",
    "LocalEmbeddingService",
    "get_embedding_service",
]
