"""
Query/RAG Pipeline

This module provides vector search and LLM generation orchestration
for general RAG queries across code and documentation.

Migrated from: src/routes/queryRoutes.js
"""

from .pipeline import QueryPipeline

__all__ = ["QueryPipeline"]
