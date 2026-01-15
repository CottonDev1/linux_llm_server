"""
Code Assistance Query Pipeline

This module provides RAG-based code assistance for C# codebase Q&A,
including multi-step retrieval and LLM-powered response generation.

Migrated from: src/routes/codeRoutes.js
"""

from .pipeline import CodeAssistancePipeline

__all__ = ["CodeAssistancePipeline"]
