"""
Git Repository Analysis Pipeline

This module provides git repository synchronization, Roslyn analysis
invocation, and vector database import orchestration.

Migrated from: src/routes/gitRoutes.js
"""

from .pipeline import GitSyncPipeline, create_pipeline

# Alias for backward compatibility
GitPipeline = GitSyncPipeline

__all__ = ["GitSyncPipeline", "GitPipeline", "create_pipeline"]
