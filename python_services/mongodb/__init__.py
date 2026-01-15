"""
MongoDB Service Package

Modular MongoDB service composed of specialized mixins for different domains.

This package refactors the monolithic mongodb_service.py into focused modules:
- base.py: Core infrastructure (connection, vector search, indexes)
- documents.py: Document CRUD operations
- code_context.py: Code context storage and search
- sql.py: SQL-related operations (knowledge, examples, corrections, schema, procedures)
- feedback.py: Feedback and quality scoring
- audio.py: Audio analysis operations
- ticket_matching.py: Ticket matching and phone mapping
- helpers.py: Shared utility functions

Usage:
    from mongodb import MongoDBService, get_mongodb_service

    # Get singleton instance
    mongo = get_mongodb_service()
    await mongo.initialize()

    # Or create new instance
    mongo = MongoDBService()
    await mongo.initialize()
"""

from typing import Optional

from .base import MongoDBBase, VectorSearchError, VectorSearchUnavailableError
from .documents import DocumentsMixin
from .code_context import CodeContextMixin
from .sql import SQLMixin
from .feedback import FeedbackMixin
from .audio import AudioMixin
from .ticket_matching import TicketMatchingMixin


class MongoDBService(
    MongoDBBase,
    DocumentsMixin,
    CodeContextMixin,
    SQLMixin,
    FeedbackMixin,
    AudioMixin,
    TicketMatchingMixin
):
    """
    Unified MongoDB service composed of specialized mixins.

    This class inherits from all domain-specific mixins, providing a single
    unified interface for all MongoDB operations while keeping code organized
    by functional domain.

    Mixins:
        MongoDBBase: Core connection, vector search, and index management
        DocumentsMixin: Document storage, search, and management
        CodeContextMixin: Code context operations
        SQLMixin: All SQL-related operations (knowledge, examples, corrections, schema, procedures)
        FeedbackMixin: Feedback storage and quality scoring
        AudioMixin: Audio analysis operations
        TicketMatchingMixin: Ticket matching and phone mapping

    Example:
        mongo = MongoDBService()
        await mongo.initialize()

        # Document operations
        doc_id = await mongo.store_document(title="Test", content="Content")
        results = await mongo.search_documents("query")

        # SQL operations
        context = await mongo.get_comprehensive_sql_context("find tickets", "EWRCentral")

        # Audio operations
        await mongo.store_audio_analysis(...)

        await mongo.close()
    """

    _instance: Optional['MongoDBService'] = None

    @classmethod
    def get_instance(cls) -> 'MongoDBService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_mongodb_service() -> MongoDBService:
    """Get the singleton MongoDB service instance"""
    return MongoDBService.get_instance()


__all__ = [
    'MongoDBService',
    'get_mongodb_service',
    'MongoDBBase',
    'DocumentsMixin',
    'CodeContextMixin',
    'SQLMixin',
    'FeedbackMixin',
    'AudioMixin',
    'TicketMatchingMixin',
    'VectorSearchError',
    'VectorSearchUnavailableError'
]
