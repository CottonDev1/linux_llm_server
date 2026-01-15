"""
Vector Search Service
=====================

Orchestrates vector search across different backends:
1. Code Context - MongoDB collection for code snippets, procedures, etc.
2. Documentation - MongoDB collection for knowledge base documents

Design Rationale:
-----------------
This service abstracts the vector search backend, allowing the pipeline
to switch between different search strategies without changing business logic.

Key features:
- Unified interface for code and documentation search
- Support for project-scoped searches
- Cross-project search (e.g., include EWRLibrary)
- Consistent result formatting

Architecture Notes:
- Uses MongoDB's vector search capabilities when available
- Falls back to in-memory numpy cosine similarity when needed
- Results are normalized to a common VectorSearchResult format
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from query_pipeline.models.query_models import VectorSearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchOptions:
    """Options for vector search."""
    limit: int = 10
    project: Optional[str] = None
    include_ewr_library: bool = False
    similarity_threshold: float = 0.4


class VectorSearchService:
    """
    Service for performing vector similarity search.

    This service provides a unified interface for searching across:
    - Code context (stored procedures, code snippets, configs)
    - Documentation (knowledge base, work instructions)

    The service uses MongoDB for storage and can leverage either:
    - MongoDB Atlas Vector Search ($vectorSearch aggregation)
    - In-memory numpy-based cosine similarity (fallback)

    Usage:
        service = VectorSearchService()
        await service.initialize()

        results = await service.search_code_context(
            query="RecapGet stored procedure",
            options=SearchOptions(limit=10, project="gin")
        )
    """

    _instance: Optional["VectorSearchService"] = None

    def __init__(self):
        """Initialize the service (use get_instance for singleton)."""
        self._mongodb_service = None
        self._embedding_service = None
        self._initialized = False

    @classmethod
    async def get_instance(cls) -> "VectorSearchService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """Initialize MongoDB and embedding service connections."""
        if self._initialized:
            return

        # Import here to avoid circular dependencies
        from mongodb import MongoDBService
        from embedding_service import get_embedding_service

        self._mongodb_service = MongoDBService.get_instance()
        if not self._mongodb_service.is_initialized:
            await self._mongodb_service.initialize()

        self._embedding_service = get_embedding_service()

        self._initialized = True
        logger.info("VectorSearchService initialized")

    async def search_code_context(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> List[VectorSearchResult]:
        """
        Search the code context collection.

        This searches across:
        - Stored procedures
        - Code snippets
        - Configuration files
        - SQL knowledge

        Args:
            query: Search query text
            options: Search options (limit, project, etc.)

        Returns:
            List of VectorSearchResult objects sorted by similarity
        """
        if not self._initialized:
            await self.initialize()

        options = options or SearchOptions()

        logger.info(
            f"Searching code context: query='{query[:50]}...', "
            f"project={options.project}, limit={options.limit}"
        )

        try:
            # Generate embedding for query
            query_vector = await self._embedding_service.get_embedding(query)

            # Search in code_context collection
            results = await self._mongodb_service.search_code_context(
                query_vector=query_vector,
                limit=options.limit,
                project=options.project,
                similarity_threshold=options.similarity_threshold
            )

            # If including EWRLibrary and searching a different project
            if (
                options.include_ewr_library
                and options.project
                and options.project.lower() != "ewrlibrary"
            ):
                library_results = await self._mongodb_service.search_code_context(
                    query_vector=query_vector,
                    limit=options.limit,
                    project="EWRLibrary",
                    similarity_threshold=options.similarity_threshold
                )
                results.extend(library_results)
                logger.info(f"Added {len(library_results)} EWRLibrary results")

            # Convert to VectorSearchResult objects
            formatted_results = [
                self._format_code_result(r) for r in results
            ]

            # Sort by similarity and take top N
            formatted_results.sort(
                key=lambda x: x.similarity or x.score or 0,
                reverse=True
            )
            formatted_results = formatted_results[:options.limit * 2]

            logger.info(f"Found {len(formatted_results)} code context results")
            return formatted_results

        except Exception as e:
            logger.error(f"Code context search failed: {e}", exc_info=True)
            return []

    async def search_documents(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> List[VectorSearchResult]:
        """
        Search the documents/knowledge base collection.

        This searches the EWR documentation, work instructions,
        and other uploaded documents.

        Args:
            query: Search query text
            options: Search options (limit, etc.)

        Returns:
            List of VectorSearchResult objects sorted by relevance
        """
        if not self._initialized:
            await self.initialize()

        options = options or SearchOptions()

        logger.info(
            f"Searching documents: query='{query[:50]}...', limit={options.limit}"
        )

        try:
            # Generate embedding for query
            query_vector = await self._embedding_service.get_embedding(query)

            # Search in documents collection
            results = await self._mongodb_service.search_documents(
                query_vector=query_vector,
                limit=options.limit
            )

            # Convert to VectorSearchResult objects
            formatted_results = [
                self._format_document_result(r) for r in results
            ]

            logger.info(f"Found {len(formatted_results)} document results")
            return formatted_results

        except Exception as e:
            logger.error(f"Document search failed: {e}", exc_info=True)
            return []

    async def search(
        self,
        query: str,
        search_type: str = "code",
        options: Optional[SearchOptions] = None
    ) -> List[VectorSearchResult]:
        """
        Unified search method that routes to appropriate backend.

        Args:
            query: Search query text
            search_type: "code" for code context, "documents" for knowledge base
            options: Search options

        Returns:
            List of VectorSearchResult objects
        """
        if search_type == "documents" or search_type == "knowledge_base":
            return await self.search_documents(query, options)
        else:
            return await self.search_code_context(query, options)

    def _format_code_result(self, result: Dict[str, Any]) -> VectorSearchResult:
        """
        Format a raw MongoDB code context result to VectorSearchResult.

        Handles the various field names used in the code_context collection.
        """
        metadata = result.get("metadata", {})

        return VectorSearchResult(
            content=result.get("content", ""),
            similarity=result.get("similarity", result.get("score", 0.0)),
            score=result.get("score"),
            project=metadata.get("project", result.get("project")),
            metadata=metadata,
            title=metadata.get("title"),
            file_name=metadata.get("file", result.get("file")),
            department=None,
            relevance_score=None
        )

    def _format_document_result(self, result: Dict[str, Any]) -> VectorSearchResult:
        """
        Format a raw MongoDB document result to VectorSearchResult.

        Handles the field names used in the documents collection.
        """
        return VectorSearchResult(
            content=result.get("content", ""),
            similarity=result.get("similarity", result.get("relevance_score", 0.0)),
            score=result.get("relevance_score"),
            project="knowledge_base",
            metadata={
                "department": result.get("department"),
                "type": result.get("type"),
                "title": result.get("title"),
            },
            title=result.get("title"),
            file_name=result.get("file_name"),
            department=result.get("department"),
            relevance_score=result.get("relevance_score")
        )


# Module-level singleton accessor
_vector_search_service: Optional[VectorSearchService] = None


async def get_vector_search_service() -> VectorSearchService:
    """Get or create the global vector search service instance."""
    global _vector_search_service
    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
        await _vector_search_service.initialize()
    return _vector_search_service
