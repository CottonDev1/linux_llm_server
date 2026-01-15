"""
Query Pipeline
==============

Main orchestration class for the RAG query pipeline.

This module provides the QueryPipeline class that orchestrates:
1. Query enhancement with conversation history
2. Vector search across code or documentation
3. Response caching for performance
4. LLM generation with retrieved context
5. SSE streaming for real-time responses

Architecture:
-------------
The pipeline follows a layered architecture:

    QueryPipeline (Orchestration)
         |
    +----+----+----+----+
    |    |    |    |    |
  Query  Vector  LLM  Response
  Enhancer Search Service Cache

Each service is a singleton with lazy initialization to minimize
startup time and resource usage.

Design Patterns:
- Dependency Injection: Services passed at construction
- Singleton: Each service maintains a single instance
- Strategy: Different search backends for code vs docs
- Template Method: Common pipeline structure with variations

Usage:
    pipeline = QueryPipeline()
    await pipeline.initialize()

    # Non-streaming query
    result = await pipeline.query(
        query="How does RecapGet work?",
        project="gin"
    )

    # Streaming query
    async for event in pipeline.query_stream(
        query="Explain the bale processing flow",
        project="gin"
    ):
        send_sse(event)
"""

import logging
import time
from typing import Optional, List, Dict, Any, AsyncGenerator

from query_pipeline.models.query_models import (
    QueryRequest,
    StreamQueryRequest,
    SearchRequest,
    SearchResponse,
    QueryResponse,
    SearchSource,
    TokenUsage,
    TimingInfo,
    ChatMessage,
    VectorSearchResult,
    QueryConfig,
    ProjectInfo,
    SourcesEvent,
    TokenEvent,
    DoneEvent,
    ErrorEvent,
)
from query_pipeline.services.vector_search import (
    VectorSearchService,
    SearchOptions,
    get_vector_search_service,
)
from query_pipeline.services.llm_generation import (
    LLMGenerationService,
    GenerationConfig,
    get_llm_generation_service,
)
from query_pipeline.services.response_cache import (
    ResponseCache,
    get_response_cache,
)
from query_pipeline.services.query_enhancer import (
    QueryEnhancer,
    get_query_enhancer,
)

logger = logging.getLogger(__name__)


# Available projects for queries
AVAILABLE_PROJECTS: List[ProjectInfo] = [
    ProjectInfo(id="all", name="All Projects", description="Search across all projects"),
    ProjectInfo(id="gin", name="Gin", description="Cotton Gin application"),
    ProjectInfo(id="EWRLibrary", name="EWR Library", description="EWR shared library"),
    ProjectInfo(id="warehouse", name="Warehouse", description="Warehouse management"),
    ProjectInfo(id="marketing", name="Marketing", description="Marketing application"),
    ProjectInfo(id="knowledge_base", name="Knowledge Base", description="EWR Documentation"),
]


class QueryPipeline:
    """
    Main orchestration class for RAG queries.

    This pipeline handles the full RAG workflow:
    1. Receive natural language query
    2. Enhance query with conversation context (for follow-ups)
    3. Search for relevant documents (code or documentation)
    4. Check response cache
    5. Generate answer using LLM with context
    6. Cache response for future queries

    The pipeline supports two modes:
    - Blocking: Returns complete response when finished
    - Streaming: Yields SSE events as processing progresses

    Attributes:
        config: Query processing configuration
        _vector_search: Vector search service
        _llm_generation: LLM generation service
        _response_cache: Response cache
        _query_enhancer: Query enhancement service
        _initialized: Whether services are initialized
    """

    def __init__(
        self,
        config: Optional[QueryConfig] = None,
        vector_search: Optional[VectorSearchService] = None,
        llm_generation: Optional[LLMGenerationService] = None,
        response_cache: Optional[ResponseCache] = None,
        query_enhancer: Optional[QueryEnhancer] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Query configuration (uses defaults if None)
            vector_search: Vector search service (created if None)
            llm_generation: LLM generation service (created if None)
            response_cache: Response cache (created if None)
            query_enhancer: Query enhancer (created if None)
        """
        self.config = config or QueryConfig()
        self._vector_search = vector_search
        self._llm_generation = llm_generation
        self._response_cache = response_cache or get_response_cache()
        self._query_enhancer = query_enhancer or get_query_enhancer()
        self._initialized = False

    async def initialize(self):
        """Initialize all service dependencies."""
        if self._initialized:
            return

        if self._vector_search is None:
            self._vector_search = await get_vector_search_service()

        if self._llm_generation is None:
            self._llm_generation = await get_llm_generation_service()

        self._initialized = True
        logger.info("QueryPipeline initialized")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform direct vector search without LLM processing.

        This endpoint returns raw search results, useful for:
        - MCP server integration
        - Search-only use cases
        - Debugging retrieval quality

        Args:
            request: Search request with query, project, limit

        Returns:
            SearchResponse with sources and count
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        logger.info(
            f"Direct search: query='{request.query[:50]}...', "
            f"project={request.project}, limit={request.limit}"
        )

        # Search via vector service
        options = SearchOptions(
            limit=request.limit,
            project=request.project
        )

        results = await self._vector_search.search(
            query=request.query,
            search_type="code",
            options=options
        )

        # Format sources
        sources = [self._format_source(r) for r in results]

        duration = time.time() - start_time
        logger.info(f"Search completed: {len(sources)} results in {duration:.2f}s")

        return SearchResponse(
            sources=sources,
            total_results=len(sources)
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a RAG query with vector search and LLM generation.

        This is the main query endpoint that:
        1. Enhances query with conversation context
        2. Checks response cache
        3. Searches for relevant documents
        4. Generates answer using LLM
        5. Caches response

        Args:
            request: Query request with all parameters

        Returns:
            QueryResponse with answer, sources, and metadata
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        is_knowledge_base = request.project == "knowledge_base"

        logger.info(
            f"Query: '{request.query[:50]}...', "
            f"project={request.project}, "
            f"history_length={len(request.history)}"
        )

        # Check cache first
        cache_key = {
            "query": request.query,
            "project": request.project,
            "limit": request.limit,
            "includeEWRLibrary": request.include_ewr_library
        }

        cached = await self._response_cache.get(request.query, cache_key)
        if cached:
            logger.info("Returning cached response")
            return QueryResponse(
                answer=cached.get("answer", ""),
                sources=[SearchSource(**s) for s in cached.get("sources", [])],
                query=request.query,
                model=cached.get("model", "unknown"),
                search_strategy="mongodb-vector",
                cached=True,
                timing=TimingInfo(
                    search=0,
                    llm=0,
                    total=time.time() - start_time
                ),
                token_usage=TokenUsage(**cached.get("token_usage", {})) if cached.get("token_usage") else None
            )

        # Enhance query for better search
        history = request.history or []
        enhancement = self._query_enhancer.enhance(
            query=request.query,
            history=history
        )

        search_query = enhancement.enhanced_query

        # Perform vector search
        search_start = time.time()
        options = SearchOptions(
            limit=request.limit * 2,  # Get extra for filtering
            project=request.project if not is_knowledge_base else None,
            include_ewr_library=request.include_ewr_library
        )

        if is_knowledge_base:
            results = await self._vector_search.search_documents(
                query=search_query,
                options=options
            )
        else:
            results = await self._vector_search.search_code_context(
                query=search_query,
                options=options
            )

        search_duration = time.time() - search_start

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information for your question. Please try rephrasing or contact support.",
                sources=[],
                query=request.query,
                model=request.model or "unknown",
                search_strategy="mongodb-vector",
                cached=False,
                timing=TimingInfo(
                    search=search_duration,
                    llm=0,
                    total=time.time() - start_time
                )
            )

        # Take top results
        results = results[:request.limit]

        # Generate LLM response
        llm_start = time.time()
        gen_result = await self._llm_generation.generate(
            query=request.query,
            context=results,
            history=history,
            is_knowledge_base=is_knowledge_base,
            config=GenerationConfig(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
        )
        llm_duration = time.time() - llm_start

        if not gen_result.success:
            return QueryResponse(
                answer=f"Failed to generate response: {gen_result.error}",
                sources=[self._format_source(r) for r in results],
                query=request.query,
                model=request.model or "unknown",
                search_strategy="mongodb-vector",
                cached=False,
                timing=TimingInfo(
                    search=search_duration,
                    llm=llm_duration,
                    total=time.time() - start_time
                )
            )

        total_duration = time.time() - start_time

        # Build response
        response = QueryResponse(
            answer=gen_result.text,
            sources=[self._format_source(r) for r in results],
            query=request.query,
            model=request.model or self.config.__class__.__name__,
            search_strategy="mongodb-vector",
            cached=False,
            timing=TimingInfo(
                search=search_duration,
                llm=llm_duration,
                total=total_duration
            ),
            token_usage=gen_result.token_usage
        )

        # Cache the response
        await self._response_cache.set(
            query=request.query,
            cache_params=cache_key,
            response={
                "answer": response.answer,
                "sources": [s.model_dump() for s in response.sources],
                "model": response.model,
                "token_usage": response.token_usage.model_dump() if response.token_usage else None
            },
            ttl=self.config.cache_ttl_seconds
        )

        logger.info(
            f"Query completed: {len(results)} sources, "
            f"{len(gen_result.text)} chars, {total_duration:.2f}s"
        )

        return response

    async def query_stream(
        self,
        request: StreamQueryRequest
    ) -> AsyncGenerator[str, None]:
        """
        Process a streaming RAG query with SSE events.

        Yields SSE-formatted events:
        - sources: Retrieved sources (sent first)
        - token: Each generated token
        - done: Completion with stats
        - error: On failure

        Args:
            request: Stream query request

        Yields:
            SSE-formatted event strings
        """
        if not self._initialized:
            await self.initialize()

        is_knowledge_base = request.project == "knowledge_base"

        logger.info(
            f"Stream query: '{request.query[:50]}...', project={request.project}"
        )

        try:
            # Perform vector search
            options = SearchOptions(
                limit=request.limit,
                project=request.project if not is_knowledge_base else None
            )

            if is_knowledge_base:
                results = await self._vector_search.search_documents(
                    query=request.query,
                    options=options
                )
            else:
                results = await self._vector_search.search_code_context(
                    query=request.query,
                    options=options
                )

            # Send sources first
            sources_event = SourcesEvent(
                sources=[
                    {
                        "project": r.project or r.metadata.get("project"),
                        "file": r.file_name or r.metadata.get("file"),
                        "similarity": r.similarity or r.score
                    }
                    for r in results
                ]
            )
            yield sources_event.to_sse()

            # Stream LLM response
            async for chunk in self._llm_generation.generate_stream(
                query=request.query,
                context=results,
                history=None,  # Streaming doesn't support history currently
                is_knowledge_base=is_knowledge_base,
                config=GenerationConfig(
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            ):
                if chunk["type"] == "token":
                    yield TokenEvent(token=chunk["token"]).to_sse()
                elif chunk["type"] == "done":
                    yield DoneEvent(
                        token_usage=TokenUsage(
                            prompt_tokens=chunk.get("token_usage", {}).get("prompt_tokens", 0),
                            response_tokens=chunk.get("token_usage", {}).get("response_tokens", 0),
                            total_tokens=chunk.get("token_usage", {}).get("total_tokens", 0)
                        )
                    ).to_sse()
                elif chunk["type"] == "error":
                    yield ErrorEvent(error=chunk["error"]).to_sse()

        except Exception as e:
            logger.error(f"Stream query failed: {e}", exc_info=True)
            yield ErrorEvent(error=str(e)).to_sse()

    def get_projects(self) -> List[ProjectInfo]:
        """
        Get list of available projects for queries.

        Returns:
            List of ProjectInfo objects
        """
        return AVAILABLE_PROJECTS

    def _format_source(self, result: VectorSearchResult) -> SearchSource:
        """
        Format a VectorSearchResult to SearchSource for API response.

        Args:
            result: Raw search result

        Returns:
            Formatted SearchSource
        """
        return SearchSource(
            snippet=result.content[:200] + "..." if len(result.content) > 200 else result.content,
            project=result.project or "unknown",
            category=result.metadata.get("category", "unknown"),
            file=result.file_name or result.metadata.get("file", "unknown"),
            similarity=result.similarity or result.score or 0,
            department=result.department,
            type=result.metadata.get("type"),
            title=result.title,
            relevance=result.relevance_score
        )

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get response cache statistics."""
        return self._response_cache.get_stats()

    async def clear_cache(self) -> int:
        """Clear the response cache. Returns number of entries cleared."""
        return await self._response_cache.clear()

    async def close(self):
        """Close service connections."""
        logger.info("QueryPipeline closed")


# Module-level singleton
_query_pipeline: Optional[QueryPipeline] = None


async def get_query_pipeline(config: Optional[QueryConfig] = None) -> QueryPipeline:
    """Get or create the global query pipeline instance."""
    global _query_pipeline
    if _query_pipeline is None:
        _query_pipeline = QueryPipeline(config)
        await _query_pipeline.initialize()
    return _query_pipeline
