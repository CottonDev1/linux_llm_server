"""
Query/RAG Pipeline Tests
========================

Comprehensive E2E tests for the Query/RAG Pipeline targeting 80%+ coverage.

Test Modules:
-------------

1. test_query_e2e.py - End-to-end pipeline tests
   - Full RAG pipeline flow (query -> search -> context -> LLM -> response)
   - Knowledge base queries
   - Code queries
   - Project filtering
   - No results handling
   - Pipeline performance and latency

2. test_vector_search_integration.py - Real MongoDB vector search tests
   - Actual vector search with MongoDB ($vectorSearch or cosine similarity)
   - Similarity threshold behavior
   - Metadata filtering with vector search
   - Result ranking by similarity score
   - Empty result handling
   - Search performance validation

3. test_hybrid_retriever.py - Hybrid search tests (BM25 + Vector)
   - BM25 index building and keyword search
   - Vector search integration
   - Reciprocal Rank Fusion (RRF) combination
   - Weight balancing (vector_weight, bm25_weight)
   - Fallback behavior when one method fails

4. test_document_reranker.py - Cross-encoder reranking tests
   - Document reranking simulation
   - Score distribution validation
   - Metadata preservation during reranking
   - Top-K selection after reranking
   - Edge cases (empty, single doc)

5. test_source_attribution.py - Source citation tests
   - Source citation in responses
   - Source metadata preservation
   - Multi-source synthesis attribution
   - Source relevance scoring
   - Citation format validation
   - Source deduplication

6. test_context_window.py - Context management tests
   - Context truncation when exceeding limits
   - Chunk assembly for context
   - Surrounding chunk retrieval
   - Token counting and budget allocation
   - Overlap handling

7. test_query_api_endpoints.py - API endpoint tests
   - POST /api/query endpoint
   - POST /api/query/stream SSE endpoint
   - POST /api/search endpoint
   - Response format validation
   - Error handling
   - Request validation

8. test_query_storage.py - Document and embedding storage tests
   - Document chunk storage
   - Code context storage
   - Embedding storage and validation
   - Schema validation

9. test_query_retrieval.py - Retrieval tests
   - Document retrieval
   - Code context retrieval
   - Filtering and sorting

10. test_query_generation.py - LLM generation tests
    - RAG answer generation
    - Code explanation generation
    - Multi-source synthesis
    - Context relevance filtering

Running Tests:
--------------
    # Run all query pipeline tests
    cd testing
    python -m pytest pipelines/query/ -v

    # Run specific test file
    python -m pytest pipelines/query/test_vector_search_integration.py -v

    # Run with markers
    python -m pytest pipelines/query/ -v -m "e2e and requires_mongodb"

    # Run excluding slow tests
    python -m pytest pipelines/query/ -v -m "not slow"

Markers:
--------
    @pytest.mark.e2e - End-to-end test
    @pytest.mark.requires_mongodb - Requires MongoDB connection
    @pytest.mark.requires_llm - Requires local LLM endpoint
    @pytest.mark.slow - Slow-running test

Coverage Targets:
-----------------
    - Vector search: 90%+
    - Hybrid retrieval: 85%+
    - Reranking: 80%+
    - Source attribution: 80%+
    - Context management: 85%+
    - API endpoints: 80%+
"""
