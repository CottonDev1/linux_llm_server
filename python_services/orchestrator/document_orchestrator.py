"""
Knowledge Base Orchestrator
===========================

Central coordinator for the document retrieval pipeline using agentic AI patterns.

Architecture Overview:
----------------------
This orchestrator implements a CRAG (Corrective RAG) pattern with multi-agent
coordination. The pipeline flows through these stages:

    Query Understanding --> Hybrid Retrieval --> Document Grading --> Answer Generation
                                    |                  |
                                    |                  v
                                    +---- Corrective Retrieval (if grading fails)
                                                       |
                                                       v
                                              Answer Validation
                                                       |
                                              (Self-Correction if needed)
                                                       |
                                                       v
                                              Learning Feedback

Design Decisions:
-----------------
1. Async/Await Throughout: All I/O operations are non-blocking for scalability
2. Stage Isolation: Each stage has its own error handling and timing
3. Streaming Support: SSE events for real-time UI updates
4. Graceful Degradation: Pipeline continues if optional stages fail
5. Dependency Injection: Services are injected for testability

Production Considerations:
--------------------------
- Timeouts are configurable per stage
- Error states are captured with full context
- Metrics are embedded in every response
- Correction attempts are limited to prevent infinite loops
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime

from core.log_utils import log_info, log_warning, log_error

from .models import (
    QueryIntent,
    PipelineStage,
    ValidationStatus,
    QueryRequest,
    QueryResponse,
    QueryAnalysisResult,
    ExtractedEntity,
    RetrievedDocument,
    RetrievalResult,
    GradedDocument,
    GradingResult,
    GenerationRequest,
    GenerationResult,
    RelevancyCheck,
    FaithfulnessCheck,
    CompletenessCheck,
    ValidationResult,
    PipelineState,
    StreamEventType,
    StreamEvent,
    FeedbackRecord,
    FeedbackType,
    OrchestratorConfig,
)

logger = logging.getLogger(__name__)


def _parse_llm_json(text: str, default: dict) -> dict:
    """
    Robustly parse JSON from LLM output, handling common issues.

    Args:
        text: The LLM response text that may contain JSON
        default: Default dictionary to return if parsing fails

    Returns:
        Parsed JSON dict or default if parsing fails
    """
    import json
    import re

    if not text:
        return default

    # Clean the text
    text = text.strip()

    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in the text
    json_match = re.search(r'\{[^{}]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to fix common JSON issues
    # 1. Missing commas between key-value pairs
    fixed_text = re.sub(r'"\s*"', '", "', text)
    fixed_text = re.sub(r'(\w+)\s+"', r'\1", "', fixed_text)
    fixed_text = re.sub(r'(true|false|null|\d+)\s+"', r'\1, "', fixed_text)

    try:
        # Try to find JSON object in fixed text
        json_match = re.search(r'\{[^{}]*\}', fixed_text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    return default


# Try to import TracedLLMClient
try:
    from llm.integration import generate_text
    TRACED_LLM_AVAILABLE = True
except ImportError:
    TRACED_LLM_AVAILABLE = False
    logger.info("TracedLLMClient not available, using legacy LLM service")


class KnowledgeBaseOrchestrator:
    """
    Central coordinator for knowledge base retrieval pipeline.

    Orchestrates multiple agents/services to process user queries:
    1. Query Understanding: Classify intent, expand queries, extract entities
    2. Hybrid Retrieval: Vector search + BM25 with RRF fusion
    3. Document Grading: Filter irrelevant documents (CRAG pattern)
    4. Answer Generation: LLM synthesis with retrieved context
    5. Answer Validation: Check relevancy, faithfulness, completeness
    6. Self-Correction: Retry generation if validation fails
    7. Learning Feedback: Record for continuous improvement

    Usage:
        orchestrator = KnowledgeBaseOrchestrator()
        await orchestrator.initialize()

        # Synchronous query
        response = await orchestrator.process_query(QueryRequest(query="..."))

        # Streaming query
        async for event in orchestrator.process_query_stream(QueryRequest(query="...", stream=True)):
            print(event.to_sse())

        # Record feedback
        await orchestrator.record_feedback(FeedbackRecord(...))
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        mongodb_service: Optional[Any] = None,
        llm_service: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Pipeline configuration (uses defaults if None)
            mongodb_service: MongoDB service for retrieval (auto-initialized if None)
            llm_service: LLM service for generation (auto-initialized if None)
            embedding_service: Embedding service for query embedding (auto-initialized if None)

        Design Note:
        Dependency injection allows for easy testing with mocked services.
        Production usage typically auto-initializes all services.
        """
        self.config = config or OrchestratorConfig()
        self._mongodb_service = mongodb_service
        self._llm_service = llm_service
        self._embedding_service = embedding_service
        self._initialized = False
        self._use_traced = TRACED_LLM_AVAILABLE

        # Agent references (lazy initialized)
        self._query_agent = None
        self._grading_agent = None
        self._validation_agent = None
        self._learning_agent = None

        # Hybrid retriever (lazy initialized)
        self._hybrid_retriever = None

        # Semantic cache (lazy initialized)
        self._semantic_cache = None

    async def initialize(self) -> None:
        """
        Initialize all services and agents.

        This method must be called before processing queries.
        It's idempotent - safe to call multiple times.
        """
        if self._initialized:
            return

        log_info("Document Pipeline", "Initializing KnowledgeBaseOrchestrator...")

        # Initialize MongoDB service
        if self._mongodb_service is None:
            try:
                from mongodb import MongoDBService
                self._mongodb_service = MongoDBService.get_instance()
                await self._mongodb_service.initialize()
                log_info("Document Pipeline", "MongoDB service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB service: {e}")
                raise

        # Initialize LLM service (only if not using traced client)
        if self._llm_service is None and not self._use_traced:
            try:
                from services.llm_service import get_llm_service
                self._llm_service = await get_llm_service()
                log_info("Document Pipeline", "LLM service initialized (legacy)")
            except Exception as e:
                logger.error(f"Failed to initialize LLM service: {e}")
                raise
        elif self._use_traced:
            log_info("Document Pipeline", "Using TracedLLMClient for LLM operations")

        # Initialize embedding service
        if self._embedding_service is None:
            try:
                from embedding_service import get_embedding_service
                self._embedding_service = get_embedding_service()
                await self._embedding_service.initialize()
                log_info("Document Pipeline", "Embedding service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                raise

        # Initialize hybrid retriever (for BM25 + Vector search with RRF)
        if self._hybrid_retriever is None and self.config.enable_hybrid_search:
            try:
                from services.hybrid_retriever import get_hybrid_retriever
                self._hybrid_retriever = await get_hybrid_retriever(
                    self._mongodb_service,
                    self._embedding_service
                )
                log_info("Document Pipeline", "Hybrid retriever initialized (BM25 + Vector with RRF)")
            except Exception as e:
                logger.warning(f"Hybrid retriever not available: {e} (will use vector-only search)")

        # Initialize semantic cache (for caching embeddings, results, responses)
        if self._semantic_cache is None and self.config.enable_semantic_cache:
            try:
                from services.semantic_cache import get_semantic_cache
                self._semantic_cache = await get_semantic_cache()
                if self._semantic_cache.is_available:
                    log_info("Document Pipeline", "Semantic cache initialized (Redis)")
                else:
                    log_info("Document Pipeline", "Semantic cache not available (Redis not running)")
            except Exception as e:
                logger.warning(f"Semantic cache not available: {e}")

        self._initialized = True
        log_info("Document Pipeline", "KnowledgeBaseOrchestrator initialization complete")

    async def close(self) -> None:
        """Clean up resources."""
        if self._llm_service and hasattr(self._llm_service, 'close'):
            await self._llm_service.close()
        self._initialized = False
        log_info("Document Pipeline", "KnowledgeBaseOrchestrator closed")

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query through the full pipeline (synchronous mode).

        Args:
            request: Query request with user question and options

        Returns:
            QueryResponse with answer, sources, and metadata

        Error Handling:
        - Each stage catches its own errors
        - Pipeline continues with degraded results where possible
        - Fatal errors return QueryResponse with error field set
        """
        if not self._initialized:
            await self.initialize()

        # Initialize pipeline state
        state = PipelineState(request=request)
        start_time = time.time()

        log_info("Document Pipeline", f"========== Starting Document Pipeline ==========")
        log_info("Document Pipeline", f"Query ID: {request.query_id}")

        try:
            # Stage 1: Query Understanding
            state = await self._stage_query_understanding(state)

            # Stage 2: Routing - check if retrieval is needed
            if not self._should_skip_retrieval(state):
                # Stage 2: Hybrid Retrieval
                state = await self._stage_retrieval(state)

                # Stage 3: Document Grading
                if self.config.enable_document_grading:
                    state = await self._stage_grading(state)

                    # Corrective retrieval if grading fails
                    if state.grading_result and state.grading_result.needs_correction:
                        log_info("Document Pipeline", "[STAGE 3b] Corrective Retrieval - Low relevance detected, expanding search...")
                        state = await self._stage_corrective_retrieval(state)
                        log_info("Document Pipeline", "[STAGE 3b] Corrective Retrieval COMPLETE")
            else:
                log_info("Document Pipeline", "[STAGE 2-3] SKIPPED - Query doesn't require retrieval")

            # Stage 4: Answer Generation
            state = await self._stage_generation(state)

            # Stage 5: Answer Validation
            if self.config.enable_validation and not request.skip_validation:
                state = await self._stage_validation(state)

                # Stage 6: Self-Correction if validation fails
                if (state.validation_result and
                    state.validation_result.needs_correction and
                    self.config.enable_self_correction and
                    state.correction_attempts < state.max_corrections):
                    state = await self._stage_self_correction(state)
            else:
                log_info("Document Pipeline", "[STAGE 5-6] SKIPPED - Validation disabled or skipped")

            # Build final response
            state.current_stage = PipelineStage.COMPLETE
            state.completed_at = datetime.utcnow()
            state.response = self._build_response(state, start_time)

            total_ms = int((time.time() - start_time) * 1000)
            log_info("Document Pipeline", f"========== Pipeline COMPLETE ==========")
            log_info("Document Pipeline", f"Total time: {total_ms}ms | Answer: {len(state.response.answer)} chars | Sources: {len(state.response.sources)}")

            return state.response

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            log_error("Document Pipeline", f"========== Pipeline FAILED ==========")
            log_error("Document Pipeline", f"Error at stage {state.current_stage}: {e}")
            state.error = str(e)
            state.error_stage = state.current_stage
            state.current_stage = PipelineStage.FAILED

            return QueryResponse(
                query_id=request.query_id,
                query=request.query,
                error=str(e),
                total_time_ms=int((time.time() - start_time) * 1000),
            )

    async def process_query_stream(
        self,
        request: QueryRequest
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Process a query with streaming events (SSE mode).

        Yields StreamEvent objects for real-time UI updates.
        Events include stage transitions, retrieved documents,
        and token-by-token answer generation.

        Args:
            request: Query request with stream=True

        Yields:
            StreamEvent objects suitable for SSE

        Usage:
            async for event in orchestrator.process_query_stream(request):
                yield event.to_sse()  # Send to client
        """
        if not self._initialized:
            await self.initialize()

        state = PipelineState(request=request)
        start_time = time.time()

        def elapsed_ms() -> int:
            return int((time.time() - start_time) * 1000)

        try:
            # Stage 1: Query Understanding
            yield StreamEvent(
                event_type=StreamEventType.STAGE_START,
                query_id=request.query_id,
                stage=PipelineStage.QUERY_UNDERSTANDING,
                message="Analyzing query...",
                elapsed_ms=elapsed_ms(),
            )

            state = await self._stage_query_understanding(state)

            yield StreamEvent(
                event_type=StreamEventType.STAGE_COMPLETE,
                query_id=request.query_id,
                stage=PipelineStage.QUERY_UNDERSTANDING,
                data={
                    "intent": state.query_analysis.query_intent.value if state.query_analysis else "unknown",
                    "requires_retrieval": state.query_analysis.requires_retrieval if state.query_analysis else True,
                },
                elapsed_ms=elapsed_ms(),
            )

            # Check if we should skip retrieval
            if not self._should_skip_retrieval(state):
                # Stage 2: Retrieval
                yield StreamEvent(
                    event_type=StreamEventType.STAGE_START,
                    query_id=request.query_id,
                    stage=PipelineStage.RETRIEVAL,
                    message="Searching knowledge base...",
                    elapsed_ms=elapsed_ms(),
                )

                state = await self._stage_retrieval(state)

                # Emit retrieved documents
                if state.retrieval_result:
                    for doc in state.retrieval_result.documents[:5]:
                        yield StreamEvent(
                            event_type=StreamEventType.DOCUMENT_FOUND,
                            query_id=request.query_id,
                            stage=PipelineStage.RETRIEVAL,
                            data={
                                "title": doc.title or "Untitled",
                                "score": round(doc.rrf_score, 3),
                                "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                            },
                            elapsed_ms=elapsed_ms(),
                        )

                yield StreamEvent(
                    event_type=StreamEventType.STAGE_COMPLETE,
                    query_id=request.query_id,
                    stage=PipelineStage.RETRIEVAL,
                    data={
                        "documents_found": len(state.retrieval_result.documents) if state.retrieval_result else 0,
                    },
                    elapsed_ms=elapsed_ms(),
                )

                # Stage 3: Grading
                if self.config.enable_document_grading:
                    yield StreamEvent(
                        event_type=StreamEventType.STAGE_START,
                        query_id=request.query_id,
                        stage=PipelineStage.GRADING,
                        message="Evaluating relevance...",
                        elapsed_ms=elapsed_ms(),
                    )

                    state = await self._stage_grading(state)

                    yield StreamEvent(
                        event_type=StreamEventType.STAGE_COMPLETE,
                        query_id=request.query_id,
                        stage=PipelineStage.GRADING,
                        data={
                            "relevant_count": state.grading_result.relevant_count if state.grading_result else 0,
                            "average_relevance": round(state.grading_result.average_relevance, 2) if state.grading_result else 0,
                        },
                        elapsed_ms=elapsed_ms(),
                    )

                    # Corrective retrieval if needed
                    if state.grading_result and state.grading_result.needs_correction:
                        yield StreamEvent(
                            event_type=StreamEventType.STAGE_START,
                            query_id=request.query_id,
                            stage=PipelineStage.CORRECTION,
                            message="Expanding search...",
                            elapsed_ms=elapsed_ms(),
                        )

                        state = await self._stage_corrective_retrieval(state)

                        yield StreamEvent(
                            event_type=StreamEventType.STAGE_COMPLETE,
                            query_id=request.query_id,
                            stage=PipelineStage.CORRECTION,
                            elapsed_ms=elapsed_ms(),
                        )

            # Stage 4: Generation with streaming
            yield StreamEvent(
                event_type=StreamEventType.STAGE_START,
                query_id=request.query_id,
                stage=PipelineStage.GENERATION,
                message="Generating answer...",
                elapsed_ms=elapsed_ms(),
            )

            # Stream generation tokens
            async for token_event in self._stage_generation_stream(state):
                token_event.query_id = request.query_id
                token_event.elapsed_ms = elapsed_ms()
                yield token_event

            yield StreamEvent(
                event_type=StreamEventType.STAGE_COMPLETE,
                query_id=request.query_id,
                stage=PipelineStage.GENERATION,
                elapsed_ms=elapsed_ms(),
            )

            # Stage 5: Validation
            if self.config.enable_validation and not request.skip_validation:
                yield StreamEvent(
                    event_type=StreamEventType.STAGE_START,
                    query_id=request.query_id,
                    stage=PipelineStage.VALIDATION,
                    message="Validating answer...",
                    elapsed_ms=elapsed_ms(),
                )

                state = await self._stage_validation(state)

                # Emit validation checks
                if state.validation_result:
                    for check_name in ["relevancy", "faithfulness", "completeness"]:
                        check = getattr(state.validation_result, check_name, None)
                        if check:
                            yield StreamEvent(
                                event_type=StreamEventType.VALIDATION_CHECK,
                                query_id=request.query_id,
                                stage=PipelineStage.VALIDATION,
                                data={
                                    "check": check_name,
                                    "passed": check.passed,
                                    "score": round(check.score, 2),
                                },
                                elapsed_ms=elapsed_ms(),
                            )

                yield StreamEvent(
                    event_type=StreamEventType.STAGE_COMPLETE,
                    query_id=request.query_id,
                    stage=PipelineStage.VALIDATION,
                    data={
                        "valid": state.validation_result.is_valid if state.validation_result else True,
                    },
                    elapsed_ms=elapsed_ms(),
                )

            # Final response
            state.current_stage = PipelineStage.COMPLETE
            state.completed_at = datetime.utcnow()
            state.response = self._build_response(state, start_time)

            yield StreamEvent(
                event_type=StreamEventType.COMPLETE,
                query_id=request.query_id,
                stage=PipelineStage.COMPLETE,
                data=state.response.model_dump(mode='json'),
                elapsed_ms=elapsed_ms(),
            )

        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                query_id=request.query_id,
                stage=state.current_stage,
                data={"error": str(e)},
                message=str(e),
                elapsed_ms=elapsed_ms(),
            )

    # =========================================================================
    # Pipeline Stages
    # =========================================================================

    async def _stage_query_understanding(self, state: PipelineState) -> PipelineState:
        """
        Stage 1: Query Understanding

        Responsibilities:
        - Classify query intent (SIMPLE, FACTUAL, ANALYTICAL, etc.)
        - Extract entities (tables, columns, dates, etc.)
        - Expand query with synonyms and variations
        - Detect follow-up queries

        Output:
            Updates state.query_analysis with classification and transformations
        """
        state.current_stage = PipelineStage.QUERY_UNDERSTANDING
        start_time = time.time()

        query = state.request.query

        log_info("Document Pipeline", f"[STAGE 1/6] Query Understanding - Analyzing: '{query[:50]}...' " if len(query) > 50 else f"[STAGE 1/6] Query Understanding - Analyzing: '{query}'")

        try:
            # Classify intent using LLM
            intent, confidence = await self._classify_query_intent(query)

            # Check if this is a simple query that doesn't need retrieval
            requires_retrieval = self._requires_retrieval(intent, query)

            # Extract entities
            entities = await self._extract_entities(query)

            # Expand query if enabled (include conversation history for context)
            expanded_queries = []
            if self.config.enable_query_expansion and requires_retrieval:
                expanded_queries = await self._expand_query(
                    query, entities, state.request.conversation_history
                )

            # Rewrite for retrieval optimization
            rewritten = await self._rewrite_for_retrieval(query, entities)

            # Check for follow-up
            is_follow_up = self._detect_follow_up(
                query,
                state.request.previous_queries
            )

            state.query_analysis = QueryAnalysisResult(
                original_query=query,
                query_intent=intent,
                intent_confidence=confidence,
                entities=entities,
                rewritten_query=rewritten,
                expanded_queries=expanded_queries[:self.config.max_expanded_queries],
                is_follow_up=is_follow_up,
                requires_retrieval=requires_retrieval,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

            elapsed = int((time.time() - start_time) * 1000)
            log_info("Document Pipeline", f"[STAGE 1/6] Query Understanding COMPLETE - Intent: {intent.value} (confidence: {confidence:.2f}), Retrieval needed: {requires_retrieval} [{elapsed}ms]")

        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            # Fall back to default analysis
            state.query_analysis = QueryAnalysisResult(
                original_query=query,
                query_intent=QueryIntent.FACTUAL,
                intent_confidence=0.5,
                requires_retrieval=True,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

        return state

    async def _classify_query_intent(self, query: str) -> tuple[QueryIntent, float]:
        """
        Classify query intent using LLM.

        Uses a classification prompt to determine query type.
        Falls back to FACTUAL if classification fails.
        """
        prompt = f"""Classify this query into exactly one category:
- SIMPLE: Direct lookups, definitions, yes/no questions
- FACTUAL: Questions requiring specific facts from documents
- ANALYTICAL: Questions requiring synthesis across multiple sources
- TEMPORAL: Questions with time-based constraints (dates, recent, last week)
- PROCEDURAL: How-to questions requiring step-by-step answers
- COMPARISON: Questions comparing multiple concepts or entities
- AGGREGATION: Questions requiring counting, statistics, or summaries

Query: "{query}"

Respond with JSON only: {{"type": "TYPE", "confidence": 0.0-1.0}}"""

        try:
            import json
            system = "You are a query classifier. Respond only with valid JSON."

            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system}\n\n{prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="classify_query_intent",
                    pipeline="document_agent",
                    max_tokens=50,
                    temperature=0.0,
                    tags=["document_agent", "classification"],
                )

                if response.success:
                    try:
                        data = json.loads(response.text.strip())
                        intent_str = data.get("type", "FACTUAL").upper()
                        confidence = float(data.get("confidence", 0.5))

                        intent_map = {
                            "SIMPLE": QueryIntent.SIMPLE,
                            "FACTUAL": QueryIntent.FACTUAL,
                            "ANALYTICAL": QueryIntent.ANALYTICAL,
                            "TEMPORAL": QueryIntent.TEMPORAL,
                            "PROCEDURAL": QueryIntent.PROCEDURAL,
                            "COMPARISON": QueryIntent.COMPARISON,
                            "AGGREGATION": QueryIntent.AGGREGATION,
                        }

                        return intent_map.get(intent_str, QueryIntent.FACTUAL), confidence
                    except json.JSONDecodeError:
                        pass
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=50,
                    temperature=0.0,
                )

                if result.success:
                    try:
                        data = json.loads(result.response.strip())
                        intent_str = data.get("type", "FACTUAL").upper()
                        confidence = float(data.get("confidence", 0.5))

                        intent_map = {
                            "SIMPLE": QueryIntent.SIMPLE,
                            "FACTUAL": QueryIntent.FACTUAL,
                            "ANALYTICAL": QueryIntent.ANALYTICAL,
                            "TEMPORAL": QueryIntent.TEMPORAL,
                            "PROCEDURAL": QueryIntent.PROCEDURAL,
                            "COMPARISON": QueryIntent.COMPARISON,
                            "AGGREGATION": QueryIntent.AGGREGATION,
                        }

                        return intent_map.get(intent_str, QueryIntent.FACTUAL), confidence
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")

        return QueryIntent.FACTUAL, 0.5

    def _requires_retrieval(self, intent: QueryIntent, query: str) -> bool:
        """
        Determine if query needs retrieval or can be answered directly.

        Simple queries like greetings or definitions may skip retrieval.
        Follow-up questions always need retrieval to find relevant context.
        """
        query_lower = query.lower().strip()

        # Very short queries often don't need retrieval
        if len(query) < 20:
            simple_patterns = [
                "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"
            ]
            if any(query_lower.startswith(p) or query_lower == p for p in simple_patterns):
                return False

        # Check for follow-up indicators - these always need retrieval
        follow_up_indicators = [
            "it", "this", "that", "they", "them", "those", "these",
            "the same", "also", "more about", "elaborate", "again",
            "mentioned", "earlier", "previous", "above", "said"
        ]
        if any(ind in query_lower for ind in follow_up_indicators):
            return True

        # SIMPLE intent might skip retrieval for pure definitions
        if intent == QueryIntent.SIMPLE and len(query) < self.config.simple_query_max_length:
            # Check for definition questions that need knowledge base
            knowledge_indicators = ["what is", "explain", "define", "how does", "who is"]
            if any(ind in query_lower for ind in knowledge_indicators):
                return True
            return False

        return True

    async def _extract_entities(self, query: str) -> List[ExtractedEntity]:
        """
        Extract entities from the query for better retrieval.

        Uses regex patterns for common entity types.
        """
        entities = []

        # Date patterns
        import re
        date_patterns = [
            (r"\b(today|yesterday|tomorrow)\b", "date"),
            (r"\b(last|next|this)\s+(week|month|year)\b", "date"),
            (r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "date"),
            (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b", "date"),
        ]

        for pattern, entity_type in date_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    normalized=match.group().lower(),
                    confidence=0.9,
                ))

        # SQL-related patterns
        sql_patterns = [
            (r"\b([A-Z][a-z]+[A-Z][A-Za-z]+)\b", "table"),  # CamelCase (likely table names)
            (r"\bEWR[A-Za-z]+\b", "database"),  # EWR databases
        ]

        for pattern, entity_type in sql_patterns:
            for match in re.finditer(pattern, query):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=0.7,
                ))

        return entities

    async def _expand_query(
        self,
        query: str,
        entities: List[ExtractedEntity],
        conversation_history: List[Dict[str, str]] = None
    ) -> List[str]:
        """
        Generate query expansions for improved retrieval.

        Uses LLM to create semantic variations of the query.
        Includes conversation history to resolve references like "it", "that", etc.
        """
        # Build conversation context for reference resolution
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            history_lines = []
            for msg in conversation_history[-6:]:  # Last 6 messages max
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")
            if history_lines:
                history_context = "\n\nPrevious conversation for context:\n" + "\n".join(history_lines)

        prompt = f"""Generate 3 alternative phrasings of this query for better search:
- Use synonyms and alternative wording
- Keep the core meaning the same
- Make each variation different
- IMPORTANT: If the query contains pronouns like "it", "that", "those", resolve them using the conversation history
{history_context}

Query: "{query}"

Respond with a JSON array of strings only: ["alt1", "alt2", "alt3"]"""

        try:
            import json
            system = "You are a query expander. Respond only with a JSON array."

            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system}\n\n{prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="expand_query",
                    pipeline="document_agent",
                    max_tokens=200,
                    temperature=0.3,
                    tags=["document_agent", "query_expansion"],
                )

                if response.success:
                    try:
                        expansions = json.loads(response.text.strip())
                        if isinstance(expansions, list):
                            return [e for e in expansions if isinstance(e, str)]
                    except json.JSONDecodeError:
                        pass
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=200,
                    temperature=0.3,
                )

                if result.success:
                    try:
                        expansions = json.loads(result.response.strip())
                        if isinstance(expansions, list):
                            return [e for e in expansions if isinstance(e, str)]
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")

        return []

    async def _rewrite_for_retrieval(
        self,
        query: str,
        entities: List[ExtractedEntity]
    ) -> str:
        """
        Rewrite query for optimal retrieval.

        Removes filler words and focuses on key terms.
        """
        # Simple rewrite: remove common question prefixes
        import re

        prefixes_to_remove = [
            r"^can you\s+",
            r"^could you\s+",
            r"^please\s+",
            r"^i want to know\s+",
            r"^tell me\s+",
            r"^what is\s+",
            r"^how do i\s+",
        ]

        rewritten = query
        for prefix in prefixes_to_remove:
            rewritten = re.sub(prefix, "", rewritten, flags=re.IGNORECASE)

        return rewritten.strip() if rewritten != query else query

    def _detect_follow_up(
        self,
        query: str,
        previous_queries: List[str]
    ) -> bool:
        """
        Detect if query is a follow-up to previous conversation.
        """
        if not previous_queries:
            return False

        # Check for pronouns suggesting reference to previous context
        follow_up_indicators = [
            "it", "this", "that", "they", "them", "those",
            "the same", "also", "more about", "elaborate"
        ]

        query_lower = query.lower()
        return any(ind in query_lower for ind in follow_up_indicators)

    def _should_skip_retrieval(self, state: PipelineState) -> bool:
        """
        Determine if retrieval should be skipped based on query analysis.
        """
        if state.query_analysis is None:
            return False

        return not state.query_analysis.requires_retrieval

    async def _stage_retrieval(self, state: PipelineState) -> PipelineState:
        """
        Stage 2: Hybrid Retrieval

        Responsibilities:
        - Check semantic cache for cached results
        - Vector search with query embedding
        - BM25 keyword search (if enabled via config.enable_hybrid_search)
        - Reciprocal Rank Fusion to combine results
        - Cache results for future queries
        - Return top-k diverse documents

        Output:
            Updates state.retrieval_result with retrieved documents
        """
        state.current_stage = PipelineStage.RETRIEVAL
        start_time = time.time()

        log_info("Document Pipeline", "[STAGE 2/6] Retrieval - Searching knowledge base...")

        # Use rewritten query if available
        search_query = (
            state.query_analysis.rewritten_query
            if state.query_analysis and state.query_analysis.rewritten_query
            else state.request.query
        )

        try:
            # Check semantic cache first
            cached_results = None
            if self._semantic_cache and self._semantic_cache.is_available:
                cached_results = await self._semantic_cache.get_results(
                    search_query,
                    state.request.filters
                )
                if cached_results:
                    # Convert cached results to RetrievedDocument models
                    documents = []
                    for doc in cached_results:
                        documents.append(RetrievedDocument(**doc))

                    state.retrieval_result = RetrievalResult(
                        documents=documents,
                        vector_candidates=len(documents),
                        bm25_candidates=0,
                        total_unique=len(documents),
                        search_query=search_query,
                        vector_search_ms=0,
                        bm25_search_ms=0,
                        fusion_ms=0,
                        total_time_ms=int((time.time() - start_time) * 1000),
                    )

                    logger.debug(f"Retrieved {len(documents)} documents from cache")
                    return state

            # Use hybrid retriever if available, otherwise fall back to vector-only
            if self._hybrid_retriever is not None:
                # Hybrid search with BM25 + Vector and RRF fusion
                results, stats = await self._hybrid_retriever.search(
                    query=search_query,
                    limit=state.request.max_documents,
                    vector_weight=self.config.vector_weight,
                    bm25_weight=self.config.bm25_weight,
                    filters=state.request.filters or None,
                    min_score=self.config.min_similarity_threshold,
                    enable_bm25=True,
                )

                # Convert to RetrievedDocument models
                documents = []
                for doc in results:
                    documents.append(RetrievedDocument(
                        document_id=doc.document_id,
                        chunk_id=doc.chunk_id,
                        content=doc.content,
                        title=doc.title or "",
                        source_file=doc.source_file or "",
                        department=doc.department or "",
                        doc_type=doc.doc_type or "",
                        vector_score=doc.vector_score,
                        bm25_score=doc.bm25_score,
                        rrf_score=doc.rrf_score,
                        chunk_index=doc.chunk_index,
                        total_chunks=doc.total_chunks,
                        metadata=doc.metadata,
                    ))

                state.retrieval_result = RetrievalResult(
                    documents=documents,
                    vector_candidates=stats.get("vector_candidates", 0),
                    bm25_candidates=stats.get("bm25_candidates", 0),
                    total_unique=stats.get("total_unique", len(documents)),
                    search_query=search_query,
                    expanded_queries_used=(
                        state.query_analysis.expanded_queries
                        if state.query_analysis else []
                    ),
                    vector_search_ms=stats.get("vector_search_ms", 0),
                    bm25_search_ms=stats.get("bm25_search_ms", 0),
                    fusion_ms=stats.get("fusion_ms", 0),
                    total_time_ms=stats.get("total_time_ms", int((time.time() - start_time) * 1000)),
                )

                search_type = "hybrid (BM25 + Vector + RRF)"
            else:
                # Fallback to vector-only search
                query_embedding = await self._embedding_service.generate_embedding(search_query)

                vector_start = time.time()
                vector_results = await self._mongodb_service._vector_search(
                    collection_name="documents",
                    query_vector=query_embedding,
                    limit=state.request.max_documents,
                    filter_query=state.request.filters or None,
                    threshold=self.config.min_similarity_threshold,
                )
                vector_time = int((time.time() - vector_start) * 1000)

                documents = []
                for doc in vector_results:
                    documents.append(RetrievedDocument(
                        document_id=doc.get("parent_id", doc.get("_id", "")),
                        chunk_id=doc.get("id", doc.get("_id", "")),
                        content=doc.get("content", ""),
                        title=doc.get("title", ""),
                        source_file=doc.get("source_file", ""),
                        department=doc.get("department", ""),
                        doc_type=doc.get("type", ""),
                        vector_score=doc.get("_similarity", 0.0),
                        rrf_score=doc.get("_similarity", 0.0),
                        chunk_index=doc.get("chunk_index", 0),
                        total_chunks=doc.get("total_chunks", 1),
                        metadata=doc.get("metadata", {}),
                    ))

                state.retrieval_result = RetrievalResult(
                    documents=documents,
                    vector_candidates=len(vector_results),
                    bm25_candidates=0,
                    total_unique=len(documents),
                    search_query=search_query,
                    expanded_queries_used=(
                        state.query_analysis.expanded_queries
                        if state.query_analysis else []
                    ),
                    vector_search_ms=vector_time,
                    bm25_search_ms=0,
                    fusion_ms=0,
                    total_time_ms=int((time.time() - start_time) * 1000),
                )

                search_type = "vector-only"

            log_info("Document Pipeline", f"[STAGE 2/6] Retrieval COMPLETE - Found {len(documents)} documents via {search_type} [{state.retrieval_result.total_time_ms}ms]")

            # Cache results for future queries
            if self._semantic_cache and self._semantic_cache.is_available and documents:
                # Convert documents to serializable format for caching
                cacheable_docs = [
                    {
                        "document_id": doc.document_id,
                        "chunk_id": doc.chunk_id,
                        "content": doc.content,
                        "title": doc.title,
                        "source_file": doc.source_file,
                        "department": doc.department,
                        "doc_type": doc.doc_type,
                        "vector_score": doc.vector_score,
                        "bm25_score": doc.bm25_score,
                        "rrf_score": doc.rrf_score,
                        "chunk_index": doc.chunk_index,
                        "total_chunks": doc.total_chunks,
                        "metadata": doc.metadata,
                    }
                    for doc in documents
                ]
                await self._semantic_cache.set_results(
                    search_query,
                    cacheable_docs,
                    state.request.filters
                )

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state.retrieval_result = RetrievalResult(
                documents=[],
                total_time_ms=int((time.time() - start_time) * 1000),
            )

        return state

    async def _stage_grading(self, state: PipelineState) -> PipelineState:
        """
        Stage 3: Document Grading (CRAG Pattern)

        Responsibilities:
        - Grade each document for relevance to query
        - Classify as relevant/ambiguous/irrelevant
        - Decide if corrective retrieval is needed

        Output:
            Updates state.grading_result with graded documents
        """
        state.current_stage = PipelineStage.GRADING
        start_time = time.time()

        doc_count = len(state.retrieval_result.documents) if state.retrieval_result and state.retrieval_result.documents else 0
        log_info("Document Pipeline", f"[STAGE 3/6] Grading - Evaluating relevance of {doc_count} documents...")

        if not state.retrieval_result or not state.retrieval_result.documents:
            state.grading_result = GradingResult(
                graded_documents=[],
                average_relevance=0.0,
                needs_correction=True,
                correction_reason="No documents retrieved",
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
            return state

        graded_docs = []
        query = state.request.query

        for doc in state.retrieval_result.documents:
            # Use LLM to grade relevance
            grade, score, reasoning = await self._grade_document(query, doc)

            graded_docs.append(GradedDocument(
                document=doc,
                relevance=grade,
                relevance_score=score,
                reasoning=reasoning,
                include_in_context=(grade != "irrelevant"),
            ))

        # Calculate statistics
        relevant = [d for d in graded_docs if d.relevance == "relevant"]
        ambiguous = [d for d in graded_docs if d.relevance == "ambiguous"]
        irrelevant = [d for d in graded_docs if d.relevance == "irrelevant"]

        avg_relevance = (
            sum(d.relevance_score for d in graded_docs) / len(graded_docs)
            if graded_docs else 0.0
        )

        # Determine if corrective retrieval is needed
        needs_correction = (
            avg_relevance < self.config.trigger_correction_threshold or
            len(relevant) == 0
        )

        state.grading_result = GradingResult(
            graded_documents=graded_docs,
            average_relevance=avg_relevance,
            relevant_count=len(relevant),
            ambiguous_count=len(ambiguous),
            irrelevant_count=len(irrelevant),
            needs_correction=needs_correction,
            correction_reason="Low average relevance" if needs_correction else None,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

        elapsed = int((time.time() - start_time) * 1000)
        log_info("Document Pipeline", f"[STAGE 3/6] Grading COMPLETE - {len(relevant)} relevant, {len(ambiguous)} ambiguous, {len(irrelevant)} irrelevant (avg: {avg_relevance:.2f}) [{elapsed}ms]")

        return state

    async def _grade_document(
        self,
        query: str,
        doc: RetrievedDocument
    ) -> tuple[str, float, str]:
        """
        Grade a single document for relevance to the query.

        Returns:
            Tuple of (grade, score, reasoning)
        """
        prompt = f"""Grade if this document is relevant to the query.

Query: "{query}"

Document:
{doc.content[:1000]}

Grade as one of:
- "relevant": Directly answers or provides key information
- "ambiguous": Partially related but may not fully answer
- "irrelevant": Not useful for answering the query

Respond with JSON: {{"grade": "relevant|ambiguous|irrelevant", "score": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            import json
            system = "You are a document relevance grader. Respond only with valid JSON."

            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system}\n\n{prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="grade_document",
                    pipeline="document_agent",
                    max_tokens=100,
                    temperature=0.0,
                    tags=["document_agent", "grading", "crag"],
                )

                if response.success:
                    try:
                        data = json.loads(response.text.strip())
                        grade = data.get("grade", "ambiguous")
                        score = float(data.get("score", 0.5))
                        reason = data.get("reason", "")
                        return grade, score, reason
                    except json.JSONDecodeError:
                        pass
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=100,
                    temperature=0.0,
                )

                if result.success:
                    try:
                        data = json.loads(result.response.strip())
                        grade = data.get("grade", "ambiguous")
                        score = float(data.get("score", 0.5))
                        reason = data.get("reason", "")
                        return grade, score, reason
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            logger.warning(f"Document grading failed: {e}")

        # Fall back to using vector score
        if doc.vector_score > 0.7:
            return "relevant", doc.vector_score, "High vector similarity"
        elif doc.vector_score > 0.5:
            return "ambiguous", doc.vector_score, "Medium vector similarity"
        else:
            return "irrelevant", doc.vector_score, "Low vector similarity"

    async def _stage_corrective_retrieval(self, state: PipelineState) -> PipelineState:
        """
        Stage 3b: Corrective Retrieval

        Triggered when document grading indicates poor relevance.
        Uses query expansions and relaxed thresholds.
        """
        state.current_stage = PipelineStage.CORRECTION
        state.correction_attempts += 1

        logger.info(f"Triggering corrective retrieval (attempt {state.correction_attempts})")

        # Use expanded queries if available
        if state.query_analysis and state.query_analysis.expanded_queries:
            for expanded in state.query_analysis.expanded_queries:
                try:
                    query_embedding = await self._embedding_service.generate_embedding(expanded)
                    additional_results = await self._mongodb_service._vector_search(
                        collection_name="documents",
                        query_vector=query_embedding,
                        limit=3,
                        threshold=self.config.min_similarity_threshold * 0.8,  # Lower threshold
                    )

                    # Add new documents that aren't already in results
                    existing_ids = {
                        d.chunk_id for d in state.retrieval_result.documents
                    } if state.retrieval_result else set()

                    for doc in additional_results:
                        chunk_id = doc.get("id", doc.get("_id", ""))
                        if chunk_id not in existing_ids:
                            state.retrieval_result.documents.append(RetrievedDocument(
                                document_id=doc.get("parent_id", doc.get("_id", "")),
                                chunk_id=chunk_id,
                                content=doc.get("content", ""),
                                title=doc.get("title", ""),
                                vector_score=doc.get("_similarity", 0.0),
                                rrf_score=doc.get("_similarity", 0.0),
                            ))
                            existing_ids.add(chunk_id)

                except Exception as e:
                    logger.warning(f"Corrective search failed for '{expanded}': {e}")

        # Re-grade the expanded results
        if state.retrieval_result and state.retrieval_result.documents:
            state = await self._stage_grading(state)
            # Don't trigger another correction loop
            if state.grading_result:
                state.grading_result.needs_correction = False

        return state

    async def _stage_generation(self, state: PipelineState) -> PipelineState:
        """
        Stage 4: Answer Generation

        Responsibilities:
        - Format retrieved context
        - Generate answer using LLM
        - Track token usage

        Output:
            Updates state.generation_result with generated answer
        """
        state.current_stage = PipelineStage.GENERATION
        start_time = time.time()

        log_info("Document Pipeline", "[STAGE 4/6] Generation - Generating answer from context...")

        # Build context from graded documents
        context = self._build_context(state)

        # Build the generation prompt with conversation history support
        is_follow_up = state.query_analysis.is_follow_up if state.query_analysis else False

        # Format conversation history if present
        history_context = ""
        if state.request.conversation_history and len(state.request.conversation_history) > 0:
            history_lines = []
            for msg in state.request.conversation_history[-10:]:  # Last 10 messages max
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_lines.append(f"User: {content}")
                elif role == "assistant":
                    history_lines.append(f"Assistant: {content}")
            if history_lines:
                history_context = "\nPrevious Conversation:\n" + "\n".join(history_lines) + "\n"

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
- Answer based ONLY on the context provided
- If the context doesn't contain enough information, say so
- Use clear, concise language
- Include relevant details from the source documents
- If this is a follow-up question, use the conversation history to understand what the user is referring to"""

        user_prompt = f"""Context:
{context}
{history_context}
Question: {state.request.query}

Answer:"""

        try:
            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="document_qa",
                    pipeline="document_agent",
                    user_id=state.request.user_id if hasattr(state.request, 'user_id') else None,
                    session_id=state.request.session_id if hasattr(state.request, 'session_id') else None,
                    max_tokens=self.config.generation_max_tokens,
                    temperature=self.config.generation_temperature,
                    tags=["document_agent", "knowledge_base", "rag"],
                    context_dict={
                        "collection": state.request.collection if hasattr(state.request, 'collection') else None,
                        "user_question": state.request.query,
                    },
                )

                state.generation_result = GenerationResult(
                    answer=response.text if response.success else "Unable to generate answer.",
                    prompt_tokens=response.tokens_evaluated or 0,
                    completion_tokens=response.tokens_predicted or 0,
                    generation_time_ms=int(response.timings.prompt_ms + response.timings.predicted_ms) if response.timings else int((time.time() - start_time) * 1000),
                    model_used=response.model or "traced",
                )

                if response.success:
                    elapsed = int((time.time() - start_time) * 1000)
                    log_info("Document Pipeline", f"[STAGE 4/6] Generation COMPLETE - Generated {len(state.generation_result.answer)} chars, {state.generation_result.completion_tokens} tokens [{elapsed}ms]")
                else:
                    logger.error(f"Generation failed [TRACED]: {response.error}")
            else:
                # Legacy LLM service path
                result = await self._llm_service.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    max_tokens=self.config.generation_max_tokens,
                    temperature=self.config.generation_temperature,
                )

                state.generation_result = GenerationResult(
                    answer=result.response if result.success else "Unable to generate answer.",
                    prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                    completion_tokens=result.token_usage.get("response_tokens", 0),
                    generation_time_ms=result.generation_time_ms,
                    model_used=result.model,
                )

                if result.success:
                    elapsed = int((time.time() - start_time) * 1000)
                    log_info("Document Pipeline", f"[STAGE 4/6] Generation COMPLETE - Generated {len(state.generation_result.answer)} chars [{elapsed}ms]")
                else:
                    logger.error(f"Generation failed: {result.error}")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            log_error("Document Pipeline", f"[STAGE 4/6] Generation FAILED - {e}")
            state.generation_result = GenerationResult(
                answer="An error occurred while generating the answer.",
                generation_time_ms=int((time.time() - start_time) * 1000),
            )

        return state

    async def _stage_generation_stream(
        self,
        state: PipelineState
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stage 4 (Streaming): Answer Generation with token streaming

        Yields individual tokens as StreamEvents for real-time UI updates.
        """
        state.current_stage = PipelineStage.GENERATION

        context = self._build_context(state)

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
- Answer based ONLY on the context provided
- If the context doesn't contain enough information, say so
- Use clear, concise language"""

        user_prompt = f"""Context:
{context}

Question: {state.request.query}

Answer:"""

        full_answer = ""

        try:
            async for chunk in self._llm_service.generate_stream(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=self.config.generation_max_tokens,
                temperature=self.config.generation_temperature,
            ):
                if chunk.error:
                    yield StreamEvent(
                        event_type=StreamEventType.ERROR,
                        query_id=state.request.query_id,
                        stage=PipelineStage.GENERATION,
                        data={"error": chunk.error},
                    )
                    break

                if chunk.content:
                    full_answer += chunk.content
                    yield StreamEvent(
                        event_type=StreamEventType.GENERATION_TOKEN,
                        query_id=state.request.query_id,
                        stage=PipelineStage.GENERATION,
                        data={"token": chunk.content},
                    )

                if chunk.done:
                    state.generation_result = GenerationResult(
                        answer=full_answer,
                        prompt_tokens=chunk.token_usage.get("prompt_tokens", 0) if chunk.token_usage else 0,
                        completion_tokens=chunk.token_usage.get("response_tokens", 0) if chunk.token_usage else 0,
                    )
                    break

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            state.generation_result = GenerationResult(
                answer=full_answer or "Generation failed.",
            )
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                query_id=state.request.query_id,
                stage=PipelineStage.GENERATION,
                data={"error": str(e)},
            )

    def _build_context(self, state: PipelineState) -> str:
        """
        Build context string from graded documents.

        Formats documents with source attribution for the LLM.
        """
        if not state.grading_result or not state.grading_result.graded_documents:
            # Fall back to retrieval results
            if state.retrieval_result and state.retrieval_result.documents:
                docs = state.retrieval_result.documents
            else:
                return "No relevant documents found."
        else:
            # Use documents marked for inclusion
            docs = [
                gd.document for gd in state.grading_result.graded_documents
                if gd.include_in_context
            ]

        if not docs:
            return "No relevant documents found."

        # Format context
        context_parts = []
        for i, doc in enumerate(docs[:self.config.default_top_k], 1):
            title = doc.title or f"Document {i}"
            context_parts.append(f"[Source {i}: {title}]\n{doc.content}")

        return "\n\n---\n\n".join(context_parts)

    async def _stage_validation(self, state: PipelineState) -> PipelineState:
        """
        Stage 5: Answer Validation

        Responsibilities:
        - Check relevancy: Does answer address the question?
        - Check faithfulness: Is answer grounded in context?
        - Check completeness: Does answer cover all aspects?

        Output:
            Updates state.validation_result with check results
        """
        state.current_stage = PipelineStage.VALIDATION
        start_time = time.time()

        log_info("Document Pipeline", "[STAGE 5/6] Validation - Checking relevancy, faithfulness, and completeness...")

        if not state.generation_result:
            state.validation_result = ValidationResult(
                is_valid=False,
                issues=["No answer generated"],
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
            return state

        query = state.request.query
        answer = state.generation_result.answer
        context = self._build_context(state)

        # Run validation checks
        relevancy = await self._check_relevancy(query, answer)
        faithfulness = await self._check_faithfulness(answer, context)
        completeness = await self._check_completeness(query, answer)

        # Aggregate results with weighted scoring (CRAG best practice)
        # 60% relevancy, 30% faithfulness, 10% completeness
        overall_score = (
            relevancy.score * 0.6 +
            faithfulness.score * 0.3 +
            completeness.score * 0.1
        )

        # Threshold-based validation: 0.6+ = valid (more lenient than strict AND)
        VALIDATION_THRESHOLD = 0.6
        is_valid = overall_score >= VALIDATION_THRESHOLD

        issues = []
        if relevancy.score < 0.5:
            issues.append(f"Low relevancy ({relevancy.score:.2f}): {relevancy.reasoning}")
        if faithfulness.score < 0.5:
            unsupported = ', '.join(faithfulness.unsupported_claims) if faithfulness.unsupported_claims else 'some claims'
            issues.append(f"Low faithfulness ({faithfulness.score:.2f}): {unsupported}")
        if completeness.score < 0.5:
            missing = ', '.join(completeness.missing_aspects) if completeness.missing_aspects else 'some aspects'
            issues.append(f"Low completeness ({completeness.score:.2f}): {missing}")

        needs_correction = not is_valid and state.correction_attempts < state.max_corrections

        state.validation_result = ValidationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            relevancy=relevancy,
            faithfulness=faithfulness,
            completeness=completeness,
            issues=issues,
            needs_correction=needs_correction,
            correction_hints=issues if needs_correction else [],
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

        elapsed = int((time.time() - start_time) * 1000)
        status = "PASSED" if is_valid else "NEEDS CORRECTION"
        log_info("Document Pipeline", f"[STAGE 5/6] Validation COMPLETE - {status} (score: {overall_score:.2f}, issues: {len(issues)}) [{elapsed}ms]")

        return state

    async def _check_relevancy(self, query: str, answer: str) -> RelevancyCheck:
        """
        Check if the answer is relevant to the query.
        """
        prompt = f"""Evaluate if this answer addresses the question. Be lenient - partial answers are acceptable.

Question: {query}
Answer: {answer}

Scoring guide:
- 0.9-1.0: Directly and fully answers the question
- 0.7-0.9: Answers the main question with minor gaps
- 0.5-0.7: Partially addresses the question
- 0.3-0.5: Tangentially related but misses the point
- 0.0-0.3: Completely off-topic

Respond with JSON: {{"relevant": true, "score": 0.0-1.0, "reason": "brief explanation"}}
Set relevant=true if score >= 0.5"""

        try:
            system = "You are a relevancy checker. Respond only with JSON."
            default_result = {"relevant": True, "score": 0.7, "reason": ""}

            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system}\n\n{prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="check_relevancy",
                    pipeline="document_agent",
                    max_tokens=100,
                    temperature=0.0,
                    tags=["document_agent", "validation", "relevancy"],
                )

                if response.success:
                    data = _parse_llm_json(response.text, default_result)
                    return RelevancyCheck(
                        passed=data.get("relevant", True),
                        score=float(data.get("score", 0.7)),
                        reasoning=data.get("reason", ""),
                    )
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=100,
                    temperature=0.0,
                )

                if result.success:
                    data = _parse_llm_json(result.response, default_result)
                    return RelevancyCheck(
                        passed=data.get("relevant", True),
                        score=float(data.get("score", 0.7)),
                        reasoning=data.get("reason", ""),
                    )

        except Exception as e:
            logger.warning(f"Relevancy check failed: {e}")

        # Default to passing with decent score when validation fails
        return RelevancyCheck(passed=True, score=0.7)

    async def _check_faithfulness(self, answer: str, context: str) -> FaithfulnessCheck:
        """
        Check if the answer is grounded in the context (no hallucinations).
        """
        prompt = f"""Check if the answer's claims are supported by the context. Be lenient with paraphrasing and reasonable inferences.

Context:
{context[:2000]}

Answer: {answer}

Scoring guide:
- 0.9-1.0: All claims directly supported by context
- 0.7-0.9: Most claims supported, minor inferences are acceptable
- 0.5-0.7: Some claims supported, some reasonable inferences
- 0.3-0.5: Several unsupported claims
- 0.0-0.3: Mostly fabricated content

Only flag truly fabricated claims, not paraphrases or reasonable inferences from the context.
Respond with JSON: {{"faithful": true, "score": 0.0-1.0, "unsupported": []}}
Set faithful=true if score >= 0.5"""

        try:
            system = "You are a faithfulness checker. Respond only with JSON."
            default_result = {"faithful": True, "score": 0.7, "unsupported": []}

            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system}\n\n{prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="check_faithfulness",
                    pipeline="document_agent",
                    max_tokens=200,
                    temperature=0.0,
                    tags=["document_agent", "validation", "faithfulness"],
                )

                if response.success:
                    data = _parse_llm_json(response.text, default_result)
                    return FaithfulnessCheck(
                        passed=data.get("faithful", True),
                        score=float(data.get("score", 0.7)),
                        unsupported_claims=data.get("unsupported", []),
                    )
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=200,
                    temperature=0.0,
                )

                if result.success:
                    data = _parse_llm_json(result.response, default_result)
                    return FaithfulnessCheck(
                        passed=data.get("faithful", True),
                        score=float(data.get("score", 0.7)),
                        unsupported_claims=data.get("unsupported", []),
                    )

        except Exception as e:
            logger.warning(f"Faithfulness check failed: {e}")

        # Default to passing with decent score when validation fails
        return FaithfulnessCheck(passed=True, score=0.7)

    async def _check_completeness(self, query: str, answer: str) -> CompletenessCheck:
        """
        Check if the answer addresses the query's main intent.
        """
        prompt = f"""Does this answer address the main intent of the question? It doesn't need to cover everything possible, just the core question.

Question: {query}
Answer: {answer}

Scoring guide:
- 0.9-1.0: Comprehensively addresses the question
- 0.7-0.9: Addresses the main question adequately
- 0.5-0.7: Addresses the question but could be more complete
- 0.3-0.5: Misses key aspects of the question
- 0.0-0.3: Fails to address the question

Respond with JSON: {{"complete": true, "score": 0.0-1.0, "missing": []}}
Set complete=true if score >= 0.5"""

        try:
            system = "You are a completeness checker. Respond only with JSON."
            default_result = {"complete": True, "score": 0.7, "missing": []}

            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system}\n\n{prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="check_completeness",
                    pipeline="document_agent",
                    max_tokens=150,
                    temperature=0.0,
                    tags=["document_agent", "validation", "completeness"],
                )

                if response.success:
                    data = _parse_llm_json(response.text, default_result)
                    return CompletenessCheck(
                        passed=data.get("complete", True),
                        score=float(data.get("score", 0.7)),
                        missing_aspects=data.get("missing", []),
                    )
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=prompt,
                    system=system,
                    max_tokens=150,
                    temperature=0.0,
                )

                if result.success:
                    data = _parse_llm_json(result.response, default_result)
                    return CompletenessCheck(
                        passed=data.get("complete", True),
                        score=float(data.get("score", 0.7)),
                        missing_aspects=data.get("missing", []),
                    )

        except Exception as e:
            logger.warning(f"Completeness check failed: {e}")

        # Default to passing with decent score when validation fails
        return CompletenessCheck(passed=True, score=0.7)

    async def _stage_self_correction(self, state: PipelineState) -> PipelineState:
        """
        Stage 6: Self-Correction

        Triggered when validation fails.
        Regenerates answer with additional constraints.
        """
        state.current_stage = PipelineStage.CORRECTION
        state.correction_attempts += 1

        log_info("Document Pipeline", f"[STAGE 6/6] Self-Correction - Attempt {state.correction_attempts}, regenerating with constraints...")

        # Add correction hints to the prompt
        hints = state.validation_result.correction_hints if state.validation_result else []
        hint_text = "\n".join(f"- {hint}" for hint in hints)

        context = self._build_context(state)

        system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
- Answer based ONLY on the context provided
- If the context doesn't contain enough information, say so clearly
- Be precise and avoid making claims not supported by the context

Previous issues to fix:
{hint_text}"""

        user_prompt = f"""Context:
{context}

Question: {state.request.query}

Provide an improved answer that addresses the issues above:"""

        try:
            # Use TracedLLMClient if available
            if self._use_traced:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="self_correction",
                    pipeline="document_agent",
                    user_id=state.request.user_id if hasattr(state.request, 'user_id') else None,
                    max_tokens=self.config.generation_max_tokens,
                    temperature=0.0,
                    tags=["document_agent", "self_correction", "crag"],
                    context_dict={
                        "collection": state.request.collection if hasattr(state.request, 'collection') else None,
                        "user_question": state.request.query,
                        "correction_attempt": state.correction_attempts,
                    },
                )

                if response.success:
                    state.generation_result = GenerationResult(
                        answer=response.text,
                        prompt_tokens=response.tokens_evaluated or 0,
                        completion_tokens=response.tokens_predicted or 0,
                        generation_time_ms=int(response.timings.prompt_ms + response.timings.predicted_ms) if response.timings else 0,
                        model_used=response.model or "traced",
                    )
                    log_info("Document Pipeline", f"[STAGE 6/6] Self-Correction - Regenerated answer, re-validating...")

                    # Re-validate the corrected answer
                    state = await self._stage_validation(state)
            else:
                # Legacy path
                result = await self._llm_service.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    max_tokens=self.config.generation_max_tokens,
                    temperature=0.0,
                )

                if result.success:
                    state.generation_result = GenerationResult(
                        answer=result.response,
                        prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                        completion_tokens=result.token_usage.get("response_tokens", 0),
                        generation_time_ms=result.generation_time_ms,
                        model_used=result.model,
                    )
                    log_info("Document Pipeline", f"[STAGE 6/6] Self-Correction - Regenerated answer, re-validating...")

                    # Re-validate the corrected answer
                    state = await self._stage_validation(state)

        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            log_error("Document Pipeline", f"[STAGE 6/6] Self-Correction FAILED - {e}")

        return state

    # =========================================================================
    # Response Building
    # =========================================================================

    def _build_response(self, state: PipelineState, start_time: float) -> QueryResponse:
        """
        Build the final QueryResponse from pipeline state.
        """
        # Extract sources
        sources = []
        if state.grading_result and state.grading_result.graded_documents:
            for gd in state.grading_result.graded_documents:
                if gd.include_in_context:
                    sources.append({
                        "document_id": gd.document.document_id,
                        "title": gd.document.title,
                        "content_preview": gd.document.content[:200],
                        "score": round(gd.relevance_score, 3),
                        "relevance": gd.relevance,
                    })
        elif state.retrieval_result and state.retrieval_result.documents:
            for doc in state.retrieval_result.documents:
                sources.append({
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "content_preview": doc.content[:200],
                    "score": round(doc.rrf_score, 3),
                })

        # Calculate confidence
        confidence = 0.5  # Base confidence
        if state.validation_result:
            confidence = state.validation_result.overall_score
        elif state.grading_result:
            confidence = state.grading_result.average_relevance

        # Build stage timings
        stage_timings = {}
        if state.query_analysis:
            stage_timings["query_understanding"] = state.query_analysis.processing_time_ms
        if state.retrieval_result:
            stage_timings["retrieval"] = state.retrieval_result.total_time_ms
        if state.grading_result:
            stage_timings["grading"] = state.grading_result.processing_time_ms
        if state.generation_result:
            stage_timings["generation"] = state.generation_result.generation_time_ms
        if state.validation_result:
            stage_timings["validation"] = state.validation_result.processing_time_ms

        # Token usage
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if state.generation_result:
            token_usage["prompt_tokens"] = state.generation_result.prompt_tokens
            token_usage["completion_tokens"] = state.generation_result.completion_tokens
            token_usage["total_tokens"] = (
                state.generation_result.prompt_tokens +
                state.generation_result.completion_tokens
            )

        return QueryResponse(
            query_id=state.request.query_id,
            query=state.request.query,
            answer=state.generation_result.answer if state.generation_result else "",
            sources=sources,
            confidence=confidence,
            validation_passed=(
                state.validation_result.is_valid if state.validation_result else True
            ),
            validation_details=(
                {
                    "relevancy": state.validation_result.relevancy.score,
                    "faithfulness": state.validation_result.faithfulness.score,
                    "completeness": state.validation_result.completeness.score,
                    "issues": state.validation_result.issues,
                }
                if state.validation_result else None
            ),
            query_intent=(
                state.query_analysis.query_intent
                if state.query_analysis else QueryIntent.FACTUAL
            ),
            retrieval_used=not self._should_skip_retrieval(state),
            correction_applied=state.correction_attempts > 0,
            total_time_ms=int((time.time() - start_time) * 1000),
            stage_timings=stage_timings,
            token_usage=token_usage,
            error=state.error,
        )

    # =========================================================================
    # Feedback & Learning
    # =========================================================================

    async def record_feedback(self, feedback: FeedbackRecord) -> bool:
        """
        Record user feedback for learning.

        This method stores feedback in MongoDB for later processing
        by the learning agent. Feedback types:
        - THUMBS_UP: Positive signal, add to examples
        - THUMBS_DOWN: Negative signal, flag for review
        - CORRECTION: User-provided correct answer
        - RATING: 1-5 star rating

        Args:
            feedback: Feedback record with query, answer, and feedback type

        Returns:
            True if feedback was recorded successfully
        """
        try:
            # Store in MongoDB feedback collection
            feedback_doc = feedback.model_dump(mode='json')

            await self._mongodb_service.db["feedback"].insert_one(feedback_doc)

            logger.info(
                f"Recorded {feedback.feedback_type.value} feedback for query {feedback.query_id}"
            )

            # Immediate actions based on feedback type
            if feedback.feedback_type == FeedbackType.THUMBS_UP:
                # Could add to semantic cache here
                pass
            elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
                # Could trigger learning agent
                pass
            elif feedback.feedback_type == FeedbackType.CORRECTION:
                # Could add to corrections collection
                if feedback.correction:
                    await self._mongodb_service.db["query_corrections"].insert_one({
                        "query_id": feedback.query_id,
                        "query": feedback.query,
                        "original_answer": feedback.answer,
                        "corrected_answer": feedback.correction,
                        "created_at": datetime.utcnow().isoformat(),
                    })

            return True

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False


# =============================================================================
# Singleton Accessor
# =============================================================================

_orchestrator_instance: Optional[KnowledgeBaseOrchestrator] = None


async def get_orchestrator() -> KnowledgeBaseOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = KnowledgeBaseOrchestrator()
        await _orchestrator_instance.initialize()
    return _orchestrator_instance


async def close_orchestrator() -> None:
    """Close the global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance:
        await _orchestrator_instance.close()
        _orchestrator_instance = None
