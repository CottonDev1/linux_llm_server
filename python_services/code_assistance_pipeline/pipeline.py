"""
Code Assistance Pipeline
========================

Main orchestration class for the code assistance RAG pipeline.

Design Rationale:
-----------------
This pipeline follows the standard RAG pattern adapted for code understanding:

1. Retrieval: Multi-source semantic search across methods, classes,
   event handlers, and call relationships

2. Augmentation: Context assembly that prioritizes relevant code
   entities and formats them for LLM comprehension

3. Generation: LLM-powered response generation with streaming support

The pipeline supports both synchronous and streaming modes, enabling
flexible integration with different frontend patterns (polling vs SSE).

Architecture:
- Lazy service initialization for efficient resource usage
- Async-first design for non-blocking I/O
- Modular services that can be tested independently
- Comprehensive logging for debugging and monitoring

Key Features:
- Multi-turn conversation support via history tracking
- Call chain analysis for code flow understanding
- Feedback loop for continuous improvement
- Response caching for performance
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional, List, Dict, Any

from code_assistance_pipeline.models.query_models import (
    CodeQueryRequest,
    CodeQueryResponse,
    CodeQueryOptions,
    ConversationMessage,
    SourceInfo,
    SourceType,
    TokenUsage,
    TimingInfo,
    SSEEvent,
    CodeFeedbackRequest,
    CodeFeedbackResponse,
    CodeStatsResponse,
)
from code_assistance_pipeline.services.code_retriever import CodeRetriever
from code_assistance_pipeline.services.context_builder import ContextBuilder
from code_assistance_pipeline.services.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


class CodeAssistancePipeline:
    """
    Main pipeline for code assistance queries with RAG.

    This pipeline orchestrates:
    1. Code retrieval from MongoDB (methods, classes, events)
    2. Context assembly from retrieved entities
    3. LLM response generation with code context
    4. Response tracking for feedback integration

    Supports both synchronous and streaming response modes.

    Usage:
        pipeline = CodeAssistancePipeline()
        await pipeline.initialize()

        # Synchronous query
        response = await pipeline.process_query(request)

        # Streaming query
        async for event in pipeline.process_query_stream(request):
            print(event.to_sse())

    Attributes:
        retriever: Service for searching code entities
        context_builder: Service for assembling context
        generator: Service for LLM response generation
        mongodb_service: Service for interaction logging
    """

    def __init__(
        self,
        mongodb_uri: str = "mongodb://localhost:27017",
        llm_endpoint: str = "http://localhost:8082"
    ):
        """
        Initialize the code assistance pipeline.

        Args:
            mongodb_uri: Connection string for MongoDB
            llm_endpoint: Endpoint for code LLM (llama.cpp)
        """
        self.mongodb_uri = mongodb_uri
        self.llm_endpoint = llm_endpoint

        # Services - initialized lazily
        self._retriever: Optional[CodeRetriever] = None
        self._context_builder: Optional[ContextBuilder] = None
        self._generator: Optional[ResponseGenerator] = None
        self._mongodb_service = None

        self._initialized = False
        logger.info("CodeAssistancePipeline created")

    async def initialize(self) -> None:
        """
        Initialize all pipeline services.

        This performs lazy initialization of all dependencies.
        Safe to call multiple times - will only initialize once.
        """
        if self._initialized:
            return

        # Initialize retriever
        self._retriever = CodeRetriever()
        await self._retriever.initialize()

        # Initialize context builder (stateless, no init needed)
        self._context_builder = ContextBuilder()

        # Initialize generator
        self._generator = ResponseGenerator()
        await self._generator.initialize()

        # Initialize MongoDB service for interaction logging
        try:
            from mongodb import MongoDBService
            self._mongodb_service = MongoDBService.get_instance()
            if not self._mongodb_service.is_initialized:
                await self._mongodb_service.initialize()
        except Exception as e:
            logger.warning(f"MongoDB service initialization failed: {e}")
            # Continue without MongoDB - logging will be disabled

        self._initialized = True
        logger.info("CodeAssistancePipeline initialized")

    async def _ensure_initialized(self) -> None:
        """Ensure pipeline is initialized before use."""
        if not self._initialized:
            await self.initialize()

    async def process_query(
        self,
        request: CodeQueryRequest
    ) -> CodeQueryResponse:
        """
        Process a code assistance query (non-streaming).

        This is the main entry point for synchronous queries.
        Performs full RAG pipeline: retrieve -> augment -> generate.

        Args:
            request: CodeQueryRequest with query and options

        Returns:
            CodeQueryResponse with answer, sources, and metrics

        Design Note:
        This method blocks until the full response is generated.
        For real-time UI updates, use process_query_stream instead.
        """
        await self._ensure_initialized()

        start_time = time.time()
        response_id = str(uuid.uuid4())

        logger.info(f"Processing query: '{request.query[:50]}...' (id={response_id})")

        # Step 1: Retrieve code context
        retrieval_start = time.time()

        methods, classes, event_handlers, call_chain = await self._retriever.retrieve_comprehensive(
            query=request.query,
            project=request.project,
            method_limit=request.options.method_limit,
            class_limit=request.options.class_limit,
            event_limit=request.options.event_handler_limit,
            include_call_chains=request.options.include_call_chains,
            max_depth=request.options.max_depth
        )

        retrieval_time = int((time.time() - retrieval_start) * 1000)
        logger.info(f"Retrieval completed in {retrieval_time}ms")

        # Step 2: Build context
        context, sources = self._context_builder.build_context(
            methods=methods,
            classes=classes,
            event_handlers=event_handlers,
            call_chain=call_chain,
            history=request.history
        )

        # Step 3: Build prompt
        prompt = self._context_builder.build_prompt(
            query=request.query,
            context=context
        )

        # Step 4: Generate response
        generation_start = time.time()

        answer, token_usage, gen_time = await self._generator.generate(
            prompt=prompt,
            temperature=request.options.temperature,
            max_tokens=request.options.max_tokens
        )

        generation_time = int((time.time() - generation_start) * 1000)
        total_time = int((time.time() - start_time) * 1000)

        logger.info(f"Response generated in {generation_time}ms, total {total_time}ms")

        # Step 5: Log interaction for feedback
        await self._log_interaction(
            response_id=response_id,
            query=request.query,
            answer=answer,
            sources=sources,
            call_chain=call_chain,
            project=request.project,
            model=request.model,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time
        )

        # Build response
        timing = TimingInfo(
            retrieval_ms=retrieval_time,
            generation_ms=generation_time,
            total_ms=total_time
        )

        return CodeQueryResponse(
            answer=answer,
            sources=sources,
            call_chain=call_chain,
            response_id=response_id,
            token_usage=token_usage,
            timing=timing
        )

    async def process_query_stream(
        self,
        request: CodeQueryRequest
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Process a code assistance query with streaming response.

        Yields SSE events for each stage of the pipeline:
        1. "status" - Retrieval progress
        2. "sources" - Retrieved code sources
        3. "streaming" - LLM tokens as they arrive
        4. "complete" - Final response with metrics
        5. "error" - Error information if failure occurs

        Args:
            request: CodeQueryRequest with query and options

        Yields:
            SSEEvent objects for each pipeline stage

        Design Note:
        Streaming enables real-time UI updates. The frontend can
        display retrieval status, show sources immediately after
        retrieval, and stream the answer as it generates.
        """
        await self._ensure_initialized()

        start_time = time.time()
        response_id = str(uuid.uuid4())

        logger.info(f"Streaming query: '{request.query[:50]}...' (id={response_id})")

        try:
            # Step 1: Emit retrieving status
            yield SSEEvent(
                event="status",
                data={"status": "retrieving", "message": "Searching codebase..."}
            )

            # Step 2: Retrieve code context
            retrieval_start = time.time()

            methods, classes, event_handlers, call_chain = await self._retriever.retrieve_comprehensive(
                query=request.query,
                project=request.project,
                method_limit=request.options.method_limit,
                class_limit=request.options.class_limit,
                event_limit=request.options.event_handler_limit,
                include_call_chains=request.options.include_call_chains,
                max_depth=request.options.max_depth
            )

            retrieval_time = int((time.time() - retrieval_start) * 1000)

            # Build sources for response
            sources = []
            for method in methods[:8]:
                sources.append(self._retriever.to_source_info(method, SourceType.METHOD))
            for cls in classes[:4]:
                sources.append(self._retriever.to_source_info(cls, SourceType.CLASS))

            # Step 3: Emit sources
            yield SSEEvent(
                event="sources",
                data={
                    "status": "sources",
                    "sources": [s.model_dump() for s in sources],
                    "call_chain": call_chain,
                    "response_id": response_id
                }
            )

            # Step 4: Build context and prompt
            context, _ = self._context_builder.build_context(
                methods=methods,
                classes=classes,
                event_handlers=event_handlers,
                call_chain=call_chain,
                history=request.history
            )

            prompt = self._context_builder.build_prompt(
                query=request.query,
                context=context
            )

            # Step 5: Stream response
            yield SSEEvent(
                event="status",
                data={"status": "generating", "message": "Generating response..."}
            )

            full_answer = ""
            async for event in self._generator.generate_stream_sse(
                prompt=prompt,
                response_id=response_id,
                temperature=request.options.temperature,
                max_tokens=request.options.max_tokens
            ):
                if event.event == "streaming":
                    full_answer += event.data.get("token", "")
                yield event

            # Step 6: Log interaction
            total_time = int((time.time() - start_time) * 1000)
            await self._log_interaction(
                response_id=response_id,
                query=request.query,
                answer=full_answer[:5000],
                sources=sources,
                call_chain=call_chain,
                project=request.project,
                model=request.model,
                retrieval_time=retrieval_time,
                generation_time=total_time - retrieval_time,
                total_time=total_time
            )

        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            yield SSEEvent(
                event="error",
                data={"status": "error", "error": str(e)}
            )

    async def submit_feedback(
        self,
        feedback: CodeFeedbackRequest
    ) -> CodeFeedbackResponse:
        """
        Submit feedback on a code assistance response.

        Stores feedback in MongoDB for analysis and model improvement.

        Args:
            feedback: Feedback data including response_id and rating

        Returns:
            CodeFeedbackResponse confirming feedback was recorded

        Design Note:
        Feedback is essential for RAG improvement. It helps identify:
        - Retrieval failures (wrong methods found)
        - Generation issues (incorrect explanations)
        - Missing context (expected methods not retrieved)
        """
        await self._ensure_initialized()

        logger.info(f"Feedback received for response {feedback.response_id}")

        try:
            if self._mongodb_service:
                # Store feedback in dedicated collection
                from sql_pipeline.services.feedback_service import FeedbackService

                feedback_service = FeedbackService(self._mongodb_service)
                result = await feedback_service.store_feedback(
                    feedback_type="rating" if feedback.is_helpful else "correction",
                    query_id=feedback.response_id,
                    rating={
                        "is_helpful": feedback.is_helpful,
                        "score": 4 if feedback.is_helpful else 1,
                        "comment": feedback.comment
                    } if feedback.is_helpful else None,
                    correction={
                        "error_type": feedback.error_category.value if feedback.error_category else "other",
                        "comment": feedback.comment,
                        "expected_methods": feedback.expected_methods
                    } if not feedback.is_helpful else None,
                    metadata={
                        "source": "code_chat",
                        "error_category": feedback.error_category.value if feedback.error_category else None,
                        "expected_methods": feedback.expected_methods
                    }
                )

                return CodeFeedbackResponse(
                    success=True,
                    feedback_id=result.get("feedback_id", feedback.response_id)
                )
            else:
                # Fallback - just acknowledge
                return CodeFeedbackResponse(
                    success=True,
                    feedback_id=feedback.response_id
                )

        except Exception as e:
            logger.error(f"Feedback storage failed: {e}")
            return CodeFeedbackResponse(
                success=False,
                feedback_id=feedback.response_id
            )

    async def get_stats(self) -> CodeStatsResponse:
        """
        Get statistics about indexed code entities.

        Returns:
            CodeStatsResponse with entity counts and timestamp
        """
        await self._ensure_initialized()

        stats = await self._retriever.get_stats()

        from datetime import datetime
        return CodeStatsResponse(
            code_entities=stats,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    async def _log_interaction(
        self,
        response_id: str,
        query: str,
        answer: str,
        sources: List[SourceInfo],
        call_chain: List[str],
        project: Optional[str],
        model: str,
        retrieval_time: int,
        generation_time: int,
        total_time: int
    ) -> None:
        """
        Log interaction to MongoDB for feedback tracking.

        Args:
            response_id: Unique response ID
            query: User query
            answer: Generated answer
            sources: Retrieved sources
            call_chain: Call chain if retrieved
            project: Project filter
            model: LLM model used
            retrieval_time: Retrieval time in ms
            generation_time: Generation time in ms
            total_time: Total time in ms
        """
        try:
            if self._mongodb_service is None:
                return

            collection = self._mongodb_service.db["code_interactions"]

            document = {
                "response_id": response_id,
                "query": query,
                "answer": answer[:5000],  # Truncate for storage
                "sources": [s.name for s in sources],
                "call_chain": call_chain,
                "project": project or "all",
                "model_used": model,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "total_time_ms": total_time,
                "feedback_received": False,
                "created_at": time.time()
            }

            await collection.insert_one(document)
            logger.debug(f"Logged interaction {response_id}")

        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")

    async def close(self) -> None:
        """
        Close all pipeline resources.

        Should be called when the pipeline is no longer needed
        to release database connections and other resources.
        """
        self._initialized = False
        logger.info("CodeAssistancePipeline closed")


# =============================================================================
# Singleton Accessor
# =============================================================================

_pipeline_instance: Optional[CodeAssistancePipeline] = None


async def get_code_assistance_pipeline() -> CodeAssistancePipeline:
    """
    Get or create the global CodeAssistancePipeline instance.

    Uses singleton pattern to ensure efficient resource usage.

    Returns:
        Initialized CodeAssistancePipeline instance
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = CodeAssistancePipeline()
        await _pipeline_instance.initialize()

    return _pipeline_instance


async def close_code_assistance_pipeline() -> None:
    """Close the global CodeAssistancePipeline instance."""
    global _pipeline_instance

    if _pipeline_instance:
        await _pipeline_instance.close()
        _pipeline_instance = None
