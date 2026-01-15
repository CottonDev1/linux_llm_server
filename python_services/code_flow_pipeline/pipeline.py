"""
Code Flow Pipeline
==================

Main orchestration class for code flow analysis. Coordinates query
classification, multi-stage retrieval, call chain building, and
LLM synthesis.

Design Rationale:
-----------------
The CodeFlowPipeline is the central coordinator for code flow analysis.
It follows a multi-stage architecture similar to the SQL pipeline:

1. **Classification**: Determine query type to select retrieval strategy
2. **Retrieval**: Multi-hop vector search across code-related collections
3. **Chain Building**: Construct execution paths from method relationships
4. **Synthesis**: Use LLM to generate a coherent answer from retrieved context

This design provides:
- Separation of concerns (each stage is independently testable)
- Flexibility (stages can be customized or skipped)
- Observability (streaming events for each stage)
- Caching (results can be cached at multiple levels)

Architecture Patterns:
---------------------
- **Lazy Service Initialization**: Services are created on first use
- **Singleton Pattern**: Reuse service instances across requests
- **Async/Await**: Non-blocking I/O throughout
- **SSE Streaming**: Real-time progress updates for long operations

Production Considerations:
-------------------------
- Add request ID tracking for debugging
- Implement distributed tracing (OpenTelemetry)
- Add metrics for each stage (latency, result counts)
- Consider result caching with TTL
- Monitor LLM token usage for cost control
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime

from code_flow_pipeline.models.query_models import (
    CodeFlowRequest,
    CodeFlowResponse,
    MethodLookupRequest,
    MethodLookupResponse,
    CallChainRequest,
    CallChainResponse,
    QueryClassification,
    QueryType,
    RetrievalResults,
    CallChain,
    CallTree,
    MethodInfo,
    FormattedResult,
    SSEEvent,
)
from code_flow_pipeline.services.query_classifier import (
    QueryClassifier,
    get_query_classifier,
)
from code_flow_pipeline.services.multi_stage_retrieval import (
    MultiStageRetrieval,
    get_multi_stage_retrieval,
)
from code_flow_pipeline.services.call_chain_builder import (
    CallChainBuilder,
    get_call_chain_builder,
)

logger = logging.getLogger(__name__)

# Try to import TracedLLMClient
try:
    from llm.integration import generate_text
    TRACED_LLM_AVAILABLE = True
except ImportError:
    TRACED_LLM_AVAILABLE = False
    logger.info("TracedLLMClient not available, using legacy LLM service")


class CodeFlowPipeline:
    """
    Main pipeline for code flow analysis.

    This pipeline orchestrates:
    1. Query classification (determine type and retrieval strategy)
    2. Multi-stage retrieval (search across methods, classes, UI events, etc.)
    3. Call chain building (trace execution paths)
    4. LLM synthesis (generate coherent answer)

    Usage:
        pipeline = CodeFlowPipeline()
        response = await pipeline.analyze(request)

        # Or with streaming:
        async for event in pipeline.analyze_stream(request):
            print(event.to_sse())
    """

    # Cache TTL in seconds
    CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        mongodb_uri: str = "mongodb://localhost:27017",
        llm_endpoint: str = "http://localhost:8081",
        cache_enabled: bool = True,
    ):
        """
        Initialize the code flow pipeline.

        Args:
            mongodb_uri: MongoDB connection string
            llm_endpoint: LLM service endpoint
            cache_enabled: Whether to enable result caching
        """
        self.mongodb_uri = mongodb_uri
        self.llm_endpoint = llm_endpoint
        self.cache_enabled = cache_enabled

        # Services (lazy initialization)
        self._classifier: Optional[QueryClassifier] = None
        self._retrieval: Optional[MultiStageRetrieval] = None
        self._chain_builder: Optional[CallChainBuilder] = None
        self._llm_service = None
        self._mongodb_service = None
        self._use_traced = TRACED_LLM_AVAILABLE

        # Cache
        self._cache: Dict[str, tuple[CodeFlowResponse, float]] = {}

        logger.info(f"CodeFlowPipeline initialized (traced={self._use_traced})")

    async def _get_services(self):
        """Initialize all services lazily."""
        if self._classifier is None:
            self._classifier = get_query_classifier()

        if self._retrieval is None:
            self._retrieval = await get_multi_stage_retrieval()

        if self._chain_builder is None:
            self._chain_builder = await get_call_chain_builder()

        # Only load legacy LLM service if traced is not available
        if self._llm_service is None and not self._use_traced:
            from services.llm_service import get_llm_service
            self._llm_service = await get_llm_service()

        if self._mongodb_service is None:
            from mongodb import MongoDBService
            self._mongodb_service = MongoDBService.get_instance()
            if not self._mongodb_service.is_initialized:
                await self._mongodb_service.initialize()

    async def analyze(self, request: CodeFlowRequest) -> CodeFlowResponse:
        """
        Analyze a code flow query and return a complete response.

        Args:
            request: The code flow analysis request

        Returns:
            CodeFlowResponse with answer, sources, and metadata
        """
        await self._get_services()

        start_time = time.time()

        # Check cache
        cache_key = self._make_cache_key(request)
        cached = self._get_cached(cache_key)
        if cached:
            cached.cached = True
            return cached

        # Stage 1: Classify query
        classification = self._classifier.classify(request.query)
        logger.info(f"Query classified as '{classification.type.value}' (confidence={classification.confidence})")

        # Stage 2: Multi-stage retrieval
        retrieval_results = await self._retrieval.execute(
            query=request.query,
            project=request.project,
            classification=classification,
            include_call_graph=request.include_call_graph,
        )
        logger.info(f"Retrieved {retrieval_results.total_results} results")

        # Stage 3: Build call chains
        call_chains: List[CallChain] = []
        if request.include_call_graph and retrieval_results.methods:
            call_chains = await self._chain_builder.build_chains_from_methods(
                methods=retrieval_results.methods,
                ui_events=retrieval_results.ui_events,
                project=request.project,
                max_hops=request.max_hops,
            )
            logger.info(f"Built {len(call_chains)} call chains")

        # Stage 4: LLM synthesis
        answer = await self._synthesize_answer(
            query=request.query,
            project=request.project,
            results=retrieval_results,
            call_chains=call_chains,
        )

        # Build response
        processing_time = time.time() - start_time

        response = CodeFlowResponse(
            success=True,
            query=request.query,
            project=request.project,
            answer=answer,
            query_type=classification.type,
            confidence=classification.confidence,
            sources=self._extract_sources(retrieval_results),
            results=retrieval_results,
            call_chains=call_chains,
            total_results=retrieval_results.total_results,
            processing_time=processing_time,
            cached=False,
            metadata={
                "classification_patterns": classification.matched_patterns,
                "stages_executed": len(self._classifier.get_retrieval_stages(classification)),
            }
        )

        # Cache the response
        self._set_cached(cache_key, response)

        return response

    async def analyze_stream(
        self,
        request: CodeFlowRequest,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Analyze a code flow query with streaming progress events.

        Args:
            request: The code flow analysis request

        Yields:
            SSEEvent objects for each pipeline stage
        """
        await self._get_services()

        start_time = time.time()

        def progress_event(stage: str, message: str, step: int = 0, total_steps: int = 5) -> SSEEvent:
            """Create a progress event."""
            return SSEEvent(
                event="status",
                data=json.dumps({
                    "type": "progress",
                    "stage": stage,
                    "message": message,
                    "step": step,
                    "totalSteps": total_steps,
                    "elapsed": round(time.time() - start_time, 2)
                })
            )

        try:
            # Stage 1: Classification
            yield progress_event("classification", "Classifying query...", 1, 5)

            classification = self._classifier.classify(request.query)

            yield SSEEvent(
                event="classification",
                data=json.dumps({
                    "type": classification.type.value,
                    "confidence": classification.confidence,
                    "patterns": classification.matched_patterns,
                })
            )

            # Stage 2: Retrieval
            yield progress_event("retrieval", "Searching codebase...", 2, 5)

            retrieval_results = await self._retrieval.execute(
                query=request.query,
                project=request.project,
                classification=classification,
                include_call_graph=request.include_call_graph,
            )

            yield SSEEvent(
                event="retrieval",
                data=json.dumps({
                    "total_results": retrieval_results.total_results,
                    "methods": len(retrieval_results.methods),
                    "classes": len(retrieval_results.classes),
                    "ui_events": len(retrieval_results.ui_events),
                    "business_processes": len(retrieval_results.business_processes),
                })
            )

            # Stage 3: Call chains
            call_chains: List[CallChain] = []
            if request.include_call_graph and retrieval_results.methods:
                yield progress_event("chains", "Building call chains...", 3, 5)

                call_chains = await self._chain_builder.build_chains_from_methods(
                    methods=retrieval_results.methods,
                    ui_events=retrieval_results.ui_events,
                    project=request.project,
                    max_hops=request.max_hops,
                )

                yield SSEEvent(
                    event="chains",
                    data=json.dumps({
                        "count": len(call_chains),
                        "chains": [
                            {
                                "start": chain.start_method,
                                "end": chain.end_method,
                                "depth": chain.depth,
                                "touches_database": chain.touches_database,
                            }
                            for chain in call_chains[:5]  # Top 5 chains
                        ]
                    })
                )

            # Stage 4: Synthesis
            yield progress_event("synthesis", "Generating answer...", 4, 5)

            answer = await self._synthesize_answer(
                query=request.query,
                project=request.project,
                results=retrieval_results,
                call_chains=call_chains,
            )

            yield SSEEvent(
                event="synthesis",
                data=json.dumps({
                    "answer_length": len(answer),
                    "status": "complete",
                })
            )

            # Final result
            yield progress_event("complete", "Analysis complete", 5, 5)

            response = CodeFlowResponse(
                success=True,
                query=request.query,
                project=request.project,
                answer=answer,
                query_type=classification.type,
                confidence=classification.confidence,
                sources=self._extract_sources(retrieval_results),
                results=retrieval_results,
                call_chains=call_chains,
                total_results=retrieval_results.total_results,
                processing_time=time.time() - start_time,
            )

            yield SSEEvent(
                event="result",
                data=response.model_dump_json()
            )

            yield SSEEvent(
                event="done",
                data=json.dumps({
                    "processing_time": round(time.time() - start_time, 2)
                })
            )

        except Exception as e:
            logger.error(f"Code flow analysis error: {e}", exc_info=True)
            yield SSEEvent(
                event="error",
                data=json.dumps({
                    "error": str(e),
                    "type": type(e).__name__,
                })
            )

    async def lookup_method(
        self,
        request: MethodLookupRequest,
    ) -> MethodLookupResponse:
        """
        Look up methods by name, class, or signature.

        Args:
            request: Method lookup request

        Returns:
            MethodLookupResponse with matching methods
        """
        await self._get_services()

        # Build search query
        search_terms = []
        if request.method_name:
            search_terms.append(f"method {request.method_name}")
        if request.class_name:
            search_terms.append(f"class {request.class_name}")
        if request.signature:
            search_terms.append(f"signature {request.signature}")

        search_query = " ".join(search_terms)

        # Search methods collection
        results = await self._mongodb.search_vectors(
            query=search_query,
            project=request.project,
            category="code",
            doc_type="method",
            limit=request.limit * 2,  # Over-fetch for filtering
        )

        # Filter results
        filtered = []
        for r in results:
            metadata = r.get("metadata", {})

            # Filter by method name
            if request.method_name:
                method_name = metadata.get("methodName", "")
                if request.method_name.lower() not in method_name.lower():
                    continue

            # Filter by class name
            if request.class_name:
                class_name = metadata.get("className", "")
                if request.class_name.lower() not in class_name.lower():
                    continue

            # Filter by signature
            if request.signature:
                signature = metadata.get("signature", "") or metadata.get("fullMethodSignature", "")
                if request.signature.lower() not in signature.lower():
                    continue

            filtered.append(r)

        # Convert to MethodInfo
        methods = []
        for r in filtered[:request.limit]:
            metadata = r.get("metadata", {})
            content = r.get("content", "")

            method_info = MethodInfo(
                name=metadata.get("methodName", ""),
                class_name=metadata.get("className"),
                namespace=metadata.get("namespace"),
                signature=metadata.get("signature") or metadata.get("fullMethodSignature"),
                purpose_summary=metadata.get("purposeSummary"),
                file_path=metadata.get("filePath"),
                start_line=metadata.get("startLine"),
                end_line=metadata.get("endLine"),
                return_type=metadata.get("returnType"),
                is_public=metadata.get("isPublic"),
                is_static=metadata.get("isStatic"),
                is_async=metadata.get("isAsync"),
                calls=self._try_parse_json(metadata.get("callsMethod")),
                called_by=self._try_parse_json(metadata.get("calledByMethod")),
                database_tables=self._try_parse_json(metadata.get("databaseTables")),
                business_domain=metadata.get("businessDomain"),
                similarity=r.get("similarity", r.get("score", 0.0)),
                project=metadata.get("project"),
                description=content[:500] if content else None,
            )
            methods.append(method_info)

        logger.info(f"Found {len(methods)} methods matching query")

        # Convert MethodInfo dataclass objects to dicts for pydantic model
        methods_dicts = [asdict(m) for m in methods]

        return MethodLookupResponse(
            success=True,
            methods=methods_dicts,
            total=len(methods),
        )

    async def build_call_chain(
        self,
        request: CallChainRequest,
    ) -> CallChainResponse:
        """
        Build call chains from an entry point.

        Args:
            request: Call chain request

        Returns:
            CallChainResponse with call tree and chains
        """
        await self._get_services()

        # Search for call graph entries
        call_graph_results = await self._mongodb.search_vectors(
            query=request.entry_point,
            project=request.project,
            category="relationship",
            doc_type="method-call",
            limit=50,
        )

        # Convert to FormattedResult
        formatted_results = [
            FormattedResult(
                id=r.get("id") or r.get("_id"),
                similarity=r.get("similarity", r.get("score", 0.0)),
                content=r.get("content", "")[:300],
                metadata=r.get("metadata", {}),
            )
            for r in call_graph_results
        ]

        # Build call tree
        call_tree = await self._chain_builder.build_call_tree(
            entry_point=request.entry_point,
            call_graph_results=formatted_results,
            max_depth=request.max_depth,
        )

        # Build linear chains
        call_chains = await self._chain_builder.build_chains(
            start_method=request.entry_point,
            project=request.project,
            target_method=request.target_method,
            max_depth=request.max_depth,
        )

        return CallChainResponse(
            success=True,
            entry_point=request.entry_point,
            call_tree=call_tree,
            call_chains=call_chains,
            raw_results=len(call_graph_results),
        )

    async def _synthesize_answer(
        self,
        query: str,
        project: Optional[str],
        results: RetrievalResults,
        call_chains: List[CallChain],
    ) -> str:
        """
        Synthesize a coherent answer using LLM.

        Args:
            query: Original query
            project: Project scope
            results: Retrieved context
            call_chains: Execution paths

        Returns:
            Synthesized answer text
        """
        start_time = time.time()

        # Build context sections
        context_sections = []

        # Business processes
        if results.business_processes:
            section = "## Business Processes\n\n"
            for p in results.business_processes:
                content = p.metadata.get("content") or p.content or ""
                section += content[:500] + "\n\n---\n\n"
            context_sections.append(section)

        # Key methods
        if results.methods:
            section = "## Key Methods\n\n"
            for m in results.methods[:10]:
                method_name = m.metadata.get("methodName", "Unknown")
                class_name = m.metadata.get("className", "")
                purpose = m.metadata.get("purposeSummary", "")
                content = m.content or ""
                section += f"### {class_name}.{method_name}\n"
                if purpose:
                    section += f"Purpose: {purpose}\n"
                section += f"{content[:300]}...\n\n"
            context_sections.append(section)

        # UI entry points
        if results.ui_events:
            section = "## UI Entry Points\n\n"
            for e in results.ui_events:
                content = e.content or e.metadata.get("content", "")
                section += content[:300] + "\n\n"
            context_sections.append(section)

        # Call chains
        if call_chains:
            section = "## Call Chains\n\n"
            for idx, chain in enumerate(call_chains[:5]):
                steps = " -> ".join(step.method for step in chain.steps)
                section += f"Chain {idx + 1}: {steps}\n"
                if chain.touches_database:
                    tables = []
                    for step in chain.steps:
                        tables.extend(step.database_tables)
                    if tables:
                        section += f"  Database: {', '.join(tables)}\n"
            context_sections.append(section)

        context = "\n\n".join(context_sections)

        # Build prompt
        system_prompt = """You are a code analysis expert. Explain code flow and architecture based on the provided analysis.
Focus on answering the user's question with:
1. User workflow (how users interact with the feature)
2. Code flow (execution path from UI to database)
3. Business logic (validations, rules, data transformations)
4. Data operations (database tables affected, operations performed)
5. Error handling and edge cases

Be concise but comprehensive. Use technical terminology appropriately."""

        user_prompt = f"""Question: {query}

Project: {project or "Not specified"}

Code Analysis Results:
{context}

Please provide a clear, structured answer to the question."""

        # Query LLM - try TracedLLMClient first
        try:
            if self._use_traced:
                # Use TracedLLMClient for automatic monitoring
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await generate_text(
                    prompt=full_prompt,
                    operation="code_flow_synthesis",
                    pipeline="code_flow",
                    max_tokens=2048,
                    temperature=0.5,
                    tags=["code_flow", "synthesis"],
                    context_dict={"database": project, "user_question": query} if project else {"user_question": query},
                )

                if response.success:
                    logger.info(f"LLM synthesis completed in {time.time() - start_time:.2f}s [TRACED]")
                    return response.text
                else:
                    logger.error(f"TracedLLM synthesis failed: {response.error}")
                    return self._generate_fallback_answer(query, results, call_chains)
            else:
                # Use legacy LLM service
                result = await self._llm_service.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    max_tokens=2048,
                    temperature=0.5,
                    use_cache=True,
                )

                if result.success:
                    logger.info(f"LLM synthesis completed in {time.time() - start_time:.2f}s")
                    return result.response
                else:
                    logger.error(f"LLM synthesis failed: {result.error}")
                    return self._generate_fallback_answer(query, results, call_chains)

        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            return self._generate_fallback_answer(query, results, call_chains)

    def _generate_fallback_answer(
        self,
        query: str,
        results: RetrievalResults,
        call_chains: List[CallChain],
    ) -> str:
        """Generate a fallback answer without LLM."""
        sections = []

        sections.append(f"## Answer to: \"{query}\"\n")

        if results.business_processes:
            content = results.business_processes[0].content or ""
            sections.append(f"### Business Process Overview\n\n{content[:500]}...\n")

        if results.methods:
            methods_section = "### Key Methods Found\n\n"
            for m in results.methods[:5]:
                class_name = m.metadata.get("className", "")
                method_name = m.metadata.get("methodName", "")
                purpose = m.metadata.get("purposeSummary", "Method implementation")
                methods_section += f"- **{class_name}.{method_name}**: {purpose}\n"
            sections.append(methods_section)

        if call_chains:
            chains_section = "### Execution Flow\n\n"
            for idx, chain in enumerate(call_chains[:3]):
                chains_section += f"**Chain {idx + 1}:**\n"
                for i, step in enumerate(chain.steps):
                    chains_section += f"{i + 1}. {step.method}\n"
                chains_section += "\n"
            sections.append(chains_section)

        return "\n\n".join(sections)

    def _extract_sources(
        self,
        results: RetrievalResults,
    ) -> List[Dict[str, Any]]:
        """Extract source information from retrieval results."""
        sources = []

        for category_name, category_results in [
            ("business_process", results.business_processes),
            ("method", results.methods),
            ("class", results.classes),
            ("ui_event", results.ui_events),
        ]:
            for r in category_results[:5]:  # Top 5 per category
                source = {
                    "type": category_name,
                    "id": r.id,
                    "similarity": r.similarity,
                }
                if r.metadata.get("methodName"):
                    source["name"] = f"{r.metadata.get('className', '')}.{r.metadata.get('methodName')}"
                elif r.metadata.get("className"):
                    source["name"] = r.metadata.get("className")
                sources.append(source)

        return sources

    def _make_cache_key(self, request: CodeFlowRequest) -> str:
        """Generate a cache key for the request."""
        return f"{request.project or 'all'}:{request.query}:{request.include_call_graph}"

    def _get_cached(self, key: str) -> Optional[CodeFlowResponse]:
        """Get a cached response if valid."""
        if not self.cache_enabled:
            return None

        cached = self._cache.get(key)
        if cached is None:
            return None

        response, timestamp = cached
        if time.time() - timestamp > self.CACHE_TTL:
            del self._cache[key]
            return None

        return response

    def _set_cached(self, key: str, response: CodeFlowResponse):
        """Cache a response."""
        if not self.cache_enabled:
            return

        self._cache[key] = (response, time.time())

    def _try_parse_json(self, value: Any) -> List[str]:
        """Safely parse a JSON string to a list."""
        if not value:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return []

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        if self._chain_builder:
            self._chain_builder.clear_cache()
        logger.info("Cache cleared")

    async def close(self):
        """Close all service connections."""
        if self._llm_service:
            await self._llm_service.close()
        logger.info("CodeFlowPipeline closed")


# Singleton instance
_pipeline_instance: Optional[CodeFlowPipeline] = None


async def get_code_flow_pipeline() -> CodeFlowPipeline:
    """Get or create the singleton CodeFlowPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CodeFlowPipeline()
        await _pipeline_instance._get_services()
    return _pipeline_instance
