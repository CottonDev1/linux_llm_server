"""
Prefect Code Assistance Flow
============================

Orchestrates code assistance queries as a Prefect workflow.

This flow provides:
- Task-level retry handling
- Detailed logging and artifacts
- Performance tracking
- Async-native execution

Design Rationale:
-----------------
Running code assistance as a Prefect flow enables:
1. Built-in retries for transient failures
2. Visual progress tracking in Prefect dashboard
3. Task-level timing and artifact generation
4. Integration with larger data pipelines

Usage:
    from prefect import flow
    from code_assistance_pipeline.prefect import run_code_assist_flow

    result = run_code_assist_flow(
        query="How does the Save button work?",
        project="Gin"
    )
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class RetrievalResult:
    """Result from code retrieval task."""
    query: str
    project: Optional[str]
    methods_found: int = 0
    classes_found: int = 0
    event_handlers_found: int = 0
    call_chain_length: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Result from LLM generation task."""
    query: str
    answer: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class CodeAssistResult:
    """Complete result from code assistance flow."""
    query: str
    project: Optional[str]
    answer: str
    sources: List[str]
    call_chain: List[str]
    retrieval: RetrievalResult
    generation: GenerationResult
    total_duration_seconds: float
    success: bool
    response_id: str


@task(
    name="retrieve_code_context",
    description="Retrieve methods, classes, and event handlers from MongoDB",
    retries=2,
    retry_delay_seconds=5,
    tags=["mongodb", "retrieval"]
)
async def retrieve_code_context_task(
    query: str,
    project: Optional[str] = None,
    method_limit: int = 10,
    class_limit: int = 5,
    event_limit: int = 5,
    include_call_chains: bool = True,
    max_depth: int = 2
) -> Dict[str, Any]:
    """
    Retrieve code context from MongoDB.

    Args:
        query: Natural language query
        project: Optional project filter
        method_limit: Max methods to retrieve
        class_limit: Max classes to retrieve
        event_limit: Max event handlers to retrieve
        include_call_chains: Whether to trace call chains
        max_depth: Call chain traversal depth

    Returns:
        Dict with methods, classes, event_handlers, call_chain
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Retrieving code context for: {query[:50]}...")

    result = RetrievalResult(query=query, project=project)

    try:
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        retriever = CodeRetriever()
        await retriever.initialize()

        methods, classes, event_handlers, call_chain = await retriever.retrieve_comprehensive(
            query=query,
            project=project,
            method_limit=method_limit,
            class_limit=class_limit,
            event_limit=event_limit,
            include_call_chains=include_call_chains,
            max_depth=max_depth
        )

        result.methods_found = len(methods)
        result.classes_found = len(classes)
        result.event_handlers_found = len(event_handlers)
        result.call_chain_length = len(call_chain)
        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Retrieved {result.methods_found} methods, "
            f"{result.classes_found} classes, "
            f"{result.event_handlers_found} event handlers"
        )

        # Create Prefect artifact
        await create_markdown_artifact(
            key="code-retrieval",
            markdown=f"""
## Code Retrieval Results
- **Query**: {query}
- **Project**: {project or 'All'}
- **Methods Found**: {result.methods_found}
- **Classes Found**: {result.classes_found}
- **Event Handlers Found**: {result.event_handlers_found}
- **Call Chain Length**: {result.call_chain_length}
- **Duration**: {result.duration_seconds:.2f}s
            """,
            description="Code context retrieval results"
        )

        return {
            "methods": methods,
            "classes": classes,
            "event_handlers": event_handlers,
            "call_chain": call_chain,
            "result": result
        }

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        result.success = False
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        return {
            "methods": [],
            "classes": [],
            "event_handlers": [],
            "call_chain": [],
            "result": result
        }


@task(
    name="build_context",
    description="Build LLM context from retrieved code entities",
    tags=["context"]
)
async def build_context_task(
    retrieval_data: Dict[str, Any],
    query: str,
    history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Build LLM context from retrieved entities.

    Args:
        retrieval_data: Output from retrieve_code_context_task
        query: Original user query
        history: Optional conversation history

    Returns:
        Dict with context string and prompt
    """
    logger = get_run_logger()
    logger.info("Building context from retrieved entities...")

    from code_assistance_pipeline.services.context_builder import ContextBuilder
    from code_assistance_pipeline.models.query_models import ConversationMessage

    builder = ContextBuilder()

    # Convert history if provided
    conv_history = None
    if history:
        conv_history = [
            ConversationMessage(role=msg["role"], content=msg["content"])
            for msg in history
        ]

    context, sources = builder.build_context(
        methods=retrieval_data["methods"],
        classes=retrieval_data["classes"],
        event_handlers=retrieval_data["event_handlers"],
        call_chain=retrieval_data["call_chain"],
        history=conv_history
    )

    prompt = builder.build_prompt(query=query, context=context)

    estimated_tokens = builder.estimate_tokens(prompt)
    logger.info(f"Built prompt with ~{estimated_tokens} estimated tokens")

    return {
        "context": context,
        "prompt": prompt,
        "sources": sources,
        "estimated_tokens": estimated_tokens
    }


@task(
    name="generate_response",
    description="Generate LLM response for code question",
    retries=3,
    retry_delay_seconds=10,
    tags=["llm", "generation"]
)
async def generate_response_task(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Generate LLM response from prompt.

    Args:
        prompt: Complete prompt with context
        temperature: LLM temperature
        max_tokens: Maximum response tokens

    Returns:
        Dict with answer, token_usage, and timing
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info("Generating LLM response...")

    result = GenerationResult(query="")

    try:
        from code_assistance_pipeline.services.response_generator import ResponseGenerator

        generator = ResponseGenerator(
            temperature=temperature,
            max_tokens=max_tokens
        )
        await generator.initialize()

        answer, token_usage, gen_time = await generator.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        result.answer = answer
        result.prompt_tokens = token_usage.prompt_tokens
        result.completion_tokens = token_usage.completion_tokens
        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Generated response in {result.duration_seconds:.2f}s, "
            f"{result.prompt_tokens + result.completion_tokens} tokens"
        )

        # Create Prefect artifact
        await create_markdown_artifact(
            key="llm-generation",
            markdown=f"""
## LLM Generation Results
- **Prompt Tokens**: {result.prompt_tokens}
- **Completion Tokens**: {result.completion_tokens}
- **Total Tokens**: {result.prompt_tokens + result.completion_tokens}
- **Duration**: {result.duration_seconds:.2f}s
- **Answer Preview**: {answer[:200]}...
            """,
            description="LLM response generation results"
        )

        return {
            "answer": answer,
            "token_usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens
            },
            "result": result
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        result.success = False
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        return {
            "answer": f"Error generating response: {e}",
            "token_usage": {},
            "result": result
        }


@task(
    name="log_interaction",
    description="Log interaction to MongoDB for feedback tracking",
    tags=["mongodb", "logging"]
)
async def log_interaction_task(
    response_id: str,
    query: str,
    answer: str,
    sources: List[str],
    call_chain: List[str],
    project: Optional[str],
    retrieval_time: float,
    generation_time: float,
    total_time: float
) -> bool:
    """
    Log the code assistance interaction to MongoDB.

    Args:
        response_id: Unique response ID
        query: User query
        answer: Generated answer
        sources: Source names
        call_chain: Call chain
        project: Project filter
        retrieval_time: Retrieval duration
        generation_time: Generation duration
        total_time: Total duration

    Returns:
        True if logging succeeded
    """
    logger = get_run_logger()
    logger.info(f"Logging interaction {response_id}")

    try:
        from mongodb import MongoDBService
        import time

        mongodb = MongoDBService.get_instance()
        if not mongodb.is_initialized:
            await mongodb.initialize()

        collection = mongodb.db["code_interactions"]

        document = {
            "response_id": response_id,
            "query": query,
            "answer": answer[:5000],
            "sources": sources,
            "call_chain": call_chain,
            "project": project or "all",
            "retrieval_time_ms": int(retrieval_time * 1000),
            "generation_time_ms": int(generation_time * 1000),
            "total_time_ms": int(total_time * 1000),
            "feedback_received": False,
            "created_at": time.time(),
            "source": "prefect_flow"
        }

        await collection.insert_one(document)
        logger.info("Interaction logged successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to log interaction: {e}")
        return False


@flow(
    name="code-assistance-query",
    description="Complete code assistance RAG pipeline",
    retries=1,
    retry_delay_seconds=30
)
async def code_assist_flow(
    query: str,
    project: Optional[str] = None,
    method_limit: int = 10,
    class_limit: int = 5,
    event_limit: int = 5,
    include_call_chains: bool = True,
    max_depth: int = 2,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Complete code assistance RAG pipeline.

    This flow:
    1. Retrieves relevant code context from MongoDB
    2. Builds structured prompt with context
    3. Generates LLM response
    4. Logs interaction for feedback

    Args:
        query: Natural language question
        project: Optional project filter
        method_limit: Max methods to retrieve
        class_limit: Max classes to retrieve
        event_limit: Max event handlers to retrieve
        include_call_chains: Whether to trace call chains
        max_depth: Call chain traversal depth
        temperature: LLM temperature
        max_tokens: Max response tokens
        history: Conversation history

    Returns:
        Dict with complete code assistance result
    """
    logger = get_run_logger()
    flow_start = time.time()

    import uuid
    response_id = str(uuid.uuid4())

    logger.info(f"Starting code assistance flow for: {query[:50]}...")

    # Step 1: Retrieve code context
    retrieval_data = await retrieve_code_context_task(
        query=query,
        project=project,
        method_limit=method_limit,
        class_limit=class_limit,
        event_limit=event_limit,
        include_call_chains=include_call_chains,
        max_depth=max_depth
    )

    retrieval_result = retrieval_data["result"]

    # Step 2: Build context
    context_data = await build_context_task(
        retrieval_data=retrieval_data,
        query=query,
        history=history
    )

    # Step 3: Generate response
    generation_data = await generate_response_task(
        prompt=context_data["prompt"],
        temperature=temperature,
        max_tokens=max_tokens
    )

    generation_result = generation_data["result"]
    total_duration = time.time() - flow_start

    # Step 4: Log interaction
    source_names = [s.name for s in context_data["sources"]]
    await log_interaction_task(
        response_id=response_id,
        query=query,
        answer=generation_data["answer"],
        sources=source_names,
        call_chain=retrieval_data["call_chain"],
        project=project,
        retrieval_time=retrieval_result.duration_seconds,
        generation_time=generation_result.duration_seconds,
        total_time=total_duration
    )

    # Create final summary artifact
    success = retrieval_result.success and generation_result.success

    await create_markdown_artifact(
        key="flow-summary",
        markdown=f"""
# Code Assistance Flow Complete

## Query
{query}

## Result
- **Status**: {"Success" if success else "Failed"}
- **Response ID**: {response_id}
- **Total Duration**: {total_duration:.2f}s

## Retrieval
- Methods: {retrieval_result.methods_found}
- Classes: {retrieval_result.classes_found}
- Event Handlers: {retrieval_result.event_handlers_found}
- Duration: {retrieval_result.duration_seconds:.2f}s

## Generation
- Prompt Tokens: {generation_result.prompt_tokens}
- Completion Tokens: {generation_result.completion_tokens}
- Duration: {generation_result.duration_seconds:.2f}s

## Answer Preview
{generation_data["answer"][:500]}...
        """,
        description=f"Code assistance for: {query[:50]}..."
    )

    return {
        "success": success,
        "response_id": response_id,
        "query": query,
        "project": project,
        "answer": generation_data["answer"],
        "sources": source_names,
        "call_chain": retrieval_data["call_chain"],
        "token_usage": generation_data["token_usage"],
        "timing": {
            "retrieval_seconds": retrieval_result.duration_seconds,
            "generation_seconds": generation_result.duration_seconds,
            "total_seconds": total_duration
        }
    }


def run_code_assist_flow(
    query: str,
    project: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Convenience function to run the code assistance flow synchronously.

    Example:
        from code_assistance_pipeline.prefect import run_code_assist_flow

        result = run_code_assist_flow(
            query="How does the Save button work?",
            project="Gin"
        )
        print(result["answer"])
    """
    return asyncio.run(code_assist_flow(
        query=query,
        project=project,
        temperature=temperature,
        max_tokens=max_tokens
    ))


if __name__ == "__main__":
    # Test run
    print("Testing Code Assistance Flow...")
    result = run_code_assist_flow(
        query="How does the Save button work?",
        project="Gin"
    )
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer'][:200]}...")
