"""
LLM Integration Helpers for Pipeline Use.

Provides easy-to-use helper functions for integrating TracedLLMClient
into all 7 pipelines. Each helper function handles:
- Client instantiation
- Context building
- Error handling
- Trace storage

Usage in pipelines:
    from llm.integration import generate_sql, generate_text

    # SQL generation with tracing
    result = await generate_sql(
        prompt="SELECT query here",
        user_id="admin",
        database="EWRCentral",
        tables=["CentralTickets", "Types"],
    )

    # General text generation
    result = await generate_text(
        prompt="Summarize this...",
        operation="summarize",
        pipeline="audio",
    )
"""
import os
from typing import Optional, List, Dict, Any

from .service import LLMService
from .models import Pipeline, TraceContext, LLMResponse

# Default MongoDB URI for the LLM service (reads from environment)
DEFAULT_MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

# Singleton service instance
_service: Optional[LLMService] = None


def get_llm_service(mongodb_uri: str = DEFAULT_MONGODB_URI) -> LLMService:
    """Get or create the LLM service singleton."""
    global _service
    if _service is None:
        _service = LLMService.get_instance(mongodb_uri)
    return _service


def reset_llm_service():
    """Reset the service singleton (for testing)."""
    global _service
    _service = None
    LLMService.reset_instance()


# =============================================================================
# SQL Pipeline Helpers
# =============================================================================

async def generate_sql(
    prompt: str,
    user_id: str = None,
    session_id: str = None,
    database: str = None,
    tables: List[str] = None,
    user_question: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    stop: List[str] = None,
    tags: List[str] = None,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Generate SQL with automatic tracing.

    Args:
        prompt: Full LLM prompt with schema context and examples
        user_id: User ID for tracking
        session_id: Session ID for tracking
        database: Target database name
        tables: List of tables referenced in query
        user_question: Original natural language question
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        stop: Stop sequences
        tags: Additional tags for filtering
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with generated SQL and metrics
    """
    service = get_llm_service(mongodb_uri)
    client = service.get_sql_client()

    response = await client.generate(
        prompt=prompt,
        operation="generate_sql",
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or ["```", "\n\n\n"],
        context=TraceContext(
            user_id=user_id,
            session_id=session_id,
            database=database,
            tables_used=tables or [],
            user_question=user_question,
        ),
        tags=tags or ["sql_generation"],
    )
    return response


async def validate_sql_with_llm(
    sql: str,
    error_message: str,
    database: str = None,
    user_id: str = None,
    max_tokens: int = 1024,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Use LLM to validate/fix SQL based on error.

    Args:
        sql: The SQL that failed
        error_message: The error that occurred
        database: Target database name
        user_id: User ID for tracking
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with fixed SQL
    """
    prompt = f"""The following SQL query failed with an error:

SQL:
{sql}

Error:
{error_message}

Please provide a corrected SQL query that fixes this error. Only output the SQL, no explanation."""

    service = get_llm_service(mongodb_uri)
    client = service.get_sql_client()

    return await client.generate(
        prompt=prompt,
        operation="validate_sql",
        max_tokens=max_tokens,
        temperature=0.1,
        context=TraceContext(
            user_id=user_id,
            database=database,
        ),
        tags=["sql_validation", "auto_fix"],
    )


# =============================================================================
# Audio Pipeline Helpers
# =============================================================================

async def summarize_transcription(
    transcription: str,
    user_id: str = None,
    session_id: str = None,
    call_id: str = None,
    max_tokens: int = 512,
    temperature: float = 0.5,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Summarize audio transcription with tracing.

    Args:
        transcription: The transcription text to summarize
        user_id: User ID for tracking
        session_id: Session ID for tracking
        call_id: Call/audio file ID
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with summary
    """
    prompt = f"""Please summarize the following call transcription concisely:

{transcription}

Summary:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_audio_client()

    return await client.generate(
        prompt=prompt,
        operation="summarize_transcription",
        max_tokens=max_tokens,
        temperature=temperature,
        context=TraceContext(
            user_id=user_id,
            session_id=session_id,
            document_id=call_id,
        ),
        tags=["audio", "summarization"],
    )


async def analyze_call_content(
    transcription: str,
    analysis_type: str = "general",
    user_id: str = None,
    call_id: str = None,
    max_tokens: int = 1024,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Analyze call content for sentiment, topics, or issues.

    Args:
        transcription: The transcription to analyze
        analysis_type: Type of analysis (general, sentiment, topics, issues)
        user_id: User ID for tracking
        call_id: Call/audio file ID
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with analysis
    """
    prompt = f"""Analyze the following call transcription for {analysis_type}:

{transcription}

Analysis:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_audio_client()

    return await client.generate(
        prompt=prompt,
        operation=f"analyze_{analysis_type}",
        max_tokens=max_tokens,
        temperature=0.3,
        context=TraceContext(
            user_id=user_id,
            document_id=call_id,
        ),
        tags=["audio", "analysis", analysis_type],
    )


# =============================================================================
# Query/RAG Pipeline Helpers
# =============================================================================

async def generate_rag_response(
    query: str,
    context: str,
    user_id: str = None,
    session_id: str = None,
    project: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Generate RAG response based on retrieved context.

    Args:
        query: User's question
        context: Retrieved context from vector search
        user_id: User ID for tracking
        session_id: Session ID for tracking
        project: Project name for scoping
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with answer
    """
    prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_query_client()

    return await client.generate(
        prompt=prompt,
        operation="rag_response",
        max_tokens=max_tokens,
        temperature=temperature,
        context=TraceContext(
            user_id=user_id,
            session_id=session_id,
            user_question=query,
            database=project,  # Using database field for project
        ),
        tags=["rag", "query"],
    )


# =============================================================================
# Code Flow Pipeline Helpers
# =============================================================================

async def analyze_code_flow(
    code: str,
    method_name: str = None,
    user_id: str = None,
    project: str = None,
    max_tokens: int = 2048,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Analyze code flow and call chains.

    Args:
        code: The code to analyze
        method_name: Target method name
        user_id: User ID for tracking
        project: Project name
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with code flow analysis
    """
    prompt = f"""Analyze the following code and describe its flow:

{code}

{"Focus on method: " + method_name if method_name else ""}

Analysis:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_code_flow_client()

    return await client.generate(
        prompt=prompt,
        operation="code_flow_analysis",
        max_tokens=max_tokens,
        temperature=0.3,
        context=TraceContext(
            user_id=user_id,
            database=project,
        ),
        tags=["code_flow", "analysis"],
    )


# =============================================================================
# Code Assistance Pipeline Helpers
# =============================================================================

async def generate_code_completion(
    prompt: str,
    language: str = "csharp",
    user_id: str = None,
    project: str = None,
    file_path: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Generate code completion or suggestion.

    Args:
        prompt: Code context and request
        language: Programming language
        user_id: User ID for tracking
        project: Project name
        file_path: Source file path
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with code completion
    """
    service = get_llm_service(mongodb_uri)
    client = service.get_code_assistance_client()

    return await client.generate(
        prompt=prompt,
        operation="code_completion",
        max_tokens=max_tokens,
        temperature=temperature,
        context=TraceContext(
            user_id=user_id,
            database=project,
        ),
        tags=["code_assistance", language],
    )


async def explain_code(
    code: str,
    user_id: str = None,
    project: str = None,
    max_tokens: int = 1024,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Explain code functionality.

    Args:
        code: Code to explain
        user_id: User ID for tracking
        project: Project name
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with explanation
    """
    prompt = f"""Explain the following code in detail:

{code}

Explanation:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_code_assistance_client()

    return await client.generate(
        prompt=prompt,
        operation="explain_code",
        max_tokens=max_tokens,
        temperature=0.5,
        context=TraceContext(
            user_id=user_id,
            database=project,
        ),
        tags=["code_assistance", "explanation"],
    )


# =============================================================================
# Git Pipeline Helpers
# =============================================================================

async def analyze_git_diff(
    diff: str,
    user_id: str = None,
    repository: str = None,
    max_tokens: int = 1024,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Analyze git diff for changes.

    Args:
        diff: Git diff content
        user_id: User ID for tracking
        repository: Repository name
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with diff analysis
    """
    prompt = f"""Analyze the following git diff and summarize the changes:

{diff}

Summary:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_git_client()

    return await client.generate(
        prompt=prompt,
        operation="analyze_diff",
        max_tokens=max_tokens,
        temperature=0.3,
        context=TraceContext(
            user_id=user_id,
            database=repository,
        ),
        tags=["git", "diff_analysis"],
    )


async def generate_commit_message(
    diff: str,
    user_id: str = None,
    repository: str = None,
    max_tokens: int = 256,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Generate commit message from diff.

    Args:
        diff: Git diff content
        user_id: User ID for tracking
        repository: Repository name
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with commit message
    """
    prompt = f"""Based on the following git diff, generate a concise commit message following conventional commits format:

{diff}

Commit message:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_git_client()

    return await client.generate(
        prompt=prompt,
        operation="generate_commit_message",
        max_tokens=max_tokens,
        temperature=0.3,
        context=TraceContext(
            user_id=user_id,
            database=repository,
        ),
        tags=["git", "commit_message"],
    )


# =============================================================================
# Document Agent Helpers
# =============================================================================

async def process_document(
    content: str,
    operation: str = "summarize",
    user_id: str = None,
    document_id: str = None,
    max_tokens: int = 1024,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Process document content (summarize, extract, etc.).

    Args:
        content: Document content
        operation: Type of processing (summarize, extract, qa)
        user_id: User ID for tracking
        document_id: Document ID
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with processed content
    """
    if operation == "summarize":
        prompt = f"""Summarize the following document:

{content}

Summary:"""
    elif operation == "extract":
        prompt = f"""Extract key information from the following document:

{content}

Key Information:"""
    else:
        prompt = f"""Process the following document:

{content}

Output:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_document_client()

    return await client.generate(
        prompt=prompt,
        operation=f"document_{operation}",
        max_tokens=max_tokens,
        temperature=0.5,
        context=TraceContext(
            user_id=user_id,
            document_id=document_id,
        ),
        tags=["document_agent", operation],
    )


async def answer_document_question(
    question: str,
    context: str,
    user_id: str = None,
    document_id: str = None,
    max_tokens: int = 1024,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Answer question about document.

    Args:
        question: User's question
        context: Document context
        user_id: User ID for tracking
        document_id: Document ID
        max_tokens: Maximum tokens to generate
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with answer
    """
    prompt = f"""Based on the following document content, answer the question.

Document:
{context}

Question: {question}

Answer:"""

    service = get_llm_service(mongodb_uri)
    client = service.get_document_client()

    return await client.generate(
        prompt=prompt,
        operation="document_qa",
        max_tokens=max_tokens,
        temperature=0.5,
        context=TraceContext(
            user_id=user_id,
            document_id=document_id,
            user_question=question,
        ),
        tags=["document_agent", "qa"],
    )


# =============================================================================
# Generic Text Generation
# =============================================================================

async def generate_text(
    prompt: str,
    operation: str = "generate",
    pipeline: str = "general",
    user_id: str = None,
    session_id: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stop: List[str] = None,
    tags: List[str] = None,
    context_dict: Dict[str, Any] = None,
    mongodb_uri: str = DEFAULT_MONGODB_URI,
) -> LLMResponse:
    """
    Generic text generation with tracing.

    Args:
        prompt: Input prompt
        operation: Operation name for tracing
        pipeline: Pipeline name (sql, audio, query, git, code_flow, code_assistance, document_agent)
        user_id: User ID for tracking
        session_id: Session ID for tracking
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop: Stop sequences
        tags: Additional tags
        context_dict: Additional context as dict
        mongodb_uri: MongoDB connection string

    Returns:
        LLMResponse with generated text
    """
    pipeline_map = {
        "sql": Pipeline.SQL,
        "audio": Pipeline.AUDIO,
        "query": Pipeline.QUERY,
        "git": Pipeline.GIT,
        "code_flow": Pipeline.CODE_FLOW,
        "code_assistance": Pipeline.CODE_ASSISTANCE,
        "document_agent": Pipeline.DOCUMENT_AGENT,
        "general": Pipeline.QUERY,  # Default to query/general
    }

    pipeline_enum = pipeline_map.get(pipeline.lower(), Pipeline.QUERY)

    service = get_llm_service(mongodb_uri)
    client = service.get_client(pipeline_enum)

    # Build context
    ctx = TraceContext(
        user_id=user_id,
        session_id=session_id,
    )

    # Add additional context if provided
    if context_dict:
        if "database" in context_dict:
            ctx.database = context_dict["database"]
        if "tables_used" in context_dict:
            ctx.tables_used = context_dict["tables_used"]
        if "document_id" in context_dict:
            ctx.document_id = context_dict["document_id"]
        if "user_question" in context_dict:
            ctx.user_question = context_dict["user_question"]

    return await client.generate(
        prompt=prompt,
        operation=operation,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        context=ctx,
        tags=tags or [],
    )
