"""
Prefect Pipeline Orchestration

This module provides Prefect-based workflow orchestration for:
- SQL RAG Pipeline: Schema extraction, summarization, embedding, and storage
- SQL Security Pipeline: Prompt and SQL validation for security
- Document Pipeline: Document processing, chunking, and embedding
- Document Q&A Pipeline: RAG-based document retrieval and answer generation
- Audio Pipeline: Audio transcription with emotion detection
- Agent Learning Pipeline: Agent learning statistics, accuracy analysis, and corrections
- Ticket Matching Pipeline: Semantic audio-to-ticket matching with multi-agent architecture
- Scheduled Flows: Automated schema extraction on schedules

Prefect provides:
- Built-in retries and error handling
- Visual dashboard for monitoring at http://localhost:4200
- Task dependencies and parallel execution
- Artifact tracking and logging
- Native async/await support
- Scheduled flow execution

Usage:
    from prefect_pipelines import (
        run_sql_rag_flow,
        run_document_flow,
        run_document_qa_flow,
        run_audio_flow,
        run_security_validation,
        run_ticket_matching_flow
    )

    # Run SQL RAG pipeline
    result = run_sql_rag_flow(
        server="NCSQLTEST",
        database="EWRReporting",
        user="EWRUser",
        password="password"
    )

    # Run security validation
    result = run_security_validation(
        user_prompt="show me all tickets from last month",
        source="sql-query-ui"
    )

    # Run agent learning analysis
    result = asyncio.run(run_agent_learning_flow())

    # Run ticket matching for audio analysis
    result = run_ticket_matching_flow(
        analysis_id="abc123",
        use_summary=True,
        auto_link=False
    )

    # Run Document Q&A pipeline (RAG-based Q&A)
    result = run_document_qa_flow(
        query="How do I configure MongoDB?",
        user_id="user123",
        top_k=5,
        use_prefect=True
    )

    # Run with Prefect tracking (async)
    import asyncio
    result = asyncio.run(sql_rag_flow(server, database, config))
"""

# SQL-related flows moved to sql_pipeline.prefect
from sql_pipeline.prefect.sql_rag_flow import (
    sql_rag_flow,
    run_sql_rag_flow,
    extract_schemas_task,
    summarize_schemas_task,
    generate_embeddings_task,
    store_vectors_task
)

# SQL Security flow (may not be available if sql_security_service is missing)
try:
    from sql_pipeline.prefect.sql_security_flow import (
        sql_security_validation_flow,
        run_security_validation,
        validate_prompt_security,
        validate_sql_security,
        full_security_check,
        log_security_event,
        SecurityValidationResult
    )
    _sql_security_available = True
except ImportError:
    _sql_security_available = False

# Document flow - thin wrapper around actual services
try:
    from prefect_pipelines.document_flow import (
        document_flow,
        run_document_flow,
    )
    _document_flow_available = True
except ImportError:
    _document_flow_available = False

# Audio flow - thin wrapper around actual AudioAnalysisService
try:
    from prefect_pipelines.audio_flow import (
        audio_flow,
        run_audio_flow,
    )
    _audio_flow_available = True
except ImportError:
    _audio_flow_available = False

# SQL Query flow - thin wrapper around actual QueryPipeline
try:
    from prefect_pipelines.sql_query_flow import (
        sql_query_flow,
    )
    _sql_query_flow_available = True
except ImportError:
    _sql_query_flow_available = False

# Agent Learning flow
try:
    from prefect_pipelines.agent_learning_flow import (
        run_agent_learning_flow,
        collect_learning_stats_task,
        analyze_accuracy_task,
        process_corrections_task,
        generate_report_task
    )
    _agent_learning_available = True
except ImportError:
    _agent_learning_available = False

# Scheduled flows (import later to avoid circular deps)
try:
    from prefect_pipelines.scheduled_flows import (
        scheduled_schema_extraction_flow,
        run_scheduled_extraction,
        create_daily_schedule,
        create_weekly_schedule
    )
    _scheduled_available = True
except ImportError:
    _scheduled_available = False

# RAG Metrics flow for observability
try:
    from prefect_pipelines.rag_metrics_flow import (
        rag_query_flow,
        run_rag_query_flow,
        sql_query_metrics_flow,
        run_sql_query_metrics_flow,
        daily_summary_flow,
        track_vector_search,
        track_llm_generation,
        store_metrics_to_mongodb,
        VectorSearchMetrics,
        LLMGenerationMetrics,
        RAGQueryMetrics,
    )
    _rag_metrics_available = True
except ImportError:
    _rag_metrics_available = False

# Git Analysis flow
try:
    from prefect_pipelines.git_flow import (
        git_analysis_flow,
        run_git_analysis_flow,
        pull_repository_task,
        analyze_commits_task,
    )
    _git_flow_available = True
except ImportError:
    _git_flow_available = False

# Ticket Matching flow for semantic audio-to-ticket matching
try:
    from prefect_pipelines.ticket_matching_flow import (
        ticket_matching_flow,
        run_ticket_matching_flow,
        retrieve_analysis_task,
        query_candidates_task,
        compute_embeddings_task,
        score_candidates_task,
        store_history_task
    )
    _ticket_matching_available = True
except ImportError:
    _ticket_matching_available = False

# Document Q&A flow for RAG-based document query and answer generation
try:
    from prefect_pipelines.document_qa_flow import (
        document_qa_flow,
        run_document_qa_flow,
        run_document_qa_flow_async,
        query_understanding_task,
        hybrid_retrieval_task,
        document_grading_task,
        answer_generation_task,
        validation_task,
        learning_feedback_task,
        QueryUnderstandingResult,
        HybridRetrievalResult,
        DocumentGradingResult,
        AnswerGenerationResult,
        ValidationResult,
        LearningFeedbackResult,
    )
    _document_qa_available = True
except ImportError:
    _document_qa_available = False

__all__ = [
    # SQL RAG
    "sql_rag_flow",
    "run_sql_rag_flow",
    "extract_schemas_task",
    "summarize_schemas_task",
    "generate_embeddings_task",
    "store_vectors_task",
    # SQL Security
    "sql_security_validation_flow",
    "run_security_validation",
    "validate_prompt_security",
    "validate_sql_security",
    "full_security_check",
    "log_security_event",
    "SecurityValidationResult",
    # Document (thin wrapper around actual services)
    "document_flow",
    "run_document_flow",
    # Audio (thin wrapper around actual AudioAnalysisService)
    "audio_flow",
    "run_audio_flow",
    # SQL Query (thin wrapper around actual QueryPipeline)
    "sql_query_flow",
    # Agent Learning (conditionally available)
    "run_agent_learning_flow",
    "collect_learning_stats_task",
    "analyze_accuracy_task",
    "process_corrections_task",
    "generate_report_task",
    # Scheduled (conditionally available)
    "scheduled_schema_extraction_flow",
    "run_scheduled_extraction",
    "create_daily_schedule",
    "create_weekly_schedule",
    # RAG Metrics (conditionally available)
    "rag_query_flow",
    "run_rag_query_flow",
    "sql_query_metrics_flow",
    "run_sql_query_metrics_flow",
    "daily_summary_flow",
    "track_vector_search",
    "track_llm_generation",
    "store_metrics_to_mongodb",
    "VectorSearchMetrics",
    "LLMGenerationMetrics",
    "RAGQueryMetrics",
    # Ticket Matching (conditionally available)
    "ticket_matching_flow",
    "run_ticket_matching_flow",
    "retrieve_analysis_task",
    "query_candidates_task",
    "compute_embeddings_task",
    "score_candidates_task",
    "store_history_task",
    # Document Q&A (conditionally available)
    "document_qa_flow",
    "run_document_qa_flow",
    "run_document_qa_flow_async",
    "query_understanding_task",
    "hybrid_retrieval_task",
    "document_grading_task",
    "answer_generation_task",
    "validation_task",
    "learning_feedback_task",
    "QueryUnderstandingResult",
    "HybridRetrievalResult",
    "DocumentGradingResult",
    "AnswerGenerationResult",
    "ValidationResult",
    "LearningFeedbackResult",
]
