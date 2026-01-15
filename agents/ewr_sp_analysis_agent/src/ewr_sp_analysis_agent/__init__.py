"""
EWR Stored Procedure Analysis Agent
===================================

Specialized agent for analyzing stored procedures and generating NL->SQL training data.

This agent extracts stored procedures from databases, uses LLMs to generate natural
language questions that would produce the stored procedure's output, and validates
the generated questions by comparing query results.

Capabilities:
- SP_EXTRACT: Extract stored procedures from databases
- NL_GENERATE: Generate natural language questions from SP analysis
- QUERY_TEST: Test generated questions against actual SP results
- VALIDATE: Validate question-to-result alignment
- TRAINING_EXPORT: Export validated examples as training data

Usage:
    from ewr_sp_analysis_agent import SPAnalysisAgent, SPAnalysisConfig

    config = SPAnalysisConfig(
        database="EWRCentral",
        server="localhost"
    )

    agent = SPAnalysisAgent(config=config)
    await agent.start()

    # Analyze a stored procedure
    result = await agent.analyze_procedure("usp_GetTicketsByDate")

    # Generate training examples from multiple procedures
    examples = await agent.batch_analyze(procedures=["usp_GetTickets", "usp_GetUsers"])
"""

from .agent import SPAnalysisAgent, StoredProcedure
from .models import (
    SPAnalysisConfig,
    GeneratedQuestion,
    TestQuery,
    ValidationResult,
    SPAnalysisResult,
    TrainingExample,
    BatchAnalysisResult,
    QuestionDifficulty,
    ValidationStatus,
)

__version__ = "1.0.0"

__all__ = [
    "SPAnalysisAgent",
    "StoredProcedure",
    "SPAnalysisConfig",
    "GeneratedQuestion",
    "TestQuery",
    "ValidationResult",
    "SPAnalysisResult",
    "TrainingExample",
    "BatchAnalysisResult",
    "QuestionDifficulty",
    "ValidationStatus",
]
