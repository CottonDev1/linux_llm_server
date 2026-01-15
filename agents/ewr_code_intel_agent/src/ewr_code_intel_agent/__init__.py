"""
EWR Code Intelligence Agent
===========================

Deep code analysis agent with self-validating knowledge.

This is the most important agent in the system. It:
1. Analyzes codebases to build comprehensive knowledge graphs
2. Creates call graphs and data flow traces
3. Self-validates its knowledge by querying the Code Agent
4. Refines embeddings until answers are accurate
5. Answers developer questions about unfamiliar codebases

Capabilities:
- Deep repository analysis
- Call graph and dependency graph building
- Data flow tracing
- Entry point identification
- Workflow explanation
- Self-validating embeddings
- Developer question answering

Usage:
    from ewr_code_intel_agent import CodeIntelAgent

    agent = CodeIntelAgent(name="code-intel")
    await agent.start()

    # Analyze a repository
    result = await agent.analyze_repository("/path/to/repo")

    # Build call graph
    graph = await agent.build_call_graph("MyClass.MyMethod")

    # Ask questions
    answer = await agent.answer_question("How do I create a shipping order?")

    # Self-validate knowledge
    validation = await agent.validate_knowledge([
        "What does the OrderService do?",
        "How is authentication handled?"
    ])
"""

from .agent import CodeIntelAgent
from .models import (
    AnalysisResult,
    CallGraph,
    CallNode,
    DataFlowTrace,
    EntryPoint,
    WorkflowExplanation,
    ValidationResult,
    KnowledgeGap,
    DeveloperAnswer,
    CodeChunk,
    EmbeddingResult,
)

__version__ = "1.0.0"

__all__ = [
    "CodeIntelAgent",
    "AnalysisResult",
    "CallGraph",
    "CallNode",
    "DataFlowTrace",
    "EntryPoint",
    "WorkflowExplanation",
    "ValidationResult",
    "KnowledgeGap",
    "DeveloperAnswer",
    "CodeChunk",
    "EmbeddingResult",
]
