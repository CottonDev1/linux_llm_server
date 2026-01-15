"""
Code Assistance Services
========================

Service modules for the code assistance pipeline.

Components:
- code_retriever: Searches MongoDB for code entities
- context_builder: Assembles context from retrieved entities
- response_generator: Generates LLM responses with streaming support
"""

from code_assistance_pipeline.services.code_retriever import CodeRetriever
from code_assistance_pipeline.services.context_builder import ContextBuilder
from code_assistance_pipeline.services.response_generator import ResponseGenerator

__all__ = [
    "CodeRetriever",
    "ContextBuilder",
    "ResponseGenerator",
]
