"""
SQL Extraction Package

This package provides functionality for extracting SQL Server schema information
and storing it in MongoDB for RAG-based text-to-SQL generation.

Modules:
    - config_parser: Parse JSON/XML database configuration files
    - schema_extractor: Extract table schemas, columns, keys, relationships
    - stored_procedure_extractor: Extract stored procedure definitions and parameters
    - procedure_summarizer: Generate LLM summaries for stored procedures
    - schema_summarizer: Generate LLM summaries for table schemas

Key principle: This is Python-only - no HTTP calls needed.
Direct calls to mongodb_service for storage.
"""

from .config_parser import parse_config, parse_cli_args, DatabaseConfig
from .schema_extractor import SchemaExtractor, extract_database, extract_from_config
from .stored_procedure_extractor import extract_stored_procedures
from .procedure_summarizer import ProcedureSummarizer, ProcedureSummary
from .schema_summarizer import SchemaSummarizer, SchemaSummary

__all__ = [
    # Config
    'parse_config',
    'parse_cli_args',
    'DatabaseConfig',
    # Extraction
    'SchemaExtractor',
    'extract_database',
    'extract_from_config',
    'extract_stored_procedures',
    # Summarization
    'ProcedureSummarizer',
    'ProcedureSummary',
    'SchemaSummarizer',
    'SchemaSummary',
]
