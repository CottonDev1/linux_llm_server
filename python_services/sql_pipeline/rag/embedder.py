"""
Embedding generators for SQL RAG pipeline

Following nilenso best practices:
- Embed LLM summaries, not raw SQL code
- Include keywords, operations, and relationships in embedding text
- Sample values improve accuracy by 3-4%
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding_text: str
    vector: List[float]


class SchemaEmbedder:
    """
    Generate embeddings for table schemas.

    Key insight: Create rich, human-readable text from schema + summary
    that captures the table's purpose and relationships.
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def format_embedding_text(
        self,
        table_name: str,
        columns: List[Dict],
        primary_keys: List[str],
        foreign_keys: List[Dict],
        related_tables: List[str],
        sample_values: Dict,
        summary: Optional[str] = None,
        purpose: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> str:
        """
        Format schema for embedding following nilenso best practices.

        Priority order for embedding text:
        1. LLM summary (most important for semantic search)
        2. Purpose description
        3. Keywords
        4. Column names with types
        5. Relationships
        6. Sample values
        """
        parts = []

        # Table identification
        parts.append(f"Table: {table_name}")

        # LLM-generated summary (PRIMARY for semantic matching)
        if summary:
            parts.append(f"Description: {summary}")

        # Purpose (if different from summary)
        if purpose and purpose != summary:
            parts.append(f"Purpose: {purpose}")

        # Keywords for searchability
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")

        # Columns with types (concise format)
        if columns:
            col_descriptions = []
            for col in columns:
                col_name = col.get('name', '')
                col_type = col.get('type', '')
                col_desc = f"{col_name} ({col_type})"
                col_descriptions.append(col_desc)
            parts.append("Columns: " + ", ".join(col_descriptions))

        # Primary keys
        if primary_keys:
            parts.append(f"Primary Key: {', '.join(primary_keys)}")

        # Foreign key relationships (critical for JOIN queries)
        if foreign_keys:
            fk_parts = []
            for fk in foreign_keys:
                ref_table = fk.get('referencedTable', fk.get('referenced_table', ''))
                ref_col = fk.get('referencedColumn', fk.get('referenced_column', ''))
                fk_col = fk.get('column', fk.get('columnName', ''))
                if ref_table and fk_col:
                    fk_parts.append(f"{fk_col} -> {ref_table}.{ref_col}")
            if fk_parts:
                parts.append("Relationships: " + ", ".join(fk_parts))

        # Related tables
        if related_tables:
            parts.append(f"Related Tables: {', '.join(related_tables)}")

        # Sample values (nilenso: 3-4% accuracy improvement)
        if sample_values:
            sample_parts = []
            for col, values in sample_values.items():
                if values and isinstance(values, list):
                    # Take first 3 sample values
                    sample_str = ', '.join(str(v) for v in values[:3])
                    sample_parts.append(f"{col}: {sample_str}")
            if sample_parts:
                parts.append("Sample Values: " + "; ".join(sample_parts[:5]))

        return "\n".join(parts)

    async def generate_embedding(
        self,
        table_name: str,
        columns: List[Dict],
        primary_keys: List[str] = None,
        foreign_keys: List[Dict] = None,
        related_tables: List[str] = None,
        sample_values: Dict = None,
        summary: str = None,
        purpose: str = None,
        keywords: List[str] = None
    ) -> EmbeddingResult:
        """Generate embedding for a table schema."""

        embedding_text = self.format_embedding_text(
            table_name=table_name,
            columns=columns,
            primary_keys=primary_keys or [],
            foreign_keys=foreign_keys or [],
            related_tables=related_tables or [],
            sample_values=sample_values or {},
            summary=summary,
            purpose=purpose,
            keywords=keywords
        )

        vector = await self.embedding_service.generate_embedding(embedding_text)

        return EmbeddingResult(
            embedding_text=embedding_text,
            vector=vector
        )


class ProcedureEmbedder:
    """
    Generate embeddings for stored procedures.

    Key insight: Embed summary/purpose, NOT the raw SQL code.
    Raw SQL is stored for reference but not used in semantic search.
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def format_embedding_text(
        self,
        procedure_name: str,
        schema: str = 'dbo',
        parameters: List[Dict] = None,
        summary: str = None,
        operations: List[str] = None,
        tables_referenced: List[str] = None,
        keywords: List[str] = None,
        input_description: str = None,
        output_description: str = None
    ) -> str:
        """
        Format stored procedure for embedding.

        Priority order:
        1. LLM summary (most important)
        2. Input/output descriptions
        3. Operations performed
        4. Tables referenced
        5. Keywords
        6. Parameter names
        """
        parts = []

        # Procedure identification
        parts.append(f"Stored Procedure: {schema}.{procedure_name}")

        # LLM-generated summary (PRIMARY for semantic matching)
        if summary:
            parts.append(f"Description: {summary}")

        # Input description
        if input_description:
            parts.append(f"Input: {input_description}")

        # Output description
        if output_description:
            parts.append(f"Output: {output_description}")

        # Operations performed (SELECT, INSERT, UPDATE, DELETE, etc.)
        if operations:
            parts.append(f"Operations: {', '.join(operations)}")

        # Tables referenced
        if tables_referenced:
            parts.append(f"Tables: {', '.join(tables_referenced)}")

        # Keywords for searchability
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")

        # Parameters (concise format - just names and types)
        if parameters:
            param_parts = []
            for p in parameters:
                param_name = p.get('name', p.get('parameter_name', ''))
                param_type = p.get('type', p.get('data_type', ''))
                if param_name:
                    param_parts.append(f"{param_name} ({param_type})")
            if param_parts:
                parts.append(f"Parameters: {', '.join(param_parts)}")

        return "\n".join(parts)

    async def generate_embedding(
        self,
        procedure_name: str,
        schema: str = 'dbo',
        parameters: List[Dict] = None,
        summary: str = None,
        operations: List[str] = None,
        tables_referenced: List[str] = None,
        keywords: List[str] = None,
        input_description: str = None,
        output_description: str = None
    ) -> EmbeddingResult:
        """Generate embedding for a stored procedure."""

        embedding_text = self.format_embedding_text(
            procedure_name=procedure_name,
            schema=schema,
            parameters=parameters or [],
            summary=summary,
            operations=operations or [],
            tables_referenced=tables_referenced or [],
            keywords=keywords or [],
            input_description=input_description,
            output_description=output_description
        )

        vector = await self.embedding_service.generate_embedding(embedding_text)

        return EmbeddingResult(
            embedding_text=embedding_text,
            vector=vector
        )
