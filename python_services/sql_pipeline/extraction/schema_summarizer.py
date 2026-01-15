"""
Schema Summarizer

Uses LLM to generate human-readable summaries for table schemas.
This is a key nilenso recommendation - embedding table descriptions rather than raw DDL.

The summary describes:
- What the table stores (business purpose)
- Key columns and their meaning
- Relationships to other tables
- Common query patterns

Usage:
    from sql_pipeline.extraction.schema_summarizer import SchemaSummarizer

    summarizer = SchemaSummarizer(llm_url="http://localhost:11434")
    summary = await summarizer.summarize_schema(table_name, schema_info)
"""

import httpx
import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SchemaSummary:
    """Summary generated for a table schema"""
    table_name: str
    summary: str
    purpose: str  # Business purpose of the table
    key_columns: List[str]  # Most important columns
    relationships: List[str]  # FK relationships described
    keywords: List[str]  # Business keywords for search
    common_queries: List[str]  # Suggested query patterns


class SchemaSummarizer:
    """
    Generates LLM summaries for table schemas.
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 60
    ):
        self.llm_url = llm_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    async def summarize_schema(
        self,
        table_name: str,
        schema_info: Dict[str, Any],
        verbose: bool = True
    ) -> SchemaSummary:
        """
        Generate a summary for a table schema.

        Args:
            table_name: Full table name (schema.table)
            schema_info: Dict with columns, primaryKeys, foreignKeys, relatedTables
            verbose: Print progress

        Returns:
            SchemaSummary with natural language description
        """
        columns = schema_info.get('columns', [])
        primary_keys = schema_info.get('primaryKeys', [])
        foreign_keys = schema_info.get('foreignKeys', [])
        related_tables = schema_info.get('relatedTables', [])
        sample_values = schema_info.get('sampleValues', {})

        # Format schema for prompt
        schema_text = self._format_schema(
            table_name, columns, primary_keys, foreign_keys, related_tables, sample_values
        )

        # Create prompt for LLM
        prompt = self._build_prompt(table_name, schema_text)

        if verbose:
            print(f"   Summarizing {table_name}...")

        try:
            # Call LLM API
            response = await self._call_llm(prompt)

            # Parse the response
            summary = self._parse_response(response, table_name, columns, foreign_keys)

            return summary

        except Exception as e:
            if verbose:
                print(f"   Failed to summarize {table_name}: {e}")

            # Return basic summary on failure
            return SchemaSummary(
                table_name=table_name,
                summary=f"Table {table_name} with {len(columns)} columns",
                purpose="Stores data",
                key_columns=[pk for pk in primary_keys[:3]],
                relationships=[fk.get('referencedTable', '') for fk in foreign_keys[:5]],
                keywords=self._extract_keywords_from_name(table_name),
                common_queries=[]
            )

    async def summarize_batch(
        self,
        schemas: List[Dict[str, Any]],
        database: str,
        verbose: bool = True,
        concurrency: int = 3
    ) -> List[SchemaSummary]:
        """
        Summarize multiple schemas with controlled concurrency.

        Args:
            schemas: List of schema dicts with table_name, schema_info
            database: Database name for context
            verbose: Print progress
            concurrency: Max concurrent LLM calls

        Returns:
            List of SchemaSummary objects
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def summarize_with_limit(schema):
            async with semaphore:
                return await self.summarize_schema(
                    table_name=schema['table_name'],
                    schema_info=schema['schema_info'],
                    verbose=verbose
                )

        tasks = [summarize_with_limit(s) for s in schemas]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [r for r in results if isinstance(r, SchemaSummary)]

    def _format_schema(
        self,
        table_name: str,
        columns: List[Dict],
        primary_keys: List[str],
        foreign_keys: List[Dict],
        related_tables: List[str],
        sample_values: Dict[str, List]
    ) -> str:
        """Format schema as readable text for the LLM"""
        lines = [f"TABLE: {table_name}", ""]

        # Columns
        lines.append("COLUMNS:")
        for col in columns:
            pk_marker = " [PK]" if col['name'] in primary_keys else ""
            nullable = " (nullable)" if col.get('nullable') else ""
            lines.append(f"  - {col['name']}: {col['type']}{pk_marker}{nullable}")

        # Primary Keys
        if primary_keys:
            lines.append(f"\nPRIMARY KEY: {', '.join(primary_keys)}")

        # Foreign Keys
        if foreign_keys:
            lines.append("\nFOREIGN KEYS:")
            for fk in foreign_keys:
                lines.append(f"  - {fk['column']} -> {fk['referencedTable']}.{fk['referencedColumn']}")

        # Related Tables
        if related_tables:
            lines.append(f"\nRELATED TABLES: {', '.join(related_tables)}")

        # Sample Values (if available)
        if sample_values:
            lines.append("\nSAMPLE VALUES:")
            for col, values in list(sample_values.items())[:5]:
                vals = ', '.join(str(v) for v in values[:3])
                lines.append(f"  - {col}: {vals}")

        return "\n".join(lines)

    def _build_prompt(self, table_name: str, schema_text: str) -> str:
        """Build the LLM prompt for summarization"""
        return f"""Analyze this SQL Server table schema and provide a concise summary for use in text-to-SQL systems.

{schema_text}

Respond in this exact format (keep each section brief):

SUMMARY: [One sentence describing what this table stores and its business purpose]
PURPOSE: [2-3 sentences explaining when/why this table is used]
KEY_COLUMNS: [List the 3-5 most important columns for querying, comma-separated]
RELATIONSHIPS: [Describe the foreign key relationships in plain English]
KEYWORDS: [10-15 business keywords someone might search for, comma-separated]
QUERIES: [2-3 common query patterns using this table, described in plain English]"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API to generate summary"""
        # Use explicit timeout configuration for reliability
        timeout_config = httpx.Timeout(
            connect=10.0,      # Connection timeout
            read=120.0,        # Read timeout (LLM can be slow)
            write=10.0,        # Write timeout
            pool=10.0          # Pool timeout
        )
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{self.llm_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 600,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI format: choices[0].text
            return data.get('choices', [{}])[0].get('text', '')

    def _parse_response(
        self,
        response: str,
        table_name: str,
        columns: List[Dict],
        foreign_keys: List[Dict]
    ) -> SchemaSummary:
        """Parse LLM response into structured summary"""

        def extract_field(text: str, field: str) -> str:
            pattern = rf'{field}:\s*(.+?)(?=\n[A-Z_]+:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        def extract_list(text: str, field: str) -> List[str]:
            value = extract_field(text, field)
            if not value:
                return []
            # Handle both comma and newline separated lists
            items = re.split(r'[,\n]', value)
            return [item.strip().lstrip('- ') for item in items if item.strip()]

        summary = extract_field(response, 'SUMMARY')
        purpose = extract_field(response, 'PURPOSE')
        key_columns = extract_list(response, 'KEY_COLUMNS')
        relationships = extract_list(response, 'RELATIONSHIPS')
        keywords = extract_list(response, 'KEYWORDS')
        queries = extract_list(response, 'QUERIES')

        # Fallbacks
        if not summary:
            summary = f"Table storing {table_name.split('.')[-1].replace('_', ' ').lower()} data"
        if not key_columns:
            key_columns = [c['name'] for c in columns[:5]]
        if not keywords:
            keywords = self._extract_keywords_from_name(table_name)

        return SchemaSummary(
            table_name=table_name,
            summary=summary,
            purpose=purpose or summary,
            key_columns=key_columns,
            relationships=relationships,
            keywords=keywords,
            common_queries=queries
        )

    def _extract_keywords_from_name(self, table_name: str) -> List[str]:
        """Extract keywords from table name (fallback)"""
        # Remove schema prefix
        name = table_name.split('.')[-1]

        # Split on common patterns
        words = re.split(r'[_\s]', name)

        # Also split camelCase
        result = []
        for word in words:
            # Split camelCase: "CompanyProducts" -> ["Company", "Products"]
            camel_split = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
            result.extend(camel_split)

        # Clean and lowercase
        return [w.lower() for w in result if len(w) > 2]

    async def check_llm_available(self) -> bool:
        """Check if LLM is running and model is available (llama-cpp-python)"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.llm_url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('data', [])
                    return len(models) > 0
        except:
            pass
        return False


async def summarize_schemas_for_database(
    database: str,
    llm_url: str = "http://localhost:11434",
    model: str = "llama3.2",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to summarize all schemas for a database from MongoDB.

    Args:
        database: Database lookup key
        llm_url: LLM API URL
        model: LLM model name
        verbose: Print progress

    Returns:
        Dict with success status and summary count
    """
    import sys
    from pathlib import Path

    parent = str(Path(__file__).parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from mongodb import get_mongodb_service

    summarizer = SchemaSummarizer(llm_url=llm_url, model=model)

    # Check LLM availability
    if not await summarizer.check_llm_available():
        return {
            "success": False,
            "error": f"LLM not available or model '{model}' not found at {llm_url}"
        }

    return {
        "success": True,
        "message": "Summarization ready - call summarize_schema for each table",
        "llm_url": llm_url,
        "model": model
    }
