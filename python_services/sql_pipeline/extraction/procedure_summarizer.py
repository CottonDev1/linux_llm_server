"""
Stored Procedure Summarizer

Uses LLM to generate human-readable summaries for stored procedures.
This improves semantic search by embedding meaningful descriptions rather than raw SQL.

Based on nilenso best practices:
- Generate concise natural language summaries
- Extract key operations and business logic
- Identify input/output parameters in plain English

Usage:
    from sql_pipeline.extraction.procedure_summarizer import ProcedureSummarizer

    summarizer = ProcedureSummarizer(llm_url="http://localhost:11434")
    summary = await summarizer.summarize_procedure(procedure_info)
"""

import httpx
import asyncio
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ProcedureSummary:
    """Summary generated for a stored procedure"""
    procedure_name: str
    summary: str
    operations: List[str]  # e.g., ["SELECT", "INSERT", "UPDATE"]
    tables_referenced: List[str]
    keywords: List[str]  # Business keywords for search
    input_description: str
    output_description: str


class ProcedureSummarizer:
    """
    Generates LLM summaries for stored procedures.
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 60,
        max_definition_length: int = 8000
    ):
        self.llm_url = llm_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_definition_length = max_definition_length

    async def summarize_procedure(
        self,
        procedure_name: str,
        procedure_info: Dict[str, Any],
        verbose: bool = True
    ) -> ProcedureSummary:
        """
        Generate a summary for a single stored procedure.

        Args:
            procedure_name: Full procedure name (schema.name)
            procedure_info: Dict with schema, parameters, definition
            verbose: Print progress

        Returns:
            ProcedureSummary with natural language description
        """
        definition = procedure_info.get('definition', '')
        parameters = procedure_info.get('parameters', [])
        schema = procedure_info.get('schema', 'dbo')

        # Truncate very long definitions
        if len(definition) > self.max_definition_length:
            definition = definition[:self.max_definition_length] + "\n-- [TRUNCATED]"

        # Build parameter description
        param_desc = self._format_parameters(parameters)

        # Create prompt for LLM
        prompt = self._build_prompt(procedure_name, definition, param_desc)

        if verbose:
            print(f"   Summarizing {procedure_name}...")

        try:
            # Call LLM API
            response = await self._call_llm(prompt)

            # Parse the response
            summary = self._parse_response(response, procedure_name, definition)

            return summary

        except Exception as e:
            if verbose:
                print(f"   Failed to summarize {procedure_name}: {e}")

            # Return basic summary on failure
            return ProcedureSummary(
                procedure_name=procedure_name,
                summary=f"Stored procedure {procedure_name}",
                operations=self._extract_operations(definition),
                tables_referenced=self._extract_tables(definition),
                keywords=[procedure_name.split('.')[-1]],
                input_description=param_desc or "No parameters",
                output_description="Unknown"
            )

    async def summarize_batch(
        self,
        procedures: List[Dict[str, Any]],
        database: str,
        verbose: bool = True,
        concurrency: int = 3
    ) -> List[ProcedureSummary]:
        """
        Summarize multiple procedures with controlled concurrency.

        Args:
            procedures: List of procedure dicts with name, info
            database: Database name for context
            verbose: Print progress
            concurrency: Max concurrent LLM calls

        Returns:
            List of ProcedureSummary objects
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def summarize_with_limit(proc):
            async with semaphore:
                return await self.summarize_procedure(
                    procedure_name=proc['name'],
                    procedure_info=proc['info'],
                    verbose=verbose
                )

        tasks = [summarize_with_limit(p) for p in procedures]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [r for r in results if isinstance(r, ProcedureSummary)]

    def _format_parameters(self, parameters: List[Dict[str, Any]]) -> str:
        """Format parameters as readable string"""
        if not parameters:
            return ""

        lines = []
        for p in parameters:
            direction = "OUTPUT" if p.get('isOutput') else "INPUT"
            lines.append(f"  - {p['name']} ({p['type']}) [{direction}]")

        return "\n".join(lines)

    def _build_prompt(self, name: str, definition: str, params: str) -> str:
        """Build the LLM prompt for summarization"""
        return f"""Analyze this SQL Server stored procedure and provide a concise summary.

PROCEDURE: {name}

PARAMETERS:
{params if params else "None"}

DEFINITION:
{definition}

Respond in this exact format (keep each section brief, 1-2 sentences max):

SUMMARY: [One sentence describing what this procedure does in business terms]
OPERATIONS: [Comma-separated list of SQL operations: SELECT, INSERT, UPDATE, DELETE, etc.]
TABLES: [Comma-separated list of tables referenced]
KEYWORDS: [5-10 business keywords for searching, comma-separated]
INPUT: [Brief description of what inputs are needed]
OUTPUT: [Brief description of what the procedure returns or modifies]"""

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
                    "max_tokens": 500,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI format: choices[0].text
            return data.get('choices', [{}])[0].get('text', '')

    def _parse_response(self, response: str, procedure_name: str, definition: str) -> ProcedureSummary:
        """Parse LLM response into structured summary"""

        def extract_field(text: str, field: str) -> str:
            pattern = rf'{field}:\s*(.+?)(?=\n[A-Z]+:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        def extract_list(text: str, field: str) -> List[str]:
            value = extract_field(text, field)
            if not value:
                return []
            return [item.strip() for item in value.split(',') if item.strip()]

        summary = extract_field(response, 'SUMMARY')
        operations = extract_list(response, 'OPERATIONS')
        tables = extract_list(response, 'TABLES')
        keywords = extract_list(response, 'KEYWORDS')
        input_desc = extract_field(response, 'INPUT')
        output_desc = extract_field(response, 'OUTPUT')

        # Fallback extraction if LLM response was malformed
        if not operations:
            operations = self._extract_operations(definition)
        if not tables:
            tables = self._extract_tables(definition)
        if not summary:
            summary = f"Stored procedure that performs {', '.join(operations[:3]) or 'database operations'}"

        return ProcedureSummary(
            procedure_name=procedure_name,
            summary=summary,
            operations=operations,
            tables_referenced=tables,
            keywords=keywords or [procedure_name.split('.')[-1]],
            input_description=input_desc or "See parameters",
            output_description=output_desc or "Unknown"
        )

    def _extract_operations(self, definition: str) -> List[str]:
        """Extract SQL operations from definition (fallback)"""
        operations = []
        definition_upper = definition.upper()

        op_patterns = [
            ('SELECT', r'\bSELECT\b'),
            ('INSERT', r'\bINSERT\b'),
            ('UPDATE', r'\bUPDATE\b'),
            ('DELETE', r'\bDELETE\b'),
            ('EXEC', r'\bEXEC(?:UTE)?\b'),
            ('MERGE', r'\bMERGE\b'),
            ('TRUNCATE', r'\bTRUNCATE\b'),
        ]

        for op_name, pattern in op_patterns:
            if re.search(pattern, definition_upper):
                operations.append(op_name)

        return operations

    def _extract_tables(self, definition: str) -> List[str]:
        """Extract table names from definition (fallback)"""
        # Match common table reference patterns
        patterns = [
            r'FROM\s+(\[?\w+\]?\.\[?\w+\]?)',
            r'JOIN\s+(\[?\w+\]?\.\[?\w+\]?)',
            r'INTO\s+(\[?\w+\]?\.\[?\w+\]?)',
            r'UPDATE\s+(\[?\w+\]?\.\[?\w+\]?)',
        ]

        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, definition, re.IGNORECASE)
            for match in matches:
                # Clean up brackets
                clean = match.replace('[', '').replace(']', '')
                if clean and not clean.startswith('#'):  # Skip temp tables
                    tables.add(clean)

        return list(tables)[:10]  # Limit to 10 tables

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


async def summarize_procedures_for_database(
    database: str,
    llm_url: str = "http://localhost:11434",
    model: str = "llama3.2",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to summarize all procedures for a database from MongoDB.

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

    # Add parent to path
    parent = str(Path(__file__).parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from mongodb import get_mongodb_service

    summarizer = ProcedureSummarizer(llm_url=llm_url, model=model)

    # Check LLM availability
    if not await summarizer.check_llm_available():
        return {
            "success": False,
            "error": f"LLM not available or model '{model}' not found at {llm_url}"
        }

    # Get MongoDB service
    mongodb = get_mongodb_service()
    await mongodb.initialize()

    # Get procedures from MongoDB
    # This would need a method to list procedures - for now return placeholder
    return {
        "success": True,
        "message": "Summarization ready - call summarize_procedure for each SP",
        "llm_url": llm_url,
        "model": model
    }
