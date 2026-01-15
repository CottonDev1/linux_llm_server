"""
Stored Procedure Extractor

This module extracts stored procedures (name, parameters, definition/code)
from SQL Server databases for semantic search.

This allows the LLM to:
- Find relevant stored procedures based on natural language queries
- Suggest using existing SPs instead of writing raw SQL
- Understand what operations are available in the database

Usage:
    import pymssql
    from sql_pipeline.extraction.stored_procedure_extractor import extract_stored_procedures

    conn = pymssql.connect(...)
    procedures = await extract_stored_procedures(conn, 'database_name')
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StoredProcedure:
    """Stored procedure metadata"""
    schema: str
    name: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    definition: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"


async def extract_stored_procedures(
    connection,
    database: str,
    verbose: bool = True
) -> List[StoredProcedure]:
    """
    Extract all stored procedures from a database.

    Args:
        connection: pymssql or pyodbc connection object
        database: Database name for logging
        verbose: Whether to print progress messages

    Returns:
        List of StoredProcedure objects with name, parameters, and definition
    """
    if verbose:
        print(f"\n  Extracting stored procedures from {database}...")

    cursor = connection.cursor()

    try:
        # Get list of all stored procedures (excluding system SPs)
        procedures_query = """
            SELECT
                SCHEMA_NAME(schema_id) AS SCHEMA_NAME,
                name AS PROCEDURE_NAME
            FROM sys.procedures
            WHERE is_ms_shipped = 0
            ORDER BY SCHEMA_NAME(schema_id), name
        """

        cursor.execute(procedures_query)
        procedures_list = cursor.fetchall()

        if verbose:
            print(f"   Found {len(procedures_list)} stored procedures")

        # Extract details for each procedure
        stored_procedures = []
        for row in procedures_list:
            schema = row[0]
            name = row[1]

            try:
                parameters = _extract_procedure_parameters(cursor, schema, name)
                definition = _extract_procedure_definition(cursor, schema, name)

                stored_procedures.append(StoredProcedure(
                    schema=schema,
                    name=name,
                    parameters=parameters,
                    definition=definition
                ))

                if verbose:
                    print(f"   Extracted {schema}.{name}")

            except Exception as e:
                if verbose:
                    print(f"   Failed to extract {schema}.{name}: {e}")

        return stored_procedures

    finally:
        cursor.close()


def _extract_procedure_parameters(
    cursor,
    schema: str,
    procedure_name: str
) -> List[Dict[str, Any]]:
    """
    Extract parameters for a stored procedure.

    Args:
        cursor: Database cursor
        schema: Procedure schema
        procedure_name: Procedure name

    Returns:
        List of parameter dictionaries
    """
    query = """
        SELECT
            p.name AS PARAMETER_NAME,
            TYPE_NAME(p.user_type_id) AS PARAMETER_TYPE,
            p.max_length AS MAX_LENGTH,
            p.precision AS PRECISION,
            p.scale AS SCALE,
            p.is_output AS IS_OUTPUT,
            p.has_default_value AS HAS_DEFAULT,
            p.default_value AS DEFAULT_VALUE
        FROM sys.parameters p
        JOIN sys.procedures sp
            ON p.object_id = sp.object_id
        WHERE SCHEMA_NAME(sp.schema_id) = ?
            AND sp.name = ?
        ORDER BY p.parameter_id
    """

    try:
        # Try pyodbc-style (?) first, fall back to pymssql-style (%s)
        try:
            cursor.execute(query, (schema, procedure_name))
        except:
            query = query.replace('?', '%s')
            cursor.execute(query, (schema, procedure_name))
        rows = cursor.fetchall()

        parameters = []
        for row in rows:
            parameters.append({
                'name': row[0],
                'type': row[1],
                'maxLength': row[2],
                'precision': row[3],
                'scale': row[4],
                'isOutput': bool(row[5]),
                'hasDefault': bool(row[6]),
                'defaultValue': row[7]
            })

        return parameters

    except Exception as e:
        print(f"   Failed to extract parameters for {schema}.{procedure_name}: {e}")
        return []


def _extract_procedure_definition(
    cursor,
    schema: str,
    procedure_name: str
) -> str:
    """
    Extract the SQL definition (code) of a stored procedure.

    Args:
        cursor: Database cursor
        schema: Procedure schema
        procedure_name: Procedure name

    Returns:
        SQL code of the stored procedure
    """
    full_name = f"{schema}.{procedure_name}"

    query = f"SELECT OBJECT_DEFINITION(OBJECT_ID('{full_name}')) AS DEFINITION"

    try:
        cursor.execute(query)
        row = cursor.fetchone()
        return row[0] if row and row[0] else ''

    except Exception as e:
        print(f"   Failed to extract definition for {schema}.{procedure_name}: {e}")
        return ''


def extract_stored_procedures_sync(
    connection,
    database: str,
    verbose: bool = True
) -> List[StoredProcedure]:
    """
    Synchronous version of extract_stored_procedures for non-async contexts.

    Args:
        connection: pymssql or pyodbc connection object
        database: Database name for logging
        verbose: Whether to print progress messages

    Returns:
        List of StoredProcedure objects
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        extract_stored_procedures(connection, database, verbose)
    )
