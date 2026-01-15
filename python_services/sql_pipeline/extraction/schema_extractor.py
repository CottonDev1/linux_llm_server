"""
Schema Extractor

Programmatic API for extracting SQL database schemas.
This is the Python equivalent of SQL_Context_Creation/schema-extractor-api.js

Design Philosophy:
- Direct MongoDB calls (no HTTP needed - same process)
- Async-first with optional sync wrappers
- Event-driven progress reporting via callbacks
- Comprehensive error handling

Usage:
    from sql_pipeline.extraction import SchemaExtractor

    extractor = SchemaExtractor()
    await extractor.initialize()

    # Extract single database
    result = await extractor.extract_single_database(db_config)

    # Extract from config file
    results = await extractor.extract_from_config('./config.json')
"""

import time
import sys
import os
import re
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports when running as submodule
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from .config_parser import parse_config, DatabaseConfig
from .stored_procedure_extractor import extract_stored_procedures, StoredProcedure
from .schema_summarizer import SchemaSummarizer
from .procedure_summarizer import ProcedureSummarizer

# Try pyodbc first (better Windows support), fall back to pymssql
try:
    import pyodbc
    SQL_DRIVER = 'pyodbc'
except ImportError:
    try:
        import pymssql
        SQL_DRIVER = 'pymssql'
    except ImportError:
        SQL_DRIVER = None


@dataclass
class ExtractionStats:
    """Statistics from extraction operation"""
    database: str
    tables: int = 0
    procedures: int = 0
    errors: int = 0
    duration_ms: int = 0
    error_message: Optional[str] = None


@dataclass
class TableInfo:
    """Table metadata"""
    db_schema: str
    name: str
    full_name: str
    columns: List[Dict[str, Any]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, Any]] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)
    sample_values: Dict[str, List[Any]] = field(default_factory=dict)


class SchemaExtractor:
    """
    Main API for extracting database schemas.

    Events (via callbacks):
    - on_progress(type, current, total, item): Table/procedure progress
    - on_database_start(database, lookup_key): Database extraction started
    - on_database_complete(database, stats): Database extraction completed
    - on_error(error, context): Error occurred
    """

    def __init__(
        self,
        verbose: bool = True,
        connection_timeout: int = 60,
        request_timeout: int = 60,
        llm_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2",
        enable_summarization: bool = True,
        skip_existing: bool = True,
        on_progress: Optional[Callable] = None,
        on_database_start: Optional[Callable] = None,
        on_database_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        self.verbose = verbose
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.skip_existing = skip_existing

        # Event callbacks
        self.on_progress = on_progress
        self.on_database_start = on_database_start
        self.on_database_complete = on_database_complete
        self.on_error = on_error

        # MongoDB service - will be initialized lazily
        self._mongodb_service = None
        self.is_initialized = False

        # Summarizers
        self.enable_summarization = enable_summarization
        self._schema_summarizer = None
        self._procedure_summarizer = None
        if enable_summarization:
            self._schema_summarizer = SchemaSummarizer(
                llm_url=llm_url,
                model=llm_model
            )
            self._procedure_summarizer = ProcedureSummarizer(
                llm_url=llm_url,
                model=llm_model
            )

    async def initialize(self):
        """Initialize MongoDB service connection"""
        if self.is_initialized:
            return

        if SQL_DRIVER is None:
            raise ImportError(
                "No SQL Server driver found. Install pymssql or pyodbc:\n"
                "  pip install pymssql\n"
                "  - or -\n"
                "  pip install pyodbc"
            )

        # Import MongoDB service
        from mongodb import get_mongodb_service

        self._mongodb_service = get_mongodb_service()
        await self._mongodb_service.initialize()

        # Check LLM availability if summarization is enabled
        if self.enable_summarization and self._schema_summarizer:
            llm_available = await self._schema_summarizer.check_llm_available()
            if not llm_available:
                if self.verbose:
                    print("Warning: LLM not available, summarization will be skipped")
                self.enable_summarization = False

        self.is_initialized = True
        if self.verbose:
            summarization_status = "enabled" if self.enable_summarization else "disabled"
            print(f"SchemaExtractor initialized (driver: {SQL_DRIVER}, summarization: {summarization_status})")

    def _emit_progress(self, type_: str, current: int, total: int, item: str):
        """Emit progress event"""
        if self.on_progress:
            self.on_progress(type_, current, total, item)
        elif self.verbose:
            print(f"   [{current}/{total}] {type_}: {item}")

    def _emit_error(self, error: str, context: Dict[str, Any]):
        """Emit error event"""
        if self.on_error:
            self.on_error(error, context)
        elif self.verbose:
            print(f"   ERROR: {error} (context: {context})")

    async def _get_existing_items(self, database: str) -> Dict[str, set]:
        """
        Get already extracted tables and procedures from MongoDB.

        Args:
            database: The normalized database lookup key

        Returns:
            Dict with 'tables' and 'procedures' sets of existing names
        """
        from config import COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES
        from database_name_parser import normalize_database_name

        normalized_db = normalize_database_name(database)
        existing = {'tables': set(), 'procedures': set()}

        try:
            # Get existing tables
            schema_collection = self._mongodb_service.db[COLLECTION_SQL_SCHEMA_CONTEXT]
            cursor = schema_collection.find(
                {"database": normalized_db},
                {"table_name": 1}
            )
            async for doc in cursor:
                if doc.get('table_name'):
                    existing['tables'].add(doc['table_name'])

            # Get existing procedures
            sp_collection = self._mongodb_service.db[COLLECTION_SQL_STORED_PROCEDURES]
            cursor = sp_collection.find(
                {"database": normalized_db},
                {"procedure_name": 1}
            )
            async for doc in cursor:
                if doc.get('procedure_name'):
                    existing['procedures'].add(doc['procedure_name'])

        except Exception as e:
            if self.verbose:
                print(f"   Warning: Could not fetch existing items: {e}")

        return existing

    async def extract_from_config(
        self,
        config_path: str,
        only: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract schemas from configuration file.

        Args:
            config_path: Path to JSON or XML config file
            only: Extract only specific database by name/lookupKey

        Returns:
            Dict with success, databases list, and totalStats
        """
        await self.initialize()

        # Parse configuration file
        databases = await parse_config(config_path)

        # Filter if 'only' option specified
        if only:
            databases = [
                db for db in databases
                if db.name == only or db.lookup_key == only
            ]

            if not databases:
                raise ValueError(f"Database '{only}' not found in configuration")

        # Extract each database
        results = []
        for db_config in databases:
            try:
                result = await self.extract_single_database(db_config)
                results.append(result)
            except Exception as e:
                self._emit_error(str(e), {'database': db_config.database})
                results.append(ExtractionStats(
                    database=db_config.lookup_key,
                    error_message=str(e)
                ))

        # Aggregate results
        total_stats = self._aggregate_results(results)

        return {
            'success': True,
            'databases': [self._stats_to_dict(r) for r in results],
            'totalStats': total_stats
        }

    async def extract_single_database(
        self,
        db_config: DatabaseConfig
    ) -> ExtractionStats:
        """
        Extract schema from a single database.

        Args:
            db_config: Database configuration

        Returns:
            ExtractionStats with counts and duration
        """
        await self.initialize()

        # Emit start event
        if self.on_database_start:
            self.on_database_start(db_config.database, db_config.lookup_key)
        elif self.verbose:
            print(f"\nExtracting schema from {db_config.database} (key: {db_config.lookup_key})")

        stats = ExtractionStats(database=db_config.lookup_key)
        start_time = time.time()
        connection = None

        try:
            # Connect to database
            connection = self._create_connection(db_config)

            # Get existing items from MongoDB if skip_existing is enabled
            existing_items = {'tables': set(), 'procedures': set()}
            skipped_tables = 0
            skipped_procedures = 0

            if self.skip_existing:
                existing_items = await self._get_existing_items(db_config.lookup_key)
                if self.verbose and (existing_items['tables'] or existing_items['procedures']):
                    print(f"   Found {len(existing_items['tables'])} existing tables, "
                          f"{len(existing_items['procedures'])} existing procedures in MongoDB")

            # Extract tables
            tables = self._extract_tables(connection)

            if not tables:
                if self.verbose:
                    print(f"   No tables found in {db_config.database}")
                stats.duration_ms = int((time.time() - start_time) * 1000)
                return stats

            # Process tables and views
            for i, table in enumerate(tables):
                obj_type = table.get('objectType', 'table')
                # Skip if already exists in MongoDB
                if self.skip_existing and table['fullName'] in existing_items['tables']:
                    skipped_tables += 1
                    if self.verbose:
                        print(f"   [{i + 1}/{len(tables)}] SKIP {obj_type}: {table['fullName']} (already exists)")
                    continue

                try:
                    schema_info = self._extract_table_schema(connection, table)

                    # Generate LLM summary if enabled
                    # SchemaSummary contains: summary, purpose, key_columns, relationships, keywords, common_queries
                    summary_result = None
                    if self.enable_summarization and self._schema_summarizer:
                        try:
                            summary_result = await self._schema_summarizer.summarize_schema(
                                table_name=schema_info['fullName'],
                                schema_info={
                                    'columns': schema_info['columns'],
                                    'primaryKeys': schema_info['primaryKeys'],
                                    'foreignKeys': schema_info['foreignKeys'],
                                    'relatedTables': schema_info['relatedTables'],
                                    'sampleValues': schema_info['sampleValues']
                                },
                                verbose=self.verbose
                            )
                        except Exception as e:
                            if self.verbose:
                                print(f"   Warning: Could not generate summary for {table['fullName']}: {e}")

                    # Store directly to MongoDB with full semantic metadata
                    # Pass all SchemaSummary fields for enhanced RAG retrieval
                    await self._mongodb_service.store_schema_context(
                        database=db_config.lookup_key,
                        table_name=schema_info['fullName'],
                        schema_info={
                            'schema': schema_info['schema'],
                            'columns': schema_info['columns'],
                            'primaryKeys': schema_info['primaryKeys'],
                            'foreignKeys': schema_info['foreignKeys'],
                            'relatedTables': schema_info['relatedTables'],
                            'sampleValues': schema_info['sampleValues']
                        },
                        # Pass full SchemaSummary fields (nilenso: ~27% accuracy improvement)
                        summary=summary_result.summary if summary_result else None,
                        purpose=summary_result.purpose if summary_result else None,
                        key_columns=summary_result.key_columns if summary_result else None,
                        relationships=summary_result.relationships if summary_result else None,
                        keywords=summary_result.keywords if summary_result else None,
                        common_queries=summary_result.common_queries if summary_result else None
                    )

                    stats.tables += 1
                    self._emit_progress(obj_type, i + 1, len(tables), table['fullName'])

                except Exception as e:
                    stats.errors += 1
                    self._emit_error(str(e), {obj_type: table['fullName']})

            # Extract stored procedures
            try:
                procedures = await extract_stored_procedures(
                    connection,
                    db_config.database,
                    verbose=self.verbose
                )

                for i, sp in enumerate(procedures):
                    # Skip if already exists in MongoDB
                    if self.skip_existing and sp.name in existing_items['procedures']:
                        skipped_procedures += 1
                        if self.verbose:
                            print(f"   [{i + 1}/{len(procedures)}] SKIP procedure: {sp.full_name} (already exists)")
                        continue

                    try:
                        # Generate LLM summary if enabled
                        # ProcedureSummary contains: summary, operations, tables_referenced, keywords,
                        # input_description, output_description
                        summary_info = {}
                        if self.enable_summarization and self._procedure_summarizer:
                            try:
                                summary_result = await self._procedure_summarizer.summarize_procedure(
                                    procedure_name=sp.full_name,
                                    procedure_info={
                                        'schema': sp.schema,
                                        'parameters': sp.parameters,
                                        'definition': sp.definition
                                    },
                                    verbose=self.verbose
                                )
                                # Pass full ProcedureSummary fields (keep keywords as list for better querying)
                                summary_info = {
                                    'summary': summary_result.summary,
                                    'keywords': summary_result.keywords,  # Keep as list, not joined string
                                    'operations': summary_result.operations,
                                    'tables_affected': summary_result.tables_referenced,
                                    'input_description': summary_result.input_description,
                                    'output_description': summary_result.output_description
                                }
                            except Exception as e:
                                if self.verbose:
                                    print(f"   Warning: Could not generate summary for {sp.full_name}: {e}")

                        # Store directly to MongoDB with full semantic metadata
                        await self._mongodb_service.store_stored_procedure(
                            database=db_config.lookup_key,
                            procedure_name=sp.name,
                            procedure_info={
                                'schema': sp.schema,
                                'parameters': sp.parameters,
                                'definition': sp.definition,
                                **summary_info  # Include all ProcedureSummary fields
                            }
                        )

                        stats.procedures += 1
                        self._emit_progress('procedure', i + 1, len(procedures), sp.full_name)

                    except Exception as e:
                        stats.errors += 1
                        self._emit_error(str(e), {'procedure': sp.full_name})

            except Exception as e:
                self._emit_error(f"Failed to extract stored procedures: {e}", {
                    'database': db_config.database
                })

            stats.duration_ms = int((time.time() - start_time) * 1000)

            # Emit completion event
            if self.on_database_complete:
                self.on_database_complete(db_config.database, stats)
            elif self.verbose:
                skip_msg = ""
                if skipped_tables > 0 or skipped_procedures > 0:
                    skip_msg = f" (skipped: {skipped_tables} tables, {skipped_procedures} procedures)"
                print(f"\n   Completed: {stats.tables} tables, {stats.procedures} procedures, {stats.errors} errors ({stats.duration_ms}ms){skip_msg}")

            return stats

        except Exception as e:
            stats.duration_ms = int((time.time() - start_time) * 1000)
            stats.error_message = str(e)
            self._emit_error(str(e), {'database': db_config.database})
            raise

        finally:
            # Always close connection
            if connection:
                try:
                    connection.close()
                except Exception as e:
                    self._emit_error(f"Error closing connection: {e}", {
                        'database': db_config.database
                    })

    def _create_connection(self, db_config: DatabaseConfig):
        """Create SQL Server connection"""
        if SQL_DRIVER == 'pymssql':
            return self._create_pymssql_connection(db_config)
        else:
            return self._create_pyodbc_connection(db_config)

    def _create_pymssql_connection(self, db_config: DatabaseConfig):
        """Create connection using pymssql"""
        import pymssql

        kwargs = {
            'server': db_config.server,
            'database': db_config.database,
            'port': str(db_config.port),
            'login_timeout': self.connection_timeout,
            'timeout': self.request_timeout,
            'user': db_config.user,
            'password': db_config.password,
        }

        return pymssql.connect(**kwargs)

    def _create_pyodbc_connection(self, db_config: DatabaseConfig):
        """Create connection using pyodbc"""
        import pyodbc

        # Build server string - only add port if not using local/named pipes notation
        server = db_config.server
        if db_config.port != 1433 and not server.startswith('('):
            server = f"{server},{db_config.port}"

        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={db_config.database};"
            f"UID={db_config.user};"
            f"PWD={db_config.password};"
        )

        return pyodbc.connect(
            conn_str,
            timeout=self.connection_timeout
        )

    def _extract_tables(self, connection) -> List[Dict[str, str]]:
        """Extract tables and views from database"""
        cursor = connection.cursor()

        query = """
            SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW')
            ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        return [
            {
                'schema': row[0],
                'name': row[1],
                'fullName': f"{row[0]}.{row[1]}",
                'objectType': 'view' if row[2] == 'VIEW' else 'table'
            }
            for row in rows
        ]

    def _extract_table_schema(self, connection, table: Dict[str, str]) -> Dict[str, Any]:
        """Extract complete table/view schema"""
        schema = table['schema']
        name = table['name']
        object_type = table.get('objectType', 'table')

        columns = self._extract_columns(connection, schema, name)
        primary_keys = self._extract_primary_keys(connection, schema, name)
        foreign_keys = self._extract_foreign_keys(connection, schema, name)

        # Extract sample values for context (skip for views to avoid performance issues)
        sample_values = {}
        if object_type == 'table':
            sample_values = self._extract_sample_values(connection, schema, name, columns)

        related_tables = list(set(fk['referencedTable'] for fk in foreign_keys))

        return {
            'schema': schema,
            'tableName': name,
            'fullName': table['fullName'],
            'objectType': object_type,
            'columns': columns,
            'primaryKeys': primary_keys,
            'foreignKeys': foreign_keys,
            'relatedTables': related_tables,
            'sampleValues': sample_values
        }

    def _extract_columns(self, connection, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """Extract columns for a table"""
        cursor = connection.cursor()

        query = """
            SELECT
                COLUMN_NAME, DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE,
                IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """

        # pyodbc uses ? for parameters, pymssql uses %s
        if SQL_DRIVER == 'pyodbc':
            query = query.replace('%s', '?')

        cursor.execute(query, (schema, table_name))
        rows = cursor.fetchall()
        cursor.close()

        columns = []
        for row in rows:
            data_type = row[1]
            char_max_length = row[2]
            numeric_precision = row[3]
            numeric_scale = row[4]

            # Build full type string
            if char_max_length and char_max_length > 0:
                if char_max_length == -1:
                    data_type += '(MAX)'
                else:
                    data_type += f'({char_max_length})'
            elif numeric_precision:
                if numeric_scale:
                    data_type += f'({numeric_precision},{numeric_scale})'
                else:
                    data_type += f'({numeric_precision})'

            columns.append({
                'name': row[0],
                'type': data_type,
                'nullable': row[5] == 'YES',
                'defaultValue': row[6]
            })

        return columns

    def _extract_primary_keys(self, connection, schema: str, table_name: str) -> List[str]:
        """Extract primary keys for a table"""
        cursor = connection.cursor()

        query = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
                AND tc.TABLE_NAME = kcu.TABLE_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                AND tc.TABLE_SCHEMA = %s
                AND tc.TABLE_NAME = %s
            ORDER BY kcu.ORDINAL_POSITION
        """

        if SQL_DRIVER == 'pyodbc':
            query = query.replace('%s', '?')

        cursor.execute(query, (schema, table_name))
        rows = cursor.fetchall()
        cursor.close()

        return [row[0] for row in rows]

    def _extract_foreign_keys(self, connection, schema: str, table_name: str) -> List[Dict[str, str]]:
        """Extract foreign keys for a table"""
        cursor = connection.cursor()

        query = """
            SELECT
                fk.name AS FK_NAME,
                COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS COLUMN_NAME,
                OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS REFERENCED_TABLE_SCHEMA,
                OBJECT_NAME(fk.referenced_object_id) AS REFERENCED_TABLE_NAME,
                COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS REFERENCED_COLUMN_NAME
            FROM sys.foreign_keys fk
            JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            WHERE OBJECT_SCHEMA_NAME(fk.parent_object_id) = %s
                AND OBJECT_NAME(fk.parent_object_id) = %s
            ORDER BY fk.name, fkc.constraint_column_id
        """

        if SQL_DRIVER == 'pyodbc':
            query = query.replace('%s', '?')

        cursor.execute(query, (schema, table_name))
        rows = cursor.fetchall()
        cursor.close()

        return [
            {
                'name': row[0],
                'column': row[1],
                'referencedTable': f"{row[2]}.{row[3]}",
                'referencedColumn': row[4]
            }
            for row in rows
        ]

    def _extract_sample_values(
        self,
        connection,
        schema: str,
        table_name: str,
        columns: List[Dict[str, Any]],
        max_samples: int = 5
    ) -> Dict[str, List[Any]]:
        """
        Extract sample values for columns to help LLM understand data patterns.

        Focuses on string/categorical columns that give context about the data.
        Skips large text, binary, and numeric ID columns.

        Args:
            connection: Database connection
            schema: Table schema name
            table_name: Table name
            columns: List of column info dicts
            max_samples: Max distinct values per column

        Returns:
            Dict mapping column names to lists of sample values
        """
        sample_values = {}
        cursor = connection.cursor()

        # Column types that are useful for sampling
        useful_types = [
            'varchar', 'nvarchar', 'char', 'nchar',  # String types
            'bit',  # Boolean
            'tinyint', 'smallint',  # Small integers (often enums)
        ]

        # Skip these patterns (IDs, timestamps, large fields)
        skip_patterns = ['id', 'key', 'guid', 'uuid', 'date', 'time', 'created', 'modified', 'updated']

        for col in columns:
            col_name = col['name']
            col_type = col['type'].lower()

            # Skip if column name suggests it's an ID or timestamp
            col_name_lower = col_name.lower()
            if any(pattern in col_name_lower for pattern in skip_patterns):
                continue

            # Skip if not a useful type for sampling
            if not any(t in col_type for t in useful_types):
                continue

            # Skip very long string columns (likely descriptions/content)
            if 'varchar' in col_type or 'nvarchar' in col_type:
                # Extract length if specified
                length_match = re.search(r'\((\d+)\)', col_type)
                if length_match:
                    length = int(length_match.group(1))
                    if length > 100:  # Skip long text fields
                        continue

            try:
                # Query distinct values
                query = f"""
                    SELECT DISTINCT TOP {max_samples} [{col_name}]
                    FROM [{schema}].[{table_name}]
                    WHERE [{col_name}] IS NOT NULL
                    ORDER BY [{col_name}]
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                values = [row[0] for row in rows if row[0] is not None]
                if values:
                    # Convert to strings for consistent storage
                    sample_values[col_name] = [str(v)[:50] for v in values]  # Truncate long values

            except Exception as e:
                # Skip columns that can't be queried (computed, etc.)
                continue

        cursor.close()
        return sample_values

    def _aggregate_results(self, results: List[ExtractionStats]) -> Dict[str, int]:
        """Aggregate results from multiple databases"""
        return {
            'tables': sum(r.tables for r in results),
            'procedures': sum(r.procedures for r in results),
            'errors': sum(r.errors for r in results)
        }

    def _stats_to_dict(self, stats: ExtractionStats) -> Dict[str, Any]:
        """Convert ExtractionStats to dictionary"""
        return {
            'database': stats.database,
            'tables': stats.tables,
            'procedures': stats.procedures,
            'errors': stats.errors,
            'duration_ms': stats.duration_ms,
            'error': stats.error_message
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from MongoDB service"""
        await self.initialize()
        return await self._mongodb_service.get_sql_rag_stats()


# Convenience functions for one-off extractions

async def extract_database(
    db_config: DatabaseConfig,
    verbose: bool = True
) -> ExtractionStats:
    """
    Convenience function for simple one-off extractions.

    Args:
        db_config: Database configuration
        verbose: Whether to print progress

    Returns:
        ExtractionStats with counts and duration
    """
    extractor = SchemaExtractor(verbose=verbose)
    return await extractor.extract_single_database(db_config)


async def extract_from_config(
    config_path: str,
    only: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for config file extraction.

    Args:
        config_path: Path to config file
        only: Extract only specific database
        verbose: Whether to print progress

    Returns:
        Extraction results
    """
    extractor = SchemaExtractor(verbose=verbose)
    return await extractor.extract_from_config(config_path, only=only)
