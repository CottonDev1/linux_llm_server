"""
SQL Schema Validator
Validates generated SQL queries against stored schema metadata.
Uses sqlglot for SQL parsing and in-memory schema cache for fast validation.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from core.log_utils import log_info
from datetime import datetime
from collections import defaultdict

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    logging.warning("sqlglot not installed. SQL validation will be limited.")

from config import COLLECTION_SQL_SCHEMA_CONTEXT

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates SQL queries against stored schema metadata.

    Features:
    - In-memory schema cache for O(1) column/table lookups
    - SQL parsing to extract table and column references
    - Validation with detailed error messages
    - Suggestion of valid alternatives for invalid columns
    """

    def __init__(self, mongodb_service=None):
        self.mongodb = mongodb_service
        self._initialized = False

        # In-memory schema cache
        # Structure: {database: {table_name: {column_name: column_info}}}
        self._schema_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Quick lookup indexes
        # {database: set(table_names)}
        self._table_index: Dict[str, Set[str]] = {}

        # {database: {table_name: set(column_names)}}
        self._column_index: Dict[str, Dict[str, Set[str]]] = {}

        # Cache metadata
        self._cache_loaded_at: Optional[datetime] = None
        self._cache_stats: Dict[str, int] = {
            "databases": 0,
            "tables": 0,
            "columns": 0
        }

    async def initialize(self, mongodb_service=None):
        """Initialize the validator and load schema cache."""
        if mongodb_service:
            self.mongodb = mongodb_service

        if not self.mongodb:
            logger.warning("No MongoDB service provided. Schema validation will be limited.")
            return

        await self._load_schema_cache()
        self._initialized = True
        log_info("Schema Validator", f"Initialized with {self._cache_stats['tables']} tables, "
                 f"{self._cache_stats['columns']} columns across {self._cache_stats['databases']} databases")

    async def _load_schema_cache(self):
        """Load all schema metadata into memory for fast lookups."""
        if not self.mongodb:
            return

        try:
            collection = self.mongodb.db[COLLECTION_SQL_SCHEMA_CONTEXT]

            # Clear existing cache
            self._schema_cache.clear()
            self._table_index.clear()
            self._column_index.clear()

            # Load all schema documents
            cursor = collection.find({})
            async for doc in cursor:
                database = doc.get('database', '').lower()
                table_name = doc.get('table_name', '').lower()
                columns = doc.get('columns', [])

                if not database or not table_name:
                    continue

                # Initialize database structures if needed
                if database not in self._schema_cache:
                    self._schema_cache[database] = {}
                    self._table_index[database] = set()
                    self._column_index[database] = {}

                # Store table
                self._table_index[database].add(table_name)

                # Store columns with metadata
                self._schema_cache[database][table_name] = {}
                self._column_index[database][table_name] = set()

                for col in columns:
                    col_name = col.get('name', '').lower() if isinstance(col, dict) else str(col).lower()
                    if col_name:
                        self._column_index[database][table_name].add(col_name)
                        self._schema_cache[database][table_name][col_name] = col if isinstance(col, dict) else {'name': col}

            # Update stats
            self._cache_stats['databases'] = len(self._schema_cache)
            self._cache_stats['tables'] = sum(len(tables) for tables in self._table_index.values())
            self._cache_stats['columns'] = sum(
                len(cols) for db_cols in self._column_index.values()
                for cols in db_cols.values()
            )
            self._cache_loaded_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Failed to load schema cache: {e}")
            raise

    async def refresh_cache(self):
        """Refresh the schema cache from MongoDB."""
        await self._load_schema_cache()
        logger.info(f"Schema cache refreshed at {self._cache_loaded_at}")

    def parse_sql(self, sql: str) -> Dict[str, Any]:
        """
        Parse SQL query and extract table and column references.

        Returns:
            {
                'success': bool,
                'tables': set of table names,
                'columns': dict of {table_alias: set of column names},
                'aliases': dict of {alias: actual_table_name},
                'error': error message if parsing failed
            }
        """
        if not SQLGLOT_AVAILABLE:
            return {
                'success': False,
                'tables': set(),
                'columns': {},
                'aliases': {},
                'error': 'sqlglot not installed'
            }

        try:
            # Parse the SQL using T-SQL dialect
            parsed = sqlglot.parse_one(sql, dialect="tsql")

            tables = set()
            columns = defaultdict(set)
            aliases = {}

            # Extract table references
            for table in parsed.find_all(exp.Table):
                # Get full table name (schema.table or just table)
                if table.db:
                    full_name = f"{table.db}.{table.name}"
                else:
                    full_name = table.name

                tables.add(full_name.lower())

                # Track aliases
                if table.alias:
                    aliases[table.alias.lower()] = full_name.lower()

            # Extract column references
            for column in parsed.find_all(exp.Column):
                col_name = column.name.lower()
                table_ref = column.table.lower() if column.table else '_unqualified'

                # Skip * (SELECT *)
                if col_name == '*':
                    continue

                columns[table_ref].add(col_name)

            return {
                'success': True,
                'tables': tables,
                'columns': dict(columns),
                'aliases': aliases,
                'error': None
            }

        except ParseError as e:
            return {
                'success': False,
                'tables': set(),
                'columns': {},
                'aliases': {},
                'error': f"SQL parse error: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'tables': set(),
                'columns': {},
                'aliases': {},
                'error': f"Unexpected error parsing SQL: {str(e)}"
            }

    def validate_table(self, database: str, table_name: str) -> bool:
        """Check if a table exists in the schema cache."""
        db_lower = database.lower()

        # Handle schema.table format
        table_lower = table_name.lower()

        if db_lower not in self._table_index:
            return False

        # Try exact match first
        if table_lower in self._table_index[db_lower]:
            return True

        # Try with dbo. prefix if not already present
        if not table_lower.startswith('dbo.'):
            if f"dbo.{table_lower}" in self._table_index[db_lower]:
                return True

        return False

    def validate_column(self, database: str, table_name: str, column_name: str) -> bool:
        """Check if a column exists in the specified table."""
        db_lower = database.lower()
        table_lower = table_name.lower()
        col_lower = column_name.lower()

        if db_lower not in self._column_index:
            return False

        # Try exact table match
        if table_lower in self._column_index[db_lower]:
            return col_lower in self._column_index[db_lower][table_lower]

        # Try with dbo. prefix
        if not table_lower.startswith('dbo.'):
            dbo_table = f"dbo.{table_lower}"
            if dbo_table in self._column_index[db_lower]:
                return col_lower in self._column_index[db_lower][dbo_table]

        return False

    def get_valid_columns(self, database: str, table_name: str) -> Set[str]:
        """Get all valid column names for a table."""
        db_lower = database.lower()
        table_lower = table_name.lower()

        if db_lower not in self._column_index:
            return set()

        # Try exact match
        if table_lower in self._column_index[db_lower]:
            return self._column_index[db_lower][table_lower]

        # Try with dbo. prefix
        if not table_lower.startswith('dbo.'):
            dbo_table = f"dbo.{table_lower}"
            if dbo_table in self._column_index[db_lower]:
                return self._column_index[db_lower][dbo_table]

        return set()

    def get_valid_tables(self, database: str) -> Set[str]:
        """Get all valid table names for a database."""
        db_lower = database.lower()
        return self._table_index.get(db_lower, set())

    def find_similar_columns(self, database: str, table_name: str, invalid_column: str, limit: int = 5) -> List[str]:
        """Find columns similar to an invalid column name (for suggestions)."""
        valid_columns = self.get_valid_columns(database, table_name)
        if not valid_columns:
            return []

        invalid_lower = invalid_column.lower()

        # Score columns by similarity
        scored = []
        for col in valid_columns:
            col_lower = col.lower()
            score = 0

            # Exact substring match
            if invalid_lower in col_lower or col_lower in invalid_lower:
                score += 50

            # Common prefix
            prefix_len = 0
            for i, (a, b) in enumerate(zip(invalid_lower, col_lower)):
                if a == b:
                    prefix_len = i + 1
                else:
                    break
            score += prefix_len * 3

            # Common suffix
            suffix_len = 0
            for i, (a, b) in enumerate(zip(reversed(invalid_lower), reversed(col_lower))):
                if a == b:
                    suffix_len = i + 1
                else:
                    break
            score += suffix_len * 2

            # Word overlap (split by common separators)
            invalid_words = set(re.split(r'[_\s]', invalid_lower))
            col_words = set(re.split(r'[_\s]', col_lower))
            overlap = len(invalid_words & col_words)
            score += overlap * 10

            if score > 0:
                scored.append((col, score))

        # Sort by score descending and return top matches
        scored.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in scored[:limit]]

    def validate_sql(self, sql: str, database: str) -> Dict[str, Any]:
        """
        Validate a SQL query against the schema.

        Returns:
            {
                'valid': bool,
                'errors': list of error dicts,
                'warnings': list of warning dicts,
                'tables_found': set of valid tables,
                'columns_validated': int count of validated columns,
                'suggestions': dict of {invalid_column: [suggested_columns]}
            }
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'tables_found': set(),
            'columns_validated': 0,
            'suggestions': {}
        }

        # Parse the SQL
        parsed = self.parse_sql(sql)

        if not parsed['success']:
            result['valid'] = False
            result['errors'].append({
                'type': 'parse_error',
                'message': parsed['error']
            })
            return result

        db_lower = database.lower()
        tables = parsed['tables']
        columns = parsed['columns']
        aliases = parsed['aliases']

        # Check if database exists in cache
        if db_lower not in self._table_index:
            result['warnings'].append({
                'type': 'unknown_database',
                'database': database,
                'message': f"Database '{database}' not found in schema cache. Validation may be incomplete."
            })
            return result

        # Validate tables
        valid_tables = {}  # Maps alias/name to actual table name
        for table in tables:
            if self.validate_table(database, table):
                result['tables_found'].add(table)
                valid_tables[table] = table
                # Add dbo. version if applicable
                if not table.startswith('dbo.'):
                    valid_tables[f"dbo.{table}"] = f"dbo.{table}"
            else:
                result['valid'] = False
                result['errors'].append({
                    'type': 'unknown_table',
                    'table': table,
                    'message': f"Table '{table}' not found in database '{database}'"
                })

        # Add aliases to valid_tables mapping
        for alias, actual_table in aliases.items():
            if actual_table in valid_tables:
                valid_tables[alias] = actual_table

        # Validate columns
        for table_ref, cols in columns.items():
            # Resolve table reference
            actual_table = None
            if table_ref == '_unqualified':
                # Unqualified columns - try to find in any valid table
                for col in cols:
                    found = False
                    for vt in result['tables_found']:
                        if self.validate_column(database, vt, col):
                            result['columns_validated'] += 1
                            found = True
                            break

                    if not found:
                        result['valid'] = False
                        # Get suggestions from all valid tables
                        all_suggestions = []
                        for vt in result['tables_found']:
                            suggestions = self.find_similar_columns(database, vt, col)
                            all_suggestions.extend(suggestions)

                        result['errors'].append({
                            'type': 'unknown_column',
                            'column': col,
                            'table': 'unqualified',
                            'message': f"Column '{col}' not found in any referenced table"
                        })
                        if all_suggestions:
                            result['suggestions'][col] = list(set(all_suggestions))[:5]
            else:
                # Qualified column reference
                actual_table = valid_tables.get(table_ref)

                if not actual_table:
                    # Table reference not found - might be alias we didn't catch
                    result['warnings'].append({
                        'type': 'unresolved_table_reference',
                        'reference': table_ref,
                        'message': f"Could not resolve table reference '{table_ref}'"
                    })
                    continue

                for col in cols:
                    if self.validate_column(database, actual_table, col):
                        result['columns_validated'] += 1
                    else:
                        result['valid'] = False
                        suggestions = self.find_similar_columns(database, actual_table, col)

                        result['errors'].append({
                            'type': 'unknown_column',
                            'column': col,
                            'table': actual_table,
                            'message': f"Column '{col}' not found in table '{actual_table}'"
                        })
                        if suggestions:
                            result['suggestions'][col] = suggestions

        return result

    def format_validation_feedback(self, validation_result: Dict[str, Any], database: str) -> str:
        """
        Format validation errors into feedback for LLM self-correction.

        Returns a string that can be added to the prompt to help the LLM fix the query.
        """
        if validation_result['valid']:
            return ""

        feedback_parts = ["The generated SQL has validation errors:\n"]

        for error in validation_result['errors']:
            if error['type'] == 'unknown_table':
                valid_tables = list(self.get_valid_tables(database))[:10]
                feedback_parts.append(f"- Table '{error['table']}' does not exist.")
                if valid_tables:
                    feedback_parts.append(f"  Available tables include: {', '.join(valid_tables)}")

            elif error['type'] == 'unknown_column':
                col = error['column']
                table = error.get('table', 'unknown')
                suggestions = validation_result['suggestions'].get(col, [])

                feedback_parts.append(f"- Column '{col}' does not exist in table '{table}'.")
                if suggestions:
                    feedback_parts.append(f"  Did you mean: {', '.join(suggestions)}?")
                else:
                    # Show valid columns for the table
                    if table and table != 'unqualified':
                        valid_cols = list(self.get_valid_columns(database, table))[:15]
                        if valid_cols:
                            feedback_parts.append(f"  Valid columns in '{table}': {', '.join(valid_cols)}")

            elif error['type'] == 'parse_error':
                feedback_parts.append(f"- SQL syntax error: {error['message']}")

        feedback_parts.append("\nPlease fix the SQL query using only valid table and column names.")

        return "\n".join(feedback_parts)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the schema cache."""
        return {
            'initialized': self._initialized,
            'loaded_at': self._cache_loaded_at.isoformat() if self._cache_loaded_at else None,
            'databases': self._cache_stats['databases'],
            'tables': self._cache_stats['tables'],
            'columns': self._cache_stats['columns'],
            'databases_list': list(self._schema_cache.keys())
        }


# Global validator instance
_validator_instance: Optional[SchemaValidator] = None


async def get_schema_validator(mongodb_service=None) -> SchemaValidator:
    """Get or create the global schema validator instance."""
    global _validator_instance

    if _validator_instance is None:
        _validator_instance = SchemaValidator(mongodb_service)
        if mongodb_service:
            await _validator_instance.initialize()
    elif mongodb_service and not _validator_instance._initialized:
        await _validator_instance.initialize(mongodb_service)

    return _validator_instance
