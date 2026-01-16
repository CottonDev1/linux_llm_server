"""
EnrichStoredProcedures.py
Parses SQL definitions to extract tables_affected and operations for stored procedures.
"""

import os
import re
from pymongo import MongoClient
from typing import List, Set, Tuple

# MongoDB connection - use environment variable with localhost fallback
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE = "rag_server"
COLLECTION = "sql_stored_procedures"


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL definition.
    Looks for tables in FROM, JOIN, INTO, UPDATE, DELETE FROM clauses.
    """
    if not sql:
        return []

    tables: Set[str] = set()
    sql_upper = sql.upper()

    # Patterns to find table references
    patterns = [
        # FROM table or FROM schema.table
        r'\bFROM\s+([#@]?\w+(?:\.\w+)?)',
        # JOIN table or JOIN schema.table
        r'\bJOIN\s+([#@]?\w+(?:\.\w+)?)',
        # INTO table (for INSERT)
        r'\bINTO\s+([#@]?\w+(?:\.\w+)?)',
        # UPDATE table
        r'\bUPDATE\s+([#@]?\w+(?:\.\w+)?)',
        # DELETE FROM table
        r'\bDELETE\s+FROM\s+([#@]?\w+(?:\.\w+)?)',
        # TRUNCATE TABLE
        r'\bTRUNCATE\s+TABLE\s+([#@]?\w+(?:\.\w+)?)',
        # MERGE INTO
        r'\bMERGE\s+INTO\s+([#@]?\w+(?:\.\w+)?)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        for match in matches:
            # Clean up the table name
            table = match.strip()
            # Skip temp tables (#), variables (@), and common noise
            if table.startswith('#') or table.startswith('@'):
                continue
            # Skip SQL keywords, data types, and common noise words
            noise_words = {
                # SQL keywords
                'SET', 'WHERE', 'AND', 'OR', 'SELECT', 'AS', 'ON', 'WITH', 'BEGIN', 'END',
                'DECLARE', 'IF', 'ELSE', 'THEN', 'CASE', 'WHEN', 'NULL', 'NOT', 'EXISTS',
                'TOP', 'DISTINCT', 'ALL', 'ANY', 'SOME', 'OUTPUT', 'INSERTED', 'DELETED',
                # Data types
                'INT', 'BIGINT', 'SMALLINT', 'TINYINT', 'BIT', 'DECIMAL', 'NUMERIC',
                'FLOAT', 'REAL', 'MONEY', 'SMALLMONEY', 'DATETIME', 'DATETIME2', 'DATE',
                'TIME', 'DATETIMEOFFSET', 'SMALLDATETIME', 'CHAR', 'VARCHAR', 'NCHAR',
                'NVARCHAR', 'TEXT', 'NTEXT', 'BINARY', 'VARBINARY', 'IMAGE', 'XML',
                'UNIQUEIDENTIFIER', 'SQL_VARIANT', 'TIMESTAMP', 'CURSOR', 'TABLE',
                # Common noise from comments/code
                'BECAUSE', 'THE', 'THIS', 'THAT', 'FOR', 'ARE', 'WAS', 'WERE', 'BEEN',
                'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT',
                'MUST', 'SHALL', 'CAN', 'NEED', 'USED', 'USING', 'USE', 'RETURN', 'RETURNS',
                # System objects
                'SYS', 'INFORMATION_SCHEMA', 'SYSOBJECTS', 'SYSCOLUMNS', 'SYSINDEXES',
                # Schema names alone
                'DBO',
            }
            if table.upper() in noise_words:
                continue
            # Skip very short names (likely noise)
            if len(table) < 3:
                continue
            tables.add(table)

    return sorted(list(tables))


def extract_operations_from_sql(sql: str) -> List[str]:
    """
    Extract SQL operations from definition.
    """
    if not sql:
        return []

    operations: Set[str] = set()
    sql_upper = sql.upper()

    # Check for each operation type
    if re.search(r'\bSELECT\b', sql_upper):
        operations.add('SELECT')
    if re.search(r'\bINSERT\s+INTO\b', sql_upper):
        operations.add('INSERT')
    if re.search(r'\bUPDATE\b.*\bSET\b', sql_upper, re.DOTALL):
        operations.add('UPDATE')
    if re.search(r'\bDELETE\s+FROM\b', sql_upper):
        operations.add('DELETE')
    if re.search(r'\bEXEC(?:UTE)?\b', sql_upper):
        operations.add('EXEC')
    if re.search(r'\bTRUNCATE\b', sql_upper):
        operations.add('TRUNCATE')
    if re.search(r'\bMERGE\b', sql_upper):
        operations.add('MERGE')
    if re.search(r'\bCREATE\s+TABLE\b', sql_upper):
        operations.add('CREATE')
    if re.search(r'\bALTER\s+TABLE\b', sql_upper):
        operations.add('ALTER')
    if re.search(r'\bDROP\s+TABLE\b', sql_upper):
        operations.add('DROP')

    return sorted(list(operations))


def enrich_stored_procedures(dry_run: bool = False):
    """
    Enrich stored procedures with tables_affected and operations.

    Args:
        dry_run: If True, only show what would be updated without making changes
    """
    client = MongoClient(MONGO_URI, directConnection=True)
    db = client[DATABASE]
    collection = db[COLLECTION]

    # Find procedures that need enrichment
    # Either missing tables_affected or missing operations
    query = {
        '$or': [
            {'tables_affected': {'$exists': False}},
            {'tables_affected': []},
            {'operations': {'$exists': False}},
            {'operations': []}
        ]
    }

    procedures = list(collection.find(query))
    print(f"Found {len(procedures)} procedures needing enrichment")

    if dry_run:
        print("\n=== DRY RUN - No changes will be made ===\n")

    updated_count = 0
    skipped_count = 0

    for i, proc in enumerate(procedures):
        proc_name = proc.get('procedure_name', 'Unknown')
        definition = proc.get('definition', '')

        # Extract tables and operations
        tables = extract_tables_from_sql(definition)
        operations = extract_operations_from_sql(definition)

        # Check if we found anything new
        existing_tables = proc.get('tables_affected', [])
        existing_ops = proc.get('operations', [])

        new_tables = [t for t in tables if t not in existing_tables]
        new_ops = [o for o in operations if o not in existing_ops]

        if not new_tables and not new_ops and existing_tables and existing_ops:
            skipped_count += 1
            continue

        # Merge with existing
        final_tables = list(set(existing_tables + tables))
        final_ops = list(set(existing_ops + operations))

        if dry_run:
            if (i < 10) or (new_tables or new_ops):  # Show first 10 or any with changes
                print(f"\n{proc_name}:")
                print(f"  Tables: {final_tables}")
                print(f"  Operations: {final_ops}")
        else:
            # Update the document
            collection.update_one(
                {'_id': proc['_id']},
                {'$set': {
                    'tables_affected': final_tables,
                    'operations': final_ops
                }}
            )
            updated_count += 1

            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(procedures)} ({updated_count} updated)")

    if not dry_run:
        print(f"\nComplete!")
        print(f"  Updated: {updated_count}")
        print(f"  Skipped: {skipped_count}")
    else:
        print(f"\n=== DRY RUN Summary ===")
        print(f"  Would update: {len(procedures) - skipped_count}")
        print(f"  Would skip: {skipped_count}")

    # Show final stats
    print("\n=== Final Statistics ===")
    total = collection.count_documents({})
    has_tables = collection.count_documents({'tables_affected.0': {'$exists': True}})
    has_ops = collection.count_documents({'operations.0': {'$exists': True}})
    print(f"Total procedures: {total}")
    print(f"With tables_affected: {has_tables} ({has_tables*100//total}%)")
    print(f"With operations: {has_ops} ({has_ops*100//total}%)")

    client.close()


if __name__ == '__main__':
    import sys

    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    print("=" * 60)
    print("Stored Procedure Enrichment Script")
    print(f"MongoDB: {MONGO_URI}")
    print(f"Database: {DATABASE}")
    print(f"Collection: {COLLECTION}")
    print("=" * 60)

    enrich_stored_procedures(dry_run=dry_run)
