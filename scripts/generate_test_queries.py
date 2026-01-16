"""
Generate human language test queries for all tables in target schemas.
Creates a markdown document with queries for pipeline testing.
"""

import json
import os
import re
from pymongo import MongoClient
from typing import List, Dict, Optional

# Lookup table patterns to skip
LOOKUP_PATTERNS = [
    'types', 'status', 'statuses', 'category', 'categories',
    'codes', 'lookups', 'settings', 'config', 'configuration',
    'sysdiagrams', 'aspnet_', '__ef', 'migrations'
]

# Database to test server/database mapping
DATABASE_MAPPING = {
    'ewrcentral': {
        'test_server': 'EWRSQLTEST',
        'test_database': 'EWRCentral',
        'auth': 'windows'
    },
    'ewr': {
        'test_server': 'EWRSQLTEST',
        'test_database': 'EWR',
        'auth': 'windows'
    },
    'ewr.gin.entity': {
        'test_server': 'NCSQLTEST',
        'test_database': 'EWR.Gin.Bobby:B-ADOBE',  # Customer database with data
        'auth': 'sql'
    },
    'ewr.marketing.entity': {
        'test_server': 'NCSQLTEST',
        'test_database': 'EWR.Marketing.Adobe',  # Customer database with data
        'auth': 'sql'
    },
    'ewr.warehouse.entity': {
        'test_server': 'NCSQLTEST',
        'test_database': 'EWR.Warehouse.Bobby:B_EDITPOST',  # Customer database with data
        'auth': 'sql'
    }
}

def is_lookup_table(table_name: str, summary: str) -> bool:
    """Check if a table is a lookup/reference table."""
    name_lower = table_name.lower()

    # Check name patterns
    for pattern in LOOKUP_PATTERNS:
        if pattern in name_lower:
            return True

    # Check summary for lookup indicators
    if summary:
        summary_lower = summary.lower()
        if any(x in summary_lower for x in ['lookup', 'reference', 'enum', 'constant', 'type code']):
            return True

    return False

def extract_entity_name(table_name: str) -> str:
    """Extract entity name from table name."""
    # Remove dbo. prefix
    name = table_name.replace('dbo.', '')

    # Split CamelCase
    words = re.findall(r'[A-Z][a-z]+|[a-z]+', name)
    if words:
        return ' '.join(words).lower()
    return name.lower()

def generate_question(table_name: str, summary: str, columns: List[Dict], foreign_keys: List[Dict]) -> str:
    """Generate a human language question for a table based on its schema."""
    entity = extract_entity_name(table_name)

    # Check for date columns
    date_cols = [c['name'] for c in columns if any(x in c['name'].lower() for x in ['date', 'time', 'utc'])]

    # Check for user/person columns
    user_cols = [c['name'] for c in columns if any(x in c['name'].lower() for x in ['userid', 'user', 'person', 'employee', 'createdby', 'assignedto'])]

    # Check for status columns
    status_cols = [c['name'] for c in columns if any(x in c['name'].lower() for x in ['status', 'state', 'type'])]

    # Check for amount/count columns
    amount_cols = [c['name'] for c in columns if any(x in c['name'].lower() for x in ['amount', 'total', 'count', 'quantity', 'price', 'cost'])]

    # Check for name/description columns
    name_cols = [c['name'] for c in columns if any(x in c['name'].lower() for x in ['name', 'title', 'description'])]

    # Generate question based on available columns
    questions = []

    # Basic listing question
    if name_cols:
        questions.append(f"List all {entity} with their names")
    else:
        questions.append(f"Show me all {entity}")

    # Date-based question
    if date_cols:
        questions.append(f"Show me {entity} created in the last month")
        questions.append(f"How many {entity} were added this year")

    # User-based question
    if user_cols:
        questions.append(f"Show me {entity} grouped by user")

    # Status-based question
    if status_cols:
        questions.append(f"Show me {entity} by status")

    # Amount-based question
    if amount_cols:
        questions.append(f"What is the total amount for all {entity}")

    # Foreign key question
    if foreign_keys:
        fk = foreign_keys[0]
        ref_table = fk.get('referenced_table', '')
        if ref_table:
            ref_entity = extract_entity_name(ref_table)
            questions.append(f"Show me {entity} with their related {ref_entity}")

    # Choose the most interesting question
    if len(questions) > 1:
        # Prefer more specific questions
        for q in questions:
            if 'grouped' in q or 'total' in q or 'related' in q or 'month' in q:
                return q

    return questions[0]

def main():
    """Generate test queries document."""
    print("Connecting to MongoDB...")
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongodb_uri)
    db = client['rag_server']
    tables_collection = db['sql_schema_context']

    # Output document
    output_lines = [
        "# LLM Pipeline Test Queries",
        "",
        "This document contains human language queries generated from table schemas for testing the SQL pipeline.",
        "",
        "## Format",
        "| Schema Database | Test Database | Table Name | Human Language Query | Test Result |",
        "|-----------------|---------------|------------|---------------------|-------------|",
        ""
    ]

    stats = {'total': 0, 'lookup_skipped': 0, 'generated': 0}

    for schema_db, mapping in DATABASE_MAPPING.items():
        print(f"\nProcessing {schema_db}...")

        # Get tables for this database
        tables = list(tables_collection.find(
            {'database': schema_db},
            {'table_name': 1, 'summary': 1, 'columns': 1, 'foreign_keys': 1, 'purpose': 1}
        ))

        for table in tables:
            stats['total'] += 1
            table_name = table.get('table_name', '')
            summary = table.get('summary', '') or table.get('purpose', '')
            columns = table.get('columns', [])
            foreign_keys = table.get('foreign_keys', [])

            # Skip lookup tables
            if is_lookup_table(table_name, summary):
                stats['lookup_skipped'] += 1
                continue

            # Generate question
            question = generate_question(table_name, summary, columns, foreign_keys)

            # Add to output
            output_lines.append(
                f"| {schema_db} | {mapping['test_database']} | {table_name} | {question} | PENDING |"
            )
            stats['generated'] += 1

        print(f"  Tables: {len(tables)}, Queries generated: {stats['generated']}")

    # Add summary
    output_lines.extend([
        "",
        "## Summary",
        f"- Total tables: {stats['total']}",
        f"- Lookup tables skipped: {stats['lookup_skipped']}",
        f"- Queries generated: {stats['generated']}",
        "",
        "## Server Connection Details",
        "",
        "### EWRSQLTEST (EWRCentral, EWR)",
        "- Server: EWRSQLTEST",
        "- Auth: Windows (EWR\\chad.walker)",
        "- Password: 6454@@Christina",
        "",
        "### NCSQLTEST (EWR.Gin.*, EWR.Marketing.*, EWR.Warehouse.*)",
        "- Server: NCSQLTEST",
        "- Auth: SQL (EWRUser)",
        "- Password: 66a3904d69",
    ])

    # Write output
    output_path = 'C:/projects/llm_website/docs/llm_pipeline_test_queries.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n=== Complete ===")
    print(f"Output: {output_path}")
    print(f"Total tables: {stats['total']}")
    print(f"Lookup tables skipped: {stats['lookup_skipped']}")
    print(f"Queries generated: {stats['generated']}")

    client.close()

if __name__ == '__main__':
    main()
