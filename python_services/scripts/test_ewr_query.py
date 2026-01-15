"""
Test script to verify SQL rules service and execution against EWR database.

Tests:
1. SQL rules context building from MongoDB
2. Query execution against EWRTSQL3

Run with: python -m scripts.test_ewr_query
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pymssql


async def test_rules_service():
    """Test the SQL rules service for EWR database."""
    print("=" * 60)
    print("Testing SQL Rules Service (MongoDB)")
    print("=" * 60)

    from services.sql_rules_service import get_sql_rules_service

    service = await get_sql_rules_service()

    question = "How many contacts are in the contact table"
    database = "EWR"

    print(f"\nQuestion: {question}")
    print(f"Database: {database}")
    print("-" * 60)

    # Check for exact match
    exact_match = await service.find_exact_match(question, database)
    if exact_match:
        print(f"\nFound exact match rule: {exact_match['rule_id']}")
        print(f"SQL: {exact_match['sql']}")
        return exact_match['sql']

    # Build rules context
    table_names = {"Contacts"}
    context = await service.build_rules_context(question, database, table_names)
    print(f"\nRules context generated: {len(context)} chars")

    # Since no exact match, return a sensible SQL for this query
    # The rules service provides guidance but the LLM generates the actual SQL
    # For this test, we'll use a standard count query
    sql = "SELECT COUNT(*) AS ContactCount FROM Contacts"
    print(f"\nUsing standard count query: {sql}")

    return sql


def execute_query(sql: str):
    """Execute query against EWRTSQL3."""
    print("\n" + "=" * 60)
    print("Executing Query Against EWRTSQL3")
    print("=" * 60)

    # Connection parameters (same setup as EWRSQLPROD)
    conn_params = {
        'server': 'EWRTSQL3',
        'user': 'EWR\\chad.walker',
        'password': '6454@@Christina',
        'database': 'EWR',
        'tds_version': '7.3',
        'login_timeout': 30,
    }

    print(f"\nConnecting to: {conn_params['server']}")
    print(f"Database: {conn_params['database']}")
    print(f"User: {conn_params['user']}")

    try:
        conn = pymssql.connect(**conn_params)
        cursor = conn.cursor()

        print(f"\nExecuting SQL:")
        print(f"  {sql}")

        cursor.execute(sql)
        rows = cursor.fetchall()

        print(f"\nResults:")
        # Get column names
        if cursor.description:
            columns = [col[0] for col in cursor.description]
            print(f"  Columns: {columns}")

        for row in rows:
            print(f"  {row}")

        conn.close()
        print("\nConnection closed successfully")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        return False


async def main():
    """Main test function."""
    # Step 1: Test rules service
    sql = await test_rules_service()

    # Step 2: Execute the SQL
    success = execute_query(sql)

    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("EXECUTION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
