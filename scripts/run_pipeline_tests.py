"""
Run pipeline tests for all queries in the test document.
Tests each query through the Python API and validates results.
"""

import re
import json
import time
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# API Configuration
PYTHON_API_BASE = "http://localhost:8001"
NODE_API_BASE = "http://localhost:3000"
SQL_QUERY_ENDPOINT = f"{PYTHON_API_BASE}/api/sql/query"

# Database to server/credentials mapping
DATABASE_CONFIG = {
    'ewrcentral': {
        'server': 'EWRSQLTEST',
        'database': 'EWRCentral',
        'domain': 'EWR',
        'username': 'chad.walker',
        'password': '6454@@Christina',
        'auth_type': 'windows'
    },
    'ewr': {
        'server': 'EWRSQLTEST',
        'database': 'EWR',
        'domain': 'EWR',
        'username': 'chad.walker',
        'password': '6454@@Christina',
        'auth_type': 'windows'
    },
    'ewr.gin.entity': {
        'server': 'NCSQLTEST',
        'database': 'EWR.Gin.Bobby:B-ADOBE',
        'domain': '',
        'username': 'EWRUser',
        'password': '66a3904d69',
        'auth_type': 'sql'
    },
    'ewr.marketing.entity': {
        'server': 'NCSQLTEST',
        'database': 'EWR.Marketing.Adobe',
        'domain': '',
        'username': 'EWRUser',
        'password': '66a3904d69',
        'auth_type': 'sql'
    },
    'ewr.warehouse.entity': {
        'server': 'NCSQLTEST',
        'database': 'EWR.Warehouse.Bobby:B_EDITPOST',
        'domain': '',
        'username': 'EWRUser',
        'password': '66a3904d69',
        'auth_type': 'sql'
    }
}

def parse_test_document(filepath: str) -> List[Dict]:
    """Parse the test queries document."""
    queries = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find table rows
    lines = content.split('\n')
    for line in lines:
        if line.startswith('|') and 'PENDING' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                queries.append({
                    'schema_db': parts[1],
                    'test_db': parts[2],
                    'table_name': parts[3],
                    'question': parts[4],
                    'status': parts[5],
                    'original_line': line
                })

    return queries

def test_query(query: Dict, timeout: int = 90) -> Dict:
    """Test a single query through the pipeline."""
    schema_db = query['schema_db'].lower()
    config = DATABASE_CONFIG.get(schema_db)

    if not config:
        return {
            'success': False,
            'error': f"No config for database: {schema_db}",
            'sql': None,
            'result_count': 0
        }

    # Build request payload for Python API
    payload = {
        'naturalLanguage': query['question'],
        'database': query['test_db'],
        'server': config['server'],
        'username': config['username'],
        'password': config['password'],
        'execute': True  # Execute the query
    }

    if config['domain']:
        payload['domain'] = config['domain']

    try:
        response = requests.post(
            SQL_QUERY_ENDPOINT,
            json=payload,
            timeout=timeout
        )

        if response.status_code == 200:
            result = response.json()
            # Check for success
            success = result.get('success', False) and not result.get('error')

            # Count results if execution was performed
            result_count = 0
            if result.get('execution_result'):
                exec_result = result.get('execution_result')
                if isinstance(exec_result, dict) and 'rows' in exec_result:
                    result_count = len(exec_result.get('rows', []))
                elif isinstance(exec_result, list):
                    result_count = len(exec_result)

            return {
                'success': success,
                'sql': result.get('sql'),
                'error': result.get('error'),
                'result_count': result_count,
                'matched_rules': result.get('matched_rules', []),
                'is_exact_match': result.get('is_exact_match', False)
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text[:200]}",
                'sql': None,
                'result_count': 0
            }

    except requests.Timeout:
        return {
            'success': False,
            'error': 'Request timeout (90s)',
            'sql': None,
            'result_count': 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'sql': None,
            'result_count': 0
        }

def update_document(filepath: str, results: Dict[str, Dict]) -> None:
    """Update the test document with results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    updated_lines = []

    for line in lines:
        if line.startswith('|') and ('PENDING' in line or 'PASSED' in line or 'FAILED' in line):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                key = f"{parts[1]}_{parts[3]}_{parts[4]}"
                if key in results:
                    result = results[key]
                    status = 'PASSED' if result['success'] else 'FAILED'
                    parts[5] = status
                    line = '| ' + ' | '.join(parts[1:]) + ' |'

        updated_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))

def main():
    """Run all pipeline tests."""
    doc_path = 'C:/projects/llm_website/docs/llm_pipeline_test_queries.md'
    results_path = 'C:/projects/llm_website/docs/pipeline_test_results.json'

    print("=" * 60)
    print("LLM Pipeline Test Runner")
    print("=" * 60)

    # Check API availability
    print("\nChecking API availability...")
    try:
        response = requests.get(f"{PYTHON_API_BASE}/api/sql/health", timeout=5)
        print(f"Python SQL API: OK ({response.status_code})")
    except Exception as e:
        print(f"Python SQL API: FAILED - {e}")
        print("\nPlease start the Python service first:")
        print("  cd python_services && python main.py")
        return 1

    # Parse test document
    print(f"\nParsing test document: {doc_path}")
    queries = parse_test_document(doc_path)
    print(f"Found {len(queries)} queries to test")

    # Group by database for progress tracking
    by_db = {}
    for q in queries:
        db = q['schema_db']
        if db not in by_db:
            by_db[db] = []
        by_db[db].append(q)

    print("\nQueries by database:")
    for db, qs in by_db.items():
        print(f"  {db}: {len(qs)}")

    # Run tests
    results = {}
    stats = {'passed': 0, 'failed': 0, 'total': len(queries)}
    start_time = time.time()

    print("\n" + "=" * 60)
    print("Running tests...")
    print("=" * 60)

    for i, query in enumerate(queries, 1):
        key = f"{query['schema_db']}_{query['table_name']}_{query['question']}"

        print(f"\n[{i}/{len(queries)}] {query['schema_db']}: {query['table_name']}")
        print(f"  Question: {query['question'][:60]}...")

        result = test_query(query)
        results[key] = result

        if result['success']:
            stats['passed'] += 1
            print(f"  Result: PASSED (rows: {result['result_count']})")
            if result['sql']:
                print(f"  SQL: {result['sql'][:80]}...")
        else:
            stats['failed'] += 1
            print(f"  Result: FAILED")
            print(f"  Error: {result.get('error', 'Unknown')[:80]}")

        # Save progress periodically
        if i % 25 == 0:
            update_document(doc_path, results)
            with open(results_path, 'w') as f:
                json.dump({
                    'stats': stats,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"\n  Progress saved: {stats['passed']}/{i} passed")

    # Final save
    duration = time.time() - start_time
    update_document(doc_path, results)

    final_results = {
        'stats': stats,
        'duration_seconds': duration,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total queries: {stats['total']}")
    print(f"Passed: {stats['passed']} ({100*stats['passed']/stats['total']:.1f}%)")
    print(f"Failed: {stats['failed']} ({100*stats['failed']/stats['total']:.1f}%)")
    print(f"Duration: {duration:.1f} seconds")
    print(f"\nResults saved to: {results_path}")
    print(f"Document updated: {doc_path}")

    return 0 if stats['failed'] == 0 else 1

if __name__ == '__main__':
    exit(main())
