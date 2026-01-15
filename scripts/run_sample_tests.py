"""
Run a sample of pipeline tests from each database.
Tests a representative sample for quick validation.
"""

import sys
import json
import time
import requests
from typing import Dict, List
from datetime import datetime

# API Configuration
PYTHON_API_BASE = "http://localhost:8001"
SQL_QUERY_ENDPOINT = f"{PYTHON_API_BASE}/api/sql/query"

# Database to server/credentials mapping
DATABASE_CONFIG = {
    'ewrcentral': {
        'server': 'EWRSQLTEST',
        'database': 'EWRCentral',
        'domain': 'EWR',
        'username': 'chad.walker',
        'password': '6454@@Christina',
    },
    'ewr': {
        'server': 'EWRSQLTEST',
        'database': 'EWR',
        'domain': 'EWR',
        'username': 'chad.walker',
        'password': '6454@@Christina',
    },
    'ewr.gin.entity': {
        'server': 'NCSQLTEST',
        'database': 'EWR.Gin.Bobby:B-ADOBE',
        'domain': '',
        'username': 'EWRUser',
        'password': '66a3904d69',
    },
    'ewr.marketing.entity': {
        'server': 'NCSQLTEST',
        'database': 'EWR.Marketing.Adobe',
        'domain': '',
        'username': 'EWRUser',
        'password': '66a3904d69',
    },
    'ewr.warehouse.entity': {
        'server': 'NCSQLTEST',
        'database': 'EWR.Warehouse.Bobby:B_EDITPOST',
        'domain': '',
        'username': 'EWRUser',
        'password': '66a3904d69',
    }
}

def parse_queries(filepath: str) -> Dict[str, List[Dict]]:
    """Parse queries from markdown and group by database."""
    queries_by_db = {db: [] for db in DATABASE_CONFIG.keys()}

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for line in content.split('\n'):
        if line.startswith('|') and 'PENDING' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                db = parts[1].lower()
                if db in queries_by_db:
                    queries_by_db[db].append({
                        'schema_db': parts[1],
                        'test_db': parts[2],
                        'table_name': parts[3],
                        'question': parts[4],
                    })

    return queries_by_db

def test_query(query: Dict, timeout: int = 90) -> Dict:
    """Test a single query."""
    schema_db = query['schema_db'].lower()
    config = DATABASE_CONFIG.get(schema_db)

    if not config:
        return {'success': False, 'error': f"No config for: {schema_db}", 'sql': None}

    payload = {
        'naturalLanguage': query['question'],
        'database': query['test_db'],
        'server': config['server'],
        'username': config['username'],
        'password': config['password'],
        'execute': True
    }
    if config['domain']:
        payload['domain'] = config['domain']

    try:
        response = requests.post(SQL_QUERY_ENDPOINT, json=payload, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            return {
                'success': result.get('success', False) and not result.get('error'),
                'sql': result.get('sql'),
                'error': result.get('error'),
                'matched_rules': result.get('matched_rules', [])
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}", 'sql': None}
    except Exception as e:
        return {'success': False, 'error': str(e), 'sql': None}

def main():
    """Run sample tests."""
    doc_path = 'C:/projects/llm_website/docs/llm_pipeline_test_queries.md'
    results_path = 'C:/projects/llm_website/docs/sample_test_results.json'

    samples_per_db = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("=" * 60, flush=True)
    print("LLM Pipeline Sample Test Runner", flush=True)
    print(f"Testing {samples_per_db} queries per database", flush=True)
    print("=" * 60, flush=True)

    queries_by_db = parse_queries(doc_path)

    all_results = {}
    stats = {'total': 0, 'passed': 0, 'failed': 0}
    failed_queries = []

    for db_name, queries in queries_by_db.items():
        sample = queries[:samples_per_db]
        print(f"\n{db_name}: Testing {len(sample)} queries", flush=True)
        print("-" * 40, flush=True)

        for i, q in enumerate(sample, 1):
            stats['total'] += 1
            key = f"{db_name}|{q['table_name']}"

            print(f"  [{i}/{len(sample)}] {q['table_name']}", flush=True)
            result = test_query(q)

            if result['success']:
                stats['passed'] += 1
                print(f"    PASSED", flush=True)
            else:
                stats['failed'] += 1
                error = result.get('error', 'Unknown')[:50]
                print(f"    FAILED: {error}", flush=True)
                failed_queries.append({
                    'database': db_name,
                    'table': q['table_name'],
                    'question': q['question'],
                    'error': result.get('error'),
                    'sql': result.get('sql')
                })

            all_results[key] = result

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total:  {stats['total']}", flush=True)
    print(f"Passed: {stats['passed']} ({100*stats['passed']/stats['total']:.1f}%)", flush=True)
    print(f"Failed: {stats['failed']} ({100*stats['failed']/stats['total']:.1f}%)", flush=True)

    if failed_queries:
        print(f"\nFailed Queries ({len(failed_queries)}):", flush=True)
        for fq in failed_queries[:20]:
            print(f"  - {fq['database']}: {fq['table']}", flush=True)
            print(f"    Q: {fq['question'][:60]}...", flush=True)
            print(f"    Error: {fq['error'][:60] if fq['error'] else 'None'}...", flush=True)

    # Save results
    with open(results_path, 'w') as f:
        json.dump({
            'stats': stats,
            'failed_queries': failed_queries,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

if __name__ == '__main__':
    main()
