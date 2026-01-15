#!/usr/bin/env python3
"""
Generate LLM summaries for schema and stored procedures in parallel.
Uses asyncio to process multiple items concurrently.
"""
import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first, then override the MongoDB URI
import config
config.MONGODB_URI = 'mongodb://EWRSPT-AI:27018/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=10000'

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime


async def check_llm_available(llm_url: str) -> bool:
    """Check if LLM is available"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{llm_url}/v1/models")
            return response.status_code == 200
    except:
        return False


async def generate_schema_summary(
    llm_url: str,
    table_name: str,
    schema_info: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Generate summary for a single table/view"""
    async with semaphore:
        try:
            columns_desc = ", ".join([
                f"{c['name']} ({c['type']})"
                for c in schema_info.get('columns', [])[:15]  # Limit columns
            ])

            pk_desc = ", ".join(schema_info.get('primaryKeys', []))
            related = ", ".join(schema_info.get('relatedTables', [])[:5])
            obj_type = schema_info.get('objectType', 'table')

            prompt = f"""Analyze this SQL Server {obj_type} and provide a brief summary.

{obj_type.upper()}: {table_name}
COLUMNS: {columns_desc}
PRIMARY KEYS: {pk_desc or 'None'}
RELATED TABLES: {related or 'None'}

Respond with ONLY a JSON object (no markdown):
{{"summary": "1-2 sentence description of what this {obj_type} stores/does", "purpose": "business purpose", "keywords": ["keyword1", "keyword2", "keyword3"]}}"""

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{llm_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.1
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    # Try to parse JSON from response
                    import json
                    import re
                    # Extract JSON from response
                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception as e:
            print(f"   Error summarizing {table_name}: {e}", flush=True)
        return None


async def generate_procedure_summary(
    llm_url: str,
    proc_name: str,
    proc_info: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Generate summary for a single stored procedure"""
    async with semaphore:
        try:
            params = ", ".join([
                f"{p['name']} {p['type']}"
                for p in proc_info.get('parameters', [])[:10]
            ])

            # Get first 500 chars of definition
            definition = proc_info.get('definition', '')[:500]

            prompt = f"""Analyze this SQL Server stored procedure and provide a brief summary.

PROCEDURE: {proc_name}
PARAMETERS: {params or 'None'}
DEFINITION (partial): {definition}...

Respond with ONLY a JSON object (no markdown):
{{"summary": "1-2 sentence description of what this procedure does", "operations": ["SELECT", "UPDATE", etc], "keywords": ["keyword1", "keyword2"]}}"""

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{llm_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.1
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    import json
                    import re
                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception as e:
            print(f"   Error summarizing {proc_name}: {e}", flush=True)
        return None


async def process_schemas(
    mongo_db,
    collection_name: str,
    database_key: str,
    llm_url: str,
    concurrency: int = 5
) -> int:
    """Process all schemas without summaries"""
    from config import COLLECTION_SQL_SCHEMA_CONTEXT

    collection = mongo_db[collection_name]

    # Find records without summaries
    cursor = collection.find({
        "database": database_key,
        "$or": [
            {"summary": {"$exists": False}},
            {"summary": None},
            {"summary": ""}
        ]
    })

    records = await cursor.to_list(length=None)
    total = len(records)

    if total == 0:
        print(f"   No schema records need summaries", flush=True)
        return 0

    print(f"   Processing {total} schema records ({concurrency} concurrent)...", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    updated = 0

    async def process_one(record):
        nonlocal updated
        table_name = record.get('table_name', 'unknown')
        schema_info = {
            'columns': record.get('columns', []),
            'primaryKeys': record.get('primary_keys', []),
            'relatedTables': record.get('related_tables', []),
            'objectType': record.get('object_type', 'table')
        }

        result = await generate_schema_summary(llm_url, table_name, schema_info, semaphore)

        if result:
            await collection.update_one(
                {"_id": record["_id"]},
                {"$set": {
                    "summary": result.get('summary', ''),
                    "purpose": result.get('purpose', ''),
                    "keywords": result.get('keywords', []),
                    "summary_generated_at": datetime.utcnow()
                }}
            )
            updated += 1
            print(f"   [{updated}/{total}] {table_name}", flush=True)

    # Process in batches
    tasks = [process_one(r) for r in records]
    await asyncio.gather(*tasks)

    return updated


async def process_procedures(
    mongo_db,
    collection_name: str,
    database_key: str,
    llm_url: str,
    concurrency: int = 5
) -> int:
    """Process all procedures without summaries"""

    collection = mongo_db[collection_name]

    cursor = collection.find({
        "database": database_key,
        "$or": [
            {"summary": {"$exists": False}},
            {"summary": None},
            {"summary": ""}
        ]
    })

    records = await cursor.to_list(length=None)
    total = len(records)

    if total == 0:
        print(f"   No procedure records need summaries", flush=True)
        return 0

    print(f"   Processing {total} procedure records ({concurrency} concurrent)...", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    updated = 0

    async def process_one(record):
        nonlocal updated
        proc_name = record.get('procedure_name', 'unknown')
        proc_info = {
            'parameters': record.get('parameters', []),
            'definition': record.get('definition', '')
        }

        result = await generate_procedure_summary(llm_url, proc_name, proc_info, semaphore)

        if result:
            await collection.update_one(
                {"_id": record["_id"]},
                {"$set": {
                    "summary": result.get('summary', ''),
                    "operations": result.get('operations', []),
                    "keywords": result.get('keywords', []),
                    "summary_generated_at": datetime.utcnow()
                }}
            )
            updated += 1
            print(f"   [{updated}/{total}] {proc_name}", flush=True)

    tasks = [process_one(r) for r in records]
    await asyncio.gather(*tasks)

    return updated


async def main():
    parser = argparse.ArgumentParser(description='Generate LLM summaries for schema data')
    parser.add_argument('--database', '-d', default='ewrcentral', help='Database key')
    parser.add_argument('--llm-url', default='http://localhost:8081', help='LLM API URL')
    parser.add_argument('--concurrency', '-c', type=int, default=5, help='Concurrent requests')
    parser.add_argument('--schemas-only', action='store_true', help='Only process schemas')
    parser.add_argument('--procedures-only', action='store_true', help='Only process procedures')
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"Generating LLM Summaries", flush=True)
    print(f"Database: {args.database}", flush=True)
    print(f"LLM: {args.llm_url}", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Check LLM availability
    print("Checking LLM availability...", flush=True)
    if not await check_llm_available(args.llm_url):
        print(f"ERROR: LLM not available at {args.llm_url}", flush=True)
        print("Please ensure the LLM service is running.", flush=True)
        sys.exit(1)
    print("LLM is available!\n", flush=True)

    # Initialize MongoDB
    from mongodb import get_mongodb_service
    from config import COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES

    mongo = get_mongodb_service()
    await mongo.initialize()

    schema_count = 0
    proc_count = 0

    if not args.procedures_only:
        print("Processing schema summaries...", flush=True)
        schema_count = await process_schemas(
            mongo.db,
            COLLECTION_SQL_SCHEMA_CONTEXT,
            args.database,
            args.llm_url,
            args.concurrency
        )

    if not args.schemas_only:
        print("\nProcessing stored procedure summaries...", flush=True)
        proc_count = await process_procedures(
            mongo.db,
            COLLECTION_SQL_STORED_PROCEDURES,
            args.database,
            args.llm_url,
            args.concurrency
        )

    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE: {schema_count} schemas, {proc_count} procedures summarized", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
