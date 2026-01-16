#!/usr/bin/env python3
"""
Generate AI summaries for stored procedures that don't have them.
Uses llama.cpp server on localhost:8080.
"""
import os
import sys
import asyncio
import urllib.request
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from motor.motor_asyncio import AsyncIOMotorClient

LLM_URL = 'http://127.0.0.1:8081/v1/chat/completions'
LLM_MODEL = 'qwen2.5-7b-instruct'

SUMMARY_PROMPT = """Analyze this SQL Server stored procedure and provide a brief summary.

Procedure Name: {proc_name}
Parameters: {parameters}

Definition:
{definition}

Provide a concise summary (2-3 sentences) describing:
1. What this procedure does
2. Key inputs/outputs
3. Main tables affected

Summary:"""


def generate_summary(proc_name: str, definition: str, parameters: list) -> str:
    """Generate summary using LLM."""
    param_str = ', '.join([f"{p.get('name', 'unknown')} ({p.get('type', 'unknown')})" for p in parameters[:10]])
    if len(parameters) > 10:
        param_str += f" ... and {len(parameters) - 10} more"

    # Truncate definition if too long
    max_def_len = 3000
    if len(definition) > max_def_len:
        definition = definition[:max_def_len] + "\n... [truncated]"

    prompt = SUMMARY_PROMPT.format(
        proc_name=proc_name,
        parameters=param_str or "None",
        definition=definition
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3
    }

    try:
        req = urllib.request.Request(
            LLM_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"  LLM request failed: {e}")
        return None


async def process_database(db_name: str):
    """Process all procedures in a database that don't have summaries."""
    client = AsyncIOMotorClient(config.MONGODB_URI)
    db = client['rag_server']
    collection = db.sql_stored_procedures

    # Find procedures without summaries
    query = {
        'database': db_name,
        '$or': [
            {'ai_summary': {'$exists': False}},
            {'ai_summary': None},
            {'ai_summary': ''}
        ]
    }

    total = await collection.count_documents(query)
    print(f"\n{'='*60}")
    print(f"Processing {db_name}: {total} procedures need summaries")
    print(f"{'='*60}")

    if total == 0:
        print("All procedures already have summaries!")
        client.close()
        return 0, 0

    processed = 0
    errors = 0

    cursor = collection.find(query)

    async for proc in cursor:
        proc_name = proc.get('procedure_name', 'Unknown')
        definition = proc.get('definition', '')
        parameters = proc.get('parameters', [])

        print(f"[{processed+errors+1}/{total}] {proc_name}...", end=' ', flush=True)

        summary = generate_summary(proc_name, definition, parameters)

        if summary:
            await collection.update_one(
                {'_id': proc['_id']},
                {'$set': {'ai_summary': summary}}
            )
            print(f"OK ({len(summary)} chars)")
            processed += 1
        else:
            print("FAILED")
            errors += 1

        # Small delay to not overwhelm LLM
        time.sleep(0.2)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {processed} summaries generated, {errors} errors")
    print(f"{'='*60}")

    client.close()
    return processed, errors


async def main():
    # Check LLM availability first
    print("Checking LLM availability...")
    try:
        req = urllib.request.Request('http://127.0.0.1:8081/v1/models')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            model_path = data['data'][0]['id']
            model_name = model_path.split('\\')[-1] if '\\' in model_path else model_path.split('/')[-1]
            print(f"LLM available: {model_name}")
    except Exception as e:
        print(f"LLM not available: {e}")
        sys.exit(1)

    # Process each database
    databases = ['ewr_warehouse_entity', 'ewr_gin_entity', 'ewr_marketing_entity']

    total_processed = 0
    total_errors = 0

    for db_name in databases:
        processed, errors = await process_database(db_name)
        total_processed += processed
        total_errors += errors

    print(f"\n{'='*60}")
    print(f"ALL DONE: {total_processed} total summaries, {total_errors} total errors")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())
