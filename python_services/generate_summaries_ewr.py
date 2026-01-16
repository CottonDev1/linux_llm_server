#!/usr/bin/env python3
"""
Generate AI summaries for EWR database stored procedures.
Tests LLM before any heavy imports.
"""
import urllib.request
import json
import sys

# Test LLM FIRST before any other imports
print("Testing LLM connectivity...", flush=True)
LLM_URL = 'http://127.0.0.1:8081/v1/chat/completions'
TEST_URL = 'http://127.0.0.1:8081/v1/models'

try:
    req = urllib.request.Request(TEST_URL)
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode())
        model_path = data['data'][0]['id']
        model_name = model_path.split('\\')[-1] if '\\' in model_path else model_path.split('/')[-1]
        print(f"LLM available: {model_name}", flush=True)
except Exception as e:
    print(f"LLM not available: {e}", flush=True)
    sys.exit(1)

# Now do the rest of the imports
print("Loading dependencies...", flush=True)
import os
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from motor.motor_asyncio import AsyncIOMotorClient

print("Dependencies loaded.", flush=True)

SUMMARY_PROMPT = """Analyze this SQL Server stored procedure and provide a brief summary in 2-3 sentences.

Procedure: {proc_name}
Parameters: {parameters}

Code:
{definition}

Summary:"""


def generate_summary(proc_name: str, definition: str, parameters: list) -> str:
    """Generate summary using LLM."""
    param_str = ', '.join([f"{p.get('name', '?')}" for p in parameters[:8]])
    if len(parameters) > 8:
        param_str += f" (+{len(parameters) - 8} more)"

    # Truncate definition
    max_len = 2500
    if len(definition) > max_len:
        definition = definition[:max_len] + "\n..."

    prompt = SUMMARY_PROMPT.format(
        proc_name=proc_name,
        parameters=param_str or "None",
        definition=definition
    )

    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.3
    }

    try:
        req = urllib.request.Request(
            LLM_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
            return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return None


async def main():
    client = AsyncIOMotorClient(config.MONGODB_URI)
    db = client['rag_server']
    coll = db.sql_stored_procedures

    # Target the 'ewr' database
    db_name = 'ewr'

    query = {
        'database': db_name,
        '$or': [
            {'ai_summary': {'$exists': False}},
            {'ai_summary': None},
            {'ai_summary': ''}
        ]
    }

    total = await coll.count_documents(query)
    print(f"\n{'='*50}", flush=True)
    print(f"{db_name}: {total} procedures need summaries", flush=True)
    print(f"{'='*50}", flush=True)

    if total == 0:
        print("No procedures need summaries!", flush=True)
        client.close()
        return

    count = 0
    errors = 0
    cursor = coll.find(query)

    async for proc in cursor:
        proc_name = proc.get('procedure_name', proc.get('name', 'Unknown'))
        print(f"[{count+errors+1}/{total}] {proc_name}...", end=' ', flush=True)

        summary = generate_summary(
            proc_name,
            proc.get('definition', ''),
            proc.get('parameters', [])
        )

        if summary:
            await coll.update_one(
                {'_id': proc['_id']},
                {'$set': {'ai_summary': summary}}
            )
            print(f"OK", flush=True)
            count += 1
        else:
            print(f"FAIL", flush=True)
            errors += 1

        time.sleep(0.3)

    print(f"\nDone: {count} summaries, {errors} errors", flush=True)

    client.close()
    print("\nAll done!", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
