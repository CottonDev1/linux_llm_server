#!/usr/bin/env python3
"""
Generate AI summaries for EWR stored procedures using the LLM.
Uses localhost:8081 (general model) for summary generation.
"""
import urllib.request
import json
import sys
import os

# Use LLM endpoint from environment or default to localhost
LLM_HOST = os.environ.get('LLAMACPP_HOST', 'http://localhost:8081')
LLM_URL = f'{LLM_HOST}/v1/chat/completions'
TEST_URL = f'{LLM_HOST}/v1/models'

print("Testing VM LLM connectivity...", flush=True)
try:
    req = urllib.request.Request(TEST_URL)
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode())
        model_path = data['data'][0]['id']
        model_name = model_path.split('\\')[-1] if '\\' in model_path else model_path.split('/')[-1]
        print(f"VM LLM available: {model_name}", flush=True)
except Exception as e:
    print(f"VM LLM not available: {e}", flush=True)
    sys.exit(1)

print("Loading dependencies...", flush=True)
import os
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from motor.motor_asyncio import AsyncIOMotorClient

print("Dependencies loaded.", flush=True)

SUMMARY_PROMPT = """Analyze this SQL Server stored procedure and provide a brief summary in 2-3 sentences.
Describe what the procedure does, its key inputs/outputs, and main tables affected.

Procedure: {proc_name}

Definition:
{definition}

Parameters:
{parameters}

Summary:"""


def generate_summary(proc_name: str, definition: str, parameters: list) -> str:
    """Generate summary using VM's LLM."""
    max_len = 3000
    if len(definition) > max_len:
        definition = definition[:max_len] + "\n..."

    param_str = ", ".join([f"@{p.get('name', 'unknown')} {p.get('data_type', 'unknown')}"
                          for p in parameters[:10]]) if parameters else "None"

    prompt = SUMMARY_PROMPT.format(
        proc_name=proc_name,
        definition=definition,
        parameters=param_str
    )

    payload = {
        "model": "qwen2.5-7b-instruct",
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
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
            return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return None


async def main():
    client = AsyncIOMotorClient(config.MONGODB_URI)
    db = client['rag_server']
    coll = db.sql_stored_procedures

    # Process EWR procedures in DESCENDING order (Z to A)
    # The main script processes ascending (A to Z), so we work from opposite ends
    query = {
        'database': 'ewr',
        '$or': [
            {'ai_summary': {'$exists': False}},
            {'ai_summary': None},
            {'ai_summary': ''}
        ]
    }

    total = await coll.count_documents(query)
    print(f"\n{'='*50}", flush=True)
    print(f"EWR (VM - Z to A): {total} need summaries", flush=True)
    print(f"{'='*50}", flush=True)

    if total == 0:
        print("No procedures need summaries!", flush=True)
        client.close()
        return

    count = 0
    errors = 0
    # Sort DESCENDING to work from opposite end as main script
    cursor = coll.find(query).sort('procedure_name', -1)

    async for proc in cursor:
        proc_name = proc.get('procedure_name', 'Unknown')
        print(f"[VM {count+errors+1}/{total}] {proc_name}...", end=' ', flush=True)

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

        time.sleep(0.2)

    print(f"\nDone: {count} summaries, {errors} errors", flush=True)
    client.close()


if __name__ == '__main__':
    asyncio.run(main())
