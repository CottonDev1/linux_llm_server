#!/usr/bin/env python3
"""
Generate AI summaries for Entity database views.
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
config.MONGODB_URI = 'mongodb://EWRSPT-AI:27018/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=10000'

from motor.motor_asyncio import AsyncIOMotorClient

print("Dependencies loaded.", flush=True)

SUMMARY_PROMPT = """Analyze this SQL Server view and provide a brief summary in 2-3 sentences explaining what data it exposes and its purpose.

View: {view_name}

Definition:
{definition}

Summary:"""


def generate_summary(view_name: str, definition: str) -> str:
    """Generate summary using LLM."""
    # Truncate definition
    max_len = 2500
    if len(definition) > max_len:
        definition = definition[:max_len] + "\n..."

    prompt = SUMMARY_PROMPT.format(
        view_name=view_name,
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
    coll = db.sql_views

    # Target the entity databases
    databases = ['ewr_gin_entity', 'ewr_marketing_entity', 'ewr_warehouse_entity']

    for db_name in databases:
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
        print(f"{db_name}: {total} views need summaries", flush=True)
        print(f"{'='*50}", flush=True)

        if total == 0:
            print("No views need summaries!", flush=True)
            continue

        count = 0
        errors = 0
        cursor = coll.find(query)

        async for view in cursor:
            view_name = view.get('view_name', 'Unknown')
            print(f"[{count+errors+1}/{total}] {view_name}...", end=' ', flush=True)

            summary = generate_summary(
                view_name,
                view.get('definition', '')
            )

            if summary:
                await coll.update_one(
                    {'_id': view['_id']},
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
