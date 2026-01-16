#!/usr/bin/env python3
"""
Generate AI summaries - writes to log file directly.
"""
import urllib.request
import json
import sys
import os
import time
from datetime import datetime

# Log file
LOG_FILE = "C:/Projects/llm_website/logs/summary_generation.log"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# Clear log
with open(LOG_FILE, "w") as f:
    f.write(f"=== Summary Generation Started {datetime.now()} ===\n")

log("Testing LLM connectivity...")
LLM_URL = 'http://127.0.0.1:8081/v1/chat/completions'
TEST_URL = 'http://127.0.0.1:8081/v1/models'

try:
    req = urllib.request.Request(TEST_URL)
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode())
        model_path = data['data'][0]['id']
        model_name = model_path.split('\\')[-1] if '\\' in model_path else model_path.split('/')[-1]
        log(f"LLM available: {model_name}")
except Exception as e:
    log(f"LLM not available: {e}")
    sys.exit(1)

# Now imports
log("Loading dependencies...")
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from motor.motor_asyncio import AsyncIOMotorClient

log("Dependencies loaded.")

SUMMARY_PROMPT = """Analyze this SQL stored procedure briefly.

Name: {proc_name}
Params: {parameters}

Code:
{definition}

Give 2 sentences: what it does and main tables used."""


def generate_summary(proc_name: str, definition: str, parameters: list) -> str:
    param_str = ', '.join([p.get('name', '?') for p in parameters[:6]])
    if len(parameters) > 6:
        param_str += f" (+{len(parameters) - 6})"

    max_len = 2000
    if len(definition) > max_len:
        definition = definition[:max_len] + "..."

    prompt = SUMMARY_PROMPT.format(
        proc_name=proc_name,
        parameters=param_str or "None",
        definition=definition
    )

    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.2
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
        log(f"  ERROR: {e}")
        return None


async def main():
    client = AsyncIOMotorClient(config.MONGODB_URI)
    db = client['rag_server']
    coll = db.sql_stored_procedures

    databases = ['ewr_warehouse_entity', 'ewr_gin_entity', 'ewr_marketing_entity']

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
        log(f"")
        log(f"{'='*50}")
        log(f"{db_name}: {total} need summaries")
        log(f"{'='*50}")

        if total == 0:
            continue

        count = 0
        errors = 0
        cursor = coll.find(query).limit(50)  # Process 50 at a time

        async for proc in cursor:
            proc_name = proc.get('procedure_name', 'Unknown')
            log(f"[{count+errors+1}/50] {proc_name}...")

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
                log(f"  -> OK: {summary[:80]}...")
                count += 1
            else:
                log(f"  -> FAILED")
                errors += 1

            time.sleep(0.5)

        log(f"Batch done: {count} ok, {errors} errors")

    client.close()
    log("=== COMPLETE ===")


if __name__ == '__main__':
    asyncio.run(main())
