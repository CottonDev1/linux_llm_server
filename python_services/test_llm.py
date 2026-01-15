#!/usr/bin/env python3
"""Test LLM connectivity."""
import urllib.request
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing before motor import...", flush=True)
try:
    req = urllib.request.Request("http://127.0.0.1:8080/v1/models")
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())
        print(f"Before motor: Success - {data}", flush=True)
except Exception as e:
    print(f"Before motor: Failed - {e}", flush=True)
    sys.exit(1)

# Now import motor and config
print("Importing config and motor...", flush=True)
import config
config.MONGODB_URI = 'mongodb://EWRSPT-AI:27018/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=10000'
from motor.motor_asyncio import AsyncIOMotorClient

print("Testing after motor import...", flush=True)
try:
    req2 = urllib.request.Request("http://127.0.0.1:8080/v1/models")
    with urllib.request.urlopen(req2, timeout=10) as response2:
        data2 = json.loads(response2.read().decode())
        print(f"After motor: Success - {data2}", flush=True)
except Exception as e:
    print(f"After motor: Failed - {e}", flush=True)
    sys.exit(1)

print("All tests passed!", flush=True)
