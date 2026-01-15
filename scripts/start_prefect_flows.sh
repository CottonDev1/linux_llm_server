#!/bin/bash
# Start Prefect Test Flows
# This script avoids module collision by setting PYTHONPATH correctly

cd /data/projects/llm_website
source python_services/venv/bin/activate

# Ensure prefect package is found before local testing/prefect
export PYTHONPATH="/data/projects/llm_website/python_services/venv/lib/python3.12/site-packages:/data/projects/llm_website:/data/projects/llm_website/testing"

exec python -m testing.prefect.serve_flows
