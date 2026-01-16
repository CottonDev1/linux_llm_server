# Claude Code Instructions - Python Services

## MongoDB Configuration

**CRITICAL: MongoDB is on port 27017**

```python
# CORRECT
MONGODB_URI = "mongodb://localhost:27017/?directConnection=true"

# WRONG - Do not use these:
# mongodb://EWRSPT-AI:27018  <- old remote server
# mongodb://localhost:27019   <- incorrect port
```

MongoDB runs in Docker on localhost:27017. Always use the environment variable from config.py:

```python
from config import MONGODB_URI, MONGODB_DATABASE
```

## Service Ports

| Service | Port |
|---------|------|
| Python FastAPI | 8001 |
| MongoDB | 27017 |
| LLM SQL | 8080 |
| LLM General | 8081 |
| LLM Code | 8082 |
| LLM Embedding | 8083 |

## Audio Pipeline

For audio transcription, set these environment variables:
```bash
SENSEVOICE_MODEL_PATH=/data/projects/llm_website/models/SenseVoiceSmall
CUDA_VISIBLE_DEVICES="1"  # Use GPU 1 if GPU 0 is full
```

## Prefect Configuration

**CRITICAL: Never run Prefect in ephemeral mode for testing**

Prefect must connect to the persistent server, not use temporary ephemeral servers. Before running any test flows:

```bash
# Set Prefect to use the local persistent server
prefect config set PREFECT_API_URL=http://localhost:4200/api

# Verify configuration (should NOT show ephemeral)
prefect config view
```

If you see `PREFECT_PROFILE='ephemeral'`, the configuration is wrong. Fix it:

```bash
# Create a persistent profile and configure it
prefect profile create local-server
prefect -p local-server config set PREFECT_API_URL=http://localhost:4200/api
prefect profile use local-server
```

The Prefect server runs on port 4200. All test flows should register runs to this server.

## Testing

Run tests with correct MongoDB:
```bash
MONGODB_URI="mongodb://localhost:27017/?directConnection=true" pytest
```

Run Prefect test flows (ensure server is running first):
```bash
# Start Prefect server if not running
prefect server start &

# Run test flows - they will register to http://localhost:4200
python prefect_pipelines/test_flows/sql_pipeline_test_flow.py
```
