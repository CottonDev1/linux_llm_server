# Claude Code Instructions

## MongoDB Configuration

**IMPORTANT: MongoDB runs on port 27017 on localhost**

```
MongoDB URI: mongodb://localhost:27017/?directConnection=true
Database: rag_server
```

MongoDB runs in Docker on this server. Do NOT use:
- Port 27018 (old Atlas Local configuration)
- Port 27019 (incorrect fallback)
- Remote hostnames like EWRSPT-AI (use localhost)

The correct connection string is configured in `.env`:
```
MONGODB_URI=mongodb://localhost:27017/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=10000
MONGODB_DATABASE=rag_server
```

## Project Overview

This is the SQL Pipeline Improvements branch for the LLM RAG Server project. Key components:

- **Python Services** (`python_services/`): FastAPI backend on port 8001
- **Node.js Server** (`rag-server.js`): Web frontend on port 3000
- **LLM Models**: llama.cpp servers on ports 8080-8083

## Audio Pipeline

SenseVoice transcription requires:
```
SENSEVOICE_MODEL_PATH=/data/projects/llm_website/models/SenseVoiceSmall
```

GPU 1 (GTX 1660) has ~2GB free for SenseVoice. Use `CUDA_VISIBLE_DEVICES="1"` if GPU 0 is full.

## Known Issues

1. **pyannote diarization**: API changed, `token` parameter no longer supported
2. **Many scripts have hardcoded wrong ports**: See list below

### Files with Incorrect MongoDB Ports (Need Fixing)

Port 27018 (should be 27017):
- `python_services/extract_*.py` files
- `python_services/generate_summaries*.py` files
- `python_services/llm/service.py`
- `python_services/llm/client.py`
- `python_services/test_llm.py`
- `python_services/tests/pipelines/sql/conftest.py`
- `scripts/prefect_pipelines/test_flows/` files

Port 27019 (should be 27017):
- `python_services/prefect_pipelines/agent_learning_flow.py`
- `python_services/prefect_pipelines/sp_analysis_flow.py`
- `python_services/prefect_pipelines/examples/sp_analysis_example.py`
