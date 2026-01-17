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

## Shared Services (Multi-Session Safety)

**CRITICAL: NEVER stop or restart shared services unless the user EXPLICITLY requests it.**

Multiple Claude Code sessions may be using these services simultaneously:
- Python FastAPI service (port 8001)
- Node.js server (port 3000)
- LLM servers (ports 8080-8083)

### Rules

1. **NEVER** kill, stop, or restart these services on your own
2. **Only START** a service if it's not running and needed for the task
3. If a service needs restarting, **ASK the user first**
4. If code changes require a service restart, **inform the user** that they need to restart manually

### Starting the Python Service (only if not running)

```bash
# Check first
curl -s http://localhost:8001/status && echo "Already running" || {
  cd /data/projects/llm_website/python_services
  source venv/bin/activate
  python -m uvicorn main:app --host 0.0.0.0 --port 8001
}
```

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

## Git Commits

- Do NOT include any co-author, attribution, or comments regarding Claude/AI in commit messages
- Keep commit messages succinct
