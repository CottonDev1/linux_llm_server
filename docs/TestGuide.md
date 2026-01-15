# LLM Server Test Guide

This guide explains how to run the various test scripts and ensure all services are running correctly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Checking Service Status](#checking-service-status)
3. [Running Tests](#running-tests)
4. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before running tests, ensure you have:

- SSH access to the Ubuntu server (10.101.20.21)
- `jq` installed for JSON parsing: `sudo apt install jq`
- `curl` installed (usually pre-installed)

---

## Checking Service Status

### Check All LLM Services

```bash
# Check if all llama.cpp services are running
systemctl status llama-sql llama-general llama-code llama-embedding

# Quick check (active/inactive only)
systemctl is-active llama-sql llama-general llama-code llama-embedding
```

### Check Individual Services

| Service | Port | Command |
|---------|------|---------|
| SQL Model | 8080 | `systemctl status llama-sql` |
| General Model | 8081 | `systemctl status llama-general` |
| Code Model | 8082 | `systemctl status llama-code` |
| Embedding Model | 8083 | `systemctl status llama-embedding` |
| Python Service | 8001 | `systemctl status rag-python` (if configured) |
| Node.js Server | 3000 | `systemctl status rag-node` (if configured) |

### Start/Stop Services

```bash
# Start a service
sudo systemctl start llama-sql

# Stop a service
sudo systemctl stop llama-sql

# Restart a service
sudo systemctl restart llama-sql

# Start all LLM services
sudo systemctl start llama-sql llama-general llama-code llama-embedding

# Stop all LLM services
sudo systemctl stop llama-sql llama-general llama-code llama-embedding
```

### Check Service Logs

```bash
# View recent logs for a service
journalctl -u llama-sql -n 50

# Follow logs in real-time
journalctl -u llama-sql -f

# View logs since last boot
journalctl -u llama-sql -b
```

---

## Starting the Python Service

If the Python service is not running as a systemd service:

```bash
# Navigate to project directory
cd /data/projects/llm_website

# Activate virtual environment (if using one)
source venv/bin/activate

# Start the Python service
cd python_services
python main.py

# Or run in background
nohup python main.py > /tmp/python_service.log 2>&1 &
```

---

## Starting the Node.js Server

If the Node.js server is not running as a systemd service:

```bash
# Navigate to project directory
cd /data/projects/llm_website

# Install dependencies (first time only)
npm install

# Start the server
node rag-server.js

# Or run in background
nohup node rag-server.js > /tmp/node_server.log 2>&1 &
```

---

## Running Tests

### Test Audio Pipeline

Tests the complete audio processing workflow: upload, analyze, store, search, update, delete.

```bash
cd /data/projects/llm_website

# Run with a test audio file
./scripts/test_audio_pipeline.sh /path/to/audio.mp3

# Run with testing data
./scripts/test_audio_pipeline.sh testing_data/Audio/20251208-134920_114_\(901\)302-9431_Outgoing_Auto_2265013969051.mp3
```

**What it tests:**
- Audio file upload
- SenseVoice transcription
- Emotion detection
- LLM call analysis
- MongoDB storage
- Search functionality
- Record update/delete

### Test SQL Pipeline

Tests the text-to-SQL generation pipeline.

```bash
cd /data/projects/llm_website

# Run SQL pipeline tests
./scripts/test_sql_pipeline.sh
```

**What it tests:**
- Direct LLM completion (localhost:8080)
- Natural language to SQL conversion
- Rule matching
- Query generation speed

### Test Document Pipeline

Tests document processing and vector search.

```bash
cd /data/projects/llm_website

# Run with default test document
./scripts/test_document_pipeline.sh

# Run with specific document
./scripts/test_document_pipeline.sh testing_data/Reference/ProviderSystem/Cotton\ Bale\ Loan\ Processing.docx
```

**What it tests:**
- Embedding generation
- Document upload
- Chunking
- Vector storage
- Similarity search
- Document deletion

---

## Quick Health Checks

### Check All Endpoints

```bash
# LLM SQL Model
curl -s http://localhost:8080/health

# LLM General Model
curl -s http://localhost:8081/health

# LLM Code Model
curl -s http://localhost:8082/health

# LLM Embedding Model
curl -s http://localhost:8083/health

# Python Service
curl -s http://localhost:8001/health

# Node.js Server
curl -s http://localhost:3000/health
```

### Test LLM Completion

```bash
# Test SQL model
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "SELECT", "n_predict": 20}' | jq .

# Test General model
curl -s http://localhost:8081/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "n_predict": 50}' | jq .
```

### Test Embedding

```bash
curl -s http://localhost:8083/embedding \
  -H "Content-Type: application/json" \
  -d '{"content": "Test embedding text"}' | jq '.embedding | length'
```

---

## Troubleshooting

### Service Won't Start

1. Check logs: `journalctl -u llama-sql -n 100`
2. Check GPU availability: `nvidia-smi`
3. Check port availability: `ss -tlnp | grep 8080`
4. Check file permissions on model files

### Out of GPU Memory

```bash
# Check GPU memory usage
nvidia-smi

# Stop other models to free memory
sudo systemctl stop llama-code

# Restart the needed model
sudo systemctl restart llama-sql
```

### Slow Response Times

1. Check GPU utilization: `watch -n 1 nvidia-smi`
2. Check if model is loaded: First request after start is slow (model loading)
3. Consider using smaller models or reducing context size

### Connection Refused

1. Verify service is running: `systemctl is-active llama-sql`
2. Check if port is listening: `ss -tlnp | grep 8080`
3. Check firewall: `sudo ufw status`

---

## Test Data Location

Test data is stored in `/data/projects/llm_website/testing_data/`:

```
testing_data/
├── Audio/                    # Audio files for audio pipeline tests
│   ├── *.mp3
├── AudioPipelineTest/        # Additional audio test files
├── Reference/                # Documents for document pipeline tests
│   ├── ProviderSystem/
│   │   ├── *.docx
│   │   ├── *.pdf
│   │   └── *.xlsx
│   └── *.docx
└── *.md                      # Markdown test files
```

---

## Metrics and Logging

All test scripts output:
- Step-by-step progress
- Timing for each operation
- LLM model information
- Final metrics summary

Metrics are also saved to `/tmp/` for later analysis:
- `/tmp/audio_test_metrics.json` - Audio pipeline metrics
