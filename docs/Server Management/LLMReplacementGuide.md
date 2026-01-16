# LLM Model Replacement Guide

This guide explains how to change the LLM models used by the server and restart services correctly.

## Table of Contents

1. [Current Model Configuration](#current-model-configuration)
2. [Available Model Slots](#available-model-slots)
3. [Downloading New Models](#downloading-new-models)
4. [Changing a Model](#changing-a-model)
5. [Restarting Services](#restarting-services)
6. [Verifying the New Model](#verifying-the-new-model)
7. [Rollback Procedure](#rollback-procedure)

---

## Current Model Configuration

| Service | Port | Model | GPU | Purpose |
|---------|------|-------|-----|---------|
| llama-sql | 8080 | qwen2.5-coder-7b-instruct-q4_k_m.gguf | GPU 0 (RTX 3050) | SQL generation |
| llama-general | 8081 | qwen2.5-3b-instruct-q4_k_m.gguf | GPU 1 (GTX 1660) | General chat |
| llama-code | 8082 | qwen2.5-coder-1.5b-instruct-q8_0.gguf | GPU 1 (GTX 1660) | Code assistance |
| llama-embedding | 8083 | nomic-embed-text-v1.5.Q8_0.gguf | GPU 1 (GTX 1660) | Text embeddings |

**Model Directory:** `/data/models/`

**Service Files:** `/etc/systemd/system/llama-*.service`

---

## Available Model Slots

### GPU 0 (RTX 3050 - 8GB VRAM)
- Currently used by: SQL Model (~6GB)
- Available for: Models up to ~7B Q4 quantized

### GPU 1 (GTX 1660 - 6GB VRAM)
- Currently used by: General + Code + Embedding (~4GB total)
- Available for: Smaller models, shared GPU

---

## Downloading New Models

### From Hugging Face

```bash
# Install huggingface-cli if not installed
pip install huggingface-hub

# Download a model
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --local-dir /data/models/
```

### Manual Download

```bash
cd /data/models

# Using wget
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Using curl
curl -L -o model.gguf "https://huggingface.co/..."
```

### Recommended Models by Use Case

| Use Case | Recommended Model | Size | Notes |
|----------|------------------|------|-------|
| SQL Generation | Qwen2.5-Coder-7B | ~4.5GB | Best SQL accuracy |
| General Chat | Qwen2.5-3B | ~2GB | Fast, good quality |
| Code Assistance | Qwen2.5-Coder-1.5B | ~1.6GB | Fast code completion |
| Embeddings | nomic-embed-text-v1.5 | ~275MB | 768-dim embeddings |

---

## Changing a Model

### Step 1: Stop the Service

```bash
sudo systemctl stop llama-sql
```

### Step 2: Edit the Service File

```bash
sudo nano /etc/systemd/system/llama-sql.service
```

### Step 3: Update the Model Path

Find the `ExecStart` line and change the model path:

```ini
# Before
ExecStart=/data/projects/llama.cpp/build/bin/llama-server \
    -m /data/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --ctx-size 8192

# After (example: switching to Mistral)
ExecStart=/data/projects/llama.cpp/build/bin/llama-server \
    -m /data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --ctx-size 8192
```

### Step 4: Reload and Restart

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Start the service
sudo systemctl start llama-sql

# Check status
sudo systemctl status llama-sql
```

---

## Service File Template

Here's a complete service file template:

```ini
[Unit]
Description=LLaMA SQL Model Server
After=network.target

[Service]
Type=simple
User=chad
WorkingDirectory=/data/projects/llama.cpp
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/data/projects/llama.cpp/build/bin/llama-server \
    -m /data/models/YOUR_MODEL_HERE.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --threads 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `-m` | Model file path | Full path to .gguf file |
| `--port` | Server port | 8080-8083 |
| `--n-gpu-layers` | GPU layers | 99 (all layers) |
| `--ctx-size` | Context window | 4096-8192 |
| `--threads` | CPU threads | 4 |
| `CUDA_VISIBLE_DEVICES` | GPU to use | 0 or 1 |

---

## Restarting Services

### Restart Single Service

```bash
sudo systemctl restart llama-sql
```

### Restart All LLM Services

```bash
# Create a restart script
cat > /tmp/restart_all_llms.sh << 'EOF'
#!/bin/bash
echo "Stopping all LLM services..."
sudo systemctl stop llama-sql llama-general llama-code llama-embedding

echo "Waiting for GPU memory to clear..."
sleep 5

echo "Starting LLM services..."
sudo systemctl start llama-sql
sleep 3
sudo systemctl start llama-general
sleep 3
sudo systemctl start llama-code
sleep 3
sudo systemctl start llama-embedding

echo "Checking status..."
systemctl is-active llama-sql llama-general llama-code llama-embedding

echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
EOF

chmod +x /tmp/restart_all_llms.sh
/tmp/restart_all_llms.sh
```

### Quick Restart Script

A convenience script is available:

```bash
# Restart all services
./scripts/restart_all_llms.sh
```

---

## Verifying the New Model

### Check Service Status

```bash
systemctl status llama-sql
```

### Check Model Loading

```bash
# View startup logs
journalctl -u llama-sql -n 50 | grep -i "model\|loaded\|error"
```

### Test the Model

```bash
# Send a test request
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "SELECT * FROM", "n_predict": 20}' | jq .
```

### Check GPU Memory

```bash
nvidia-smi
```

### Run Full Test Suite

```bash
./scripts/test_sql_pipeline.sh
```

---

## Rollback Procedure

If the new model doesn't work correctly:

### Step 1: Stop the Service

```bash
sudo systemctl stop llama-sql
```

### Step 2: Restore Previous Configuration

```bash
# Edit service file
sudo nano /etc/systemd/system/llama-sql.service

# Change model path back to previous model
# Save and exit
```

### Step 3: Restart

```bash
sudo systemctl daemon-reload
sudo systemctl start llama-sql
```

### Step 4: Verify

```bash
curl -s http://localhost:8080/health
./scripts/test_sql_pipeline.sh
```

---

## Troubleshooting

### Model Won't Load

1. Check file exists: `ls -la /data/models/your-model.gguf`
2. Check file permissions: `chmod 644 /data/models/your-model.gguf`
3. Check logs: `journalctl -u llama-sql -n 100`

### Out of GPU Memory

1. Use smaller quantization (Q4_K_M instead of Q8_0)
2. Reduce context size (`--ctx-size 4096`)
3. Move to different GPU
4. Stop other models

### Slow Performance

1. Check GPU layers: Ensure `--n-gpu-layers 99`
2. Check GPU utilization: `nvidia-smi`
3. Consider smaller model

### Service Keeps Crashing

1. Check logs for errors: `journalctl -u llama-sql -f`
2. Try running manually first:
   ```bash
   /data/projects/llama.cpp/build/bin/llama-server \
     -m /data/models/your-model.gguf \
     --host 0.0.0.0 --port 8080
   ```

---

## GPU Assignment Reference

| GPU | ID | VRAM | Best For |
|-----|-----|------|----------|
| RTX 3050 | 0 | 8GB | Large models (7B Q4) |
| GTX 1660 | 1 | 6GB | Smaller models, shared |

### Changing GPU Assignment

Edit the service file and change `CUDA_VISIBLE_DEVICES`:

```ini
# Use GPU 0 (RTX 3050)
Environment="CUDA_VISIBLE_DEVICES=0"

# Use GPU 1 (GTX 1660)
Environment="CUDA_VISIBLE_DEVICES=1"
```

---

## Model Compatibility Notes

- **GGUF format required** - llama.cpp only supports GGUF
- **Quantization matters** - Q4_K_M is good balance of size/quality
- **Context size** - Larger context = more memory
- **Embedding models** - Must output fixed-dimension vectors
