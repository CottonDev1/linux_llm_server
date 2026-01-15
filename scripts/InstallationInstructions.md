# RAG Server Installation Guide

This guide covers installation of the RAG Server on both Windows and Linux systems.

---

## Table of Contents

- [Windows Installation](#windows-installation)
  - [Prerequisites](#windows-prerequisites)
  - [LlamaCpp Setup](#llamacpp-setup)
  - [Node.js and Python Services](#windows-services)
- [Linux Installation (Systemd)](#linux-installation-systemd)
  - [Prerequisites](#linux-prerequisites)
  - [Service Configuration](#service-configuration)
  - [Management Commands](#management-commands)
- [Troubleshooting](#troubleshooting)

---

## Windows Installation

### Windows Prerequisites

- Windows 10/11 or Windows Server 2019+
- PowerShell 5.1+
- Node.js 18+ (LTS recommended)
- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA 12.2+ for GPU acceleration

### LlamaCpp Setup

The RAG Server uses three llama.cpp model servers:

| Service | Port | Model | Purpose |
|---------|------|-------|---------|
| LlamaCppSQL | 8080 | nsql-llama-2-7b | SQL query generation |
| LlamaCppGeneral | 8081 | Qwen2.5-7B-Instruct | General LLM tasks |
| LlamaCppCode | 8082 | qwen2.5-coder-1.5b | Code assistance |

#### Automatic Installation

Run the installation script as Administrator:

```powershell
# Open PowerShell as Administrator
cd C:\Projects\llm_website\scripts\Install\llamaCpp
.\install-llamacpp.ps1
```

The script will:
1. Download pre-built llama.cpp binaries (CUDA or CPU version auto-detected)
2. Download the three required GGUF models (~15GB total)
3. Install NSSM (Non-Sucking Service Manager)
4. Create and start Windows services

#### Script Options

```powershell
# Full installation with defaults
.\install-llamacpp.ps1

# Custom install directory
.\install-llamacpp.ps1 -InstallDir "D:\llama.cpp"

# Skip model download (if models already exist)
.\install-llamacpp.ps1 -SkipModelDownload

# Skip service installation (just download binaries/models)
.\install-llamacpp.ps1 -SkipServiceInstall

# Force reinstall everything
.\install-llamacpp.ps1 -Force
```

#### Manual Installation

If you prefer to install manually:

1. **Download llama.cpp** from [GitHub Releases](https://github.com/ggerganov/llama.cpp/releases)
   - CUDA version: `llama-bXXXX-bin-win-cuda-cu12.2.0-x64.zip`
   - CPU version: `llama-bXXXX-bin-win-avx2-x64.zip`

2. **Download models** from HuggingFace:
   ```powershell
   $ModelsDir = "C:\Projects\llm_website\models\llamacpp"

   # SQL Model
   Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/nsql-llama-2-7B-GGUF/resolve/main/nsql-llama-2-7b.Q4_K_M.gguf" -OutFile "$ModelsDir\nsql-llama-2-7b.Q4_K_M.gguf"

   # General Model
   Invoke-WebRequest -Uri "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf" -OutFile "$ModelsDir\Qwen2.5-7B-Instruct-Q4_K_M.gguf"

   # Code Model
   Invoke-WebRequest -Uri "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" -OutFile "$ModelsDir\qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
   ```

3. **Start servers manually**:
   ```powershell
   # SQL Model (port 8080)
   llama-server.exe -m "C:\Projects\llm_website\models\llamacpp\nsql-llama-2-7b.Q4_K_M.gguf" --host 0.0.0.0 --port 8080 -c 4096

   # General Model (port 8081)
   llama-server.exe -m "C:\Projects\llm_website\models\llamacpp\Qwen2.5-7B-Instruct-Q4_K_M.gguf" --host 0.0.0.0 --port 8081 -c 8192

   # Code Model (port 8082)
   llama-server.exe -m "C:\Projects\llm_website\models\llamacpp\qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" --host 0.0.0.0 --port 8082 -c 8192
   ```

#### Service Management (Windows)

```powershell
# Start all LlamaCpp services
Start-Service LlamaCppSQL, LlamaCppGeneral, LlamaCppCode

# Stop all services
Stop-Service LlamaCppSQL, LlamaCppGeneral, LlamaCppCode

# Restart all services
Restart-Service LlamaCppSQL, LlamaCppGeneral, LlamaCppCode

# Check status
Get-Service LlamaCpp*

# View logs
Get-Content C:\llama.cpp\logs\LlamaCppSQL.log -Tail 50
Get-Content C:\llama.cpp\logs\LlamaCppGeneral.log -Tail 50
Get-Content C:\llama.cpp\logs\LlamaCppCode.log -Tail 50
```

#### Verify LlamaCpp Installation

```powershell
# Test each endpoint
Invoke-RestMethod -Uri "http://localhost:8080/health"  # SQL Model
Invoke-RestMethod -Uri "http://localhost:8081/health"  # General Model
Invoke-RestMethod -Uri "http://localhost:8082/health"  # Code Model
```

### Windows Services

After LlamaCpp is running, start the Node.js and Python services:

```powershell
# Start Python service
cd C:\Projects\llm_website\python_services
.\venv\Scripts\activate
python main.py

# Start Node.js server (in another terminal)
cd C:\Projects\llm_website
node rag-server.js
```

---

## Linux Installation (Systemd)

### Linux Prerequisites

- Ubuntu 20.04+ / Debian 11+ / RHEL 8+
- Node.js 18+ (LTS)
- Python 3.10+
- systemd

### Service Configuration

| Service | Description | Port |
|---------|-------------|------|
| `rag-node` | Node.js backend server | 3000 |
| `rag-python` | Python API services | 8001 |

### Installation Steps

#### 1. Prepare the installation directory

```bash
# Create installation directory
sudo mkdir -p /opt/rag-server

# Copy application files
sudo cp -r /path/to/llm_website/* /opt/rag-server/

# Set ownership
sudo chown -R www-data:www-data /opt/rag-server
```

#### 2. Install dependencies

```bash
cd /opt/rag-server
sudo -u www-data ./scripts/Install/install.sh
```

#### 3. Install systemd services

```bash
# Copy service files
sudo cp scripts/Install/systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable rag-node rag-python

# Start services
sudo systemctl start rag-node rag-python
```

### Management Commands

```bash
# Start services
sudo systemctl start rag-node rag-python

# Stop services
sudo systemctl stop rag-node rag-python

# Restart services
sudo systemctl restart rag-node rag-python

# Check status
sudo systemctl status rag-node rag-python

# View logs
sudo journalctl -u rag-node -f
sudo journalctl -u rag-python -f

# View logs since boot
sudo journalctl -u rag-node -b
sudo journalctl -u rag-python -b
```

### Using the Management Script

A convenience script is provided:

```bash
# Start all services
./scripts/Install/systemd/rag-ctl.sh start

# Stop all services
./scripts/Install/systemd/rag-ctl.sh stop

# Restart all services
./scripts/Install/systemd/rag-ctl.sh restart

# Check status
./scripts/Install/systemd/rag-ctl.sh status

# View logs (follow mode)
./scripts/Install/systemd/rag-ctl.sh logs
```

### Configuration

#### Environment Variables

Edit the service files to customize environment variables:

**rag-node.service:**
- `NODE_ENV` - Node environment (default: production)
- `PORT` - Server port (default: 3000)

**rag-python.service:**
- `HOST` - Bind address (default: 0.0.0.0)
- `PORT` - Server port (default: 8001)

#### Installation Path

The default installation path is `/opt/rag-server`. To change it:

1. Edit both `.service` files
2. Update `WorkingDirectory` and `ExecStart` paths
3. Update `ReadWritePaths` if needed
4. Run `sudo systemctl daemon-reload`

#### User/Group

Services run as `www-data:www-data` by default. To change:

1. Edit the `User` and `Group` lines in both service files
2. Ensure the new user has access to the installation directory
3. Run `sudo systemctl daemon-reload`

---

## Troubleshooting

### LlamaCpp Issues

#### Service won't start
```powershell
# Check service status
Get-Service LlamaCppSQL

# Check logs
Get-Content C:\llama.cpp\logs\LlamaCppSQL-error.log -Tail 100

# Test running manually
cd C:\llama.cpp
.\llama-server.exe -m "C:\Projects\llm_website\models\llamacpp\nsql-llama-2-7b.Q4_K_M.gguf" --port 8080
```

#### CUDA errors
If you see CUDA-related errors but don't have a compatible GPU:
```powershell
# Reinstall with CPU-only version
.\install-llamacpp.ps1 -Force
# The script auto-detects CUDA availability
```

#### Out of memory
Reduce context size or use smaller models:
```powershell
# Edit service to reduce context size
nssm edit LlamaCppSQL
# Change -c 4096 to -c 2048
```

### Linux Service Issues

#### Service won't start
```bash
# Check detailed status
sudo systemctl status rag-node -l

# Check logs for errors
sudo journalctl -u rag-node --no-pager -n 50
```

#### Permission errors
```bash
# Fix ownership
sudo chown -R www-data:www-data /opt/rag-server

# Check SELinux (if enabled)
sudo audit2why < /var/log/audit/audit.log
```

#### Port already in use
```bash
# Find process using the port
sudo lsof -i :3000
sudo lsof -i :8001

# Kill if needed
sudo kill -9 <PID>
```

---

## Quick Reference

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Node.js RAG Server | 3000 | Main API server |
| Python Services | 8001 | Python API endpoints |
| LlamaCpp SQL | 8080 | SQL generation model |
| LlamaCpp General | 8081 | General purpose model |
| LlamaCpp Code | 8082 | Code assistance model |
| MongoDB | 27018 | Database (remote: EWRSPT-AI) |

### Health Check URLs

```bash
# Node.js
curl http://localhost:3000/health

# Python
curl http://localhost:8001/health

# LlamaCpp
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8082/health
```
