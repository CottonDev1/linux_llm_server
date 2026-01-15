# Embedding Service Setup for EWRSPT-AI
# Installs llama.cpp with nomic-embed-text-v1.5 as a Windows service
# Also installs an API wrapper to match the expected embedding service interface
#
# Architecture:
#   [Your App] -> [API Wrapper :8084] -> [llama.cpp :8083]
#
# Run this script ON EWRSPT-AI as Administrator:
#   .\SetupEmbeddingService-EWRSPTAI.ps1
#
# Or run remotely from CHAD-PC:
#   Invoke-Command -ComputerName EWRSPT-AI -FilePath .\scripts\SetupEmbeddingService-EWRSPTAI.ps1

#Requires -RunAsAdministrator

param(
    [switch]$Force,
    [int]$LlamaCppPort = 8083,    # Internal llama.cpp port
    [int]$ApiPort = 8084          # External API port (use in EMBEDDING_SERVICE_URL)
)

$ErrorActionPreference = "Stop"

# Configuration
$InstallDir = "C:\tools\llama.cpp"
$ModelsDir = "C:\tools\llama.cpp\models"
$LogsDir = "C:\tools\llama.cpp\logs"
$WrapperDir = "C:\tools\embedding-wrapper"
$NssmPath = "C:\tools\nssm\nssm.exe"
$PythonPath = "C:\Python312\python.exe"  # Python 3.12 on EWRSPT-AI

# llama.cpp release - CPU-only (AVX2) version for servers without GPU
$ReleaseTag = "b4547"
$LlamaCppUrl = "https://github.com/ggerganov/llama.cpp/releases/download/$ReleaseTag/llama-$ReleaseTag-bin-win-avx2-x64.zip"

# nomic-embed-text-v1.5 GGUF model (768 dimensions, 137M params)
$ModelName = "nomic-embed-text-v1.5.f16.gguf"
$ModelUrl = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf"

# Service names
$LlamaCppServiceName = "LlamaCpp-Embedding"
$WrapperServiceName = "Embedding-API-Wrapper"

function Write-Status {
    param([string]$Message, [string]$Color = "Green")
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $Message" -ForegroundColor $Color
}

function Stop-Services {
    Write-Status "Stopping existing services..."

    foreach ($serviceName in @($LlamaCppServiceName, $WrapperServiceName)) {
        $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        if ($service -and $service.Status -eq "Running") {
            Write-Status "  Stopping $serviceName..." -Color Yellow
            Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    }

    # Kill any processes on ports
    foreach ($port in @($LlamaCppPort, $ApiPort)) {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            Write-Status "  Killing process on port $port..." -Color Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }

    Start-Sleep -Seconds 2
}

function Install-LlamaCpp {
    Write-Status "Installing llama.cpp..."

    foreach ($dir in @($InstallDir, $ModelsDir, $LogsDir)) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    $serverExe = "$InstallDir\llama-server.exe"

    if ((Test-Path $serverExe) -and -not $Force) {
        Write-Status "  llama-server.exe already exists (use -Force to reinstall)"
        return
    }

    # Download
    $zipPath = "$env:TEMP\llama-cpp.zip"
    Write-Status "  Downloading llama.cpp ($ReleaseTag)..."
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $LlamaCppUrl -OutFile $zipPath -UseBasicParsing

    # Extract
    Write-Status "  Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $InstallDir -Force

    # Move files from nested directory if needed
    $nestedDir = Get-ChildItem -Path $InstallDir -Directory | Select-Object -First 1
    if ($nestedDir -and (Test-Path "$($nestedDir.FullName)\llama-server.exe")) {
        Get-ChildItem -Path $nestedDir.FullName | Move-Item -Destination $InstallDir -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $nestedDir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue

    if (Test-Path $serverExe) {
        Write-Status "  llama.cpp installed successfully"
    } else {
        throw "llama-server.exe not found after extraction"
    }
}

function Download-Model {
    Write-Status "Downloading embedding model..."

    $modelPath = "$ModelsDir\$ModelName"

    if ((Test-Path $modelPath) -and -not $Force) {
        $size = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
        Write-Status "  Model already exists: $ModelName ($size MB)"
        return
    }

    Write-Status "  Downloading nomic-embed-text-v1.5 (~270 MB)..." -Color Yellow

    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $ModelUrl -OutFile $modelPath -UseBasicParsing

    $size = [math]::Round((Get-Item $modelPath).Length / 1MB, 2)
    Write-Status "  Model downloaded: $size MB"
}

function Install-Wrapper {
    Write-Status "Installing API wrapper..."

    if (-not (Test-Path $WrapperDir)) {
        New-Item -ItemType Directory -Path $WrapperDir -Force | Out-Null
    }

    # Create the wrapper Python script
    $wrapperScript = @'
"""
Embedding API Wrapper for llama.cpp
Translates the expected API format to llama.cpp's format.
"""
import asyncio
import logging
import os
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://localhost:8083")
TIMEOUT = 30.0
MODEL_NAME = "nomic-embed-text-v1.5"
DIMENSIONS = 768

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding API Wrapper", version="1.0.0")

class EmbedRequest(BaseModel):
    text: str

class EmbedBatchRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embedding: List[float]

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]

class HealthResponse(BaseModel):
    status: str
    model: str
    dimensions: int

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLAMACPP_URL}/health")
            if response.status_code == 200:
                return HealthResponse(status="ok", model=MODEL_NAME, dimensions=DIMENSIONS)
            raise HTTPException(status_code=503, detail="llama.cpp unhealthy")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"llama.cpp unavailable: {e}")

@app.post("/embed", response_model=EmbedResponse)
async def embed_single(request: EmbedRequest):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{LLAMACPP_URL}/embedding",
                json={"content": request.text}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            data = response.json()
            embedding = data.get("embedding", [])
            if not embedding:
                raise HTTPException(status_code=500, detail="No embedding returned")
            return EmbedResponse(embedding=embedding)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"llama.cpp unavailable: {e}")

@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    if not request.texts:
        return EmbedBatchResponse(embeddings=[])
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            semaphore = asyncio.Semaphore(10)
            async def embed_one(text: str) -> List[float]:
                async with semaphore:
                    response = await client.post(
                        f"{LLAMACPP_URL}/embedding",
                        json={"content": text}
                    )
                    if response.status_code != 200:
                        raise HTTPException(status_code=response.status_code, detail=response.text)
                    return response.json().get("embedding", [])
            embeddings = await asyncio.gather(*[embed_one(t) for t in request.texts])
            return EmbedBatchResponse(embeddings=list(embeddings))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"llama.cpp unavailable: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8084"))
    uvicorn.run(app, host="0.0.0.0", port=port)
'@

    $wrapperScript | Out-File -FilePath "$WrapperDir\embedding_wrapper.py" -Encoding UTF8

    # Create requirements.txt
    @"
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.0.0
"@ | Out-File -FilePath "$WrapperDir\requirements.txt" -Encoding UTF8

    # Install Python dependencies
    Write-Status "  Installing Python dependencies..."
    & $PythonPath -m pip install -r "$WrapperDir\requirements.txt" --quiet 2>&1 | Out-Null

    Write-Status "  API wrapper installed"
}

function Remove-OldService {
    param([string]$ServiceName)

    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        Write-Status "  Removing old service: $ServiceName" -Color Yellow
        & $NssmPath stop $ServiceName 2>$null
        Start-Sleep -Seconds 2
        & $NssmPath remove $ServiceName confirm 2>$null
        Start-Sleep -Seconds 2
    }
}

function Create-LlamaCppService {
    Write-Status "Creating llama.cpp service..."

    $serverExe = "$InstallDir\llama-server.exe"
    $modelPath = "$ModelsDir\$ModelName"

    if (-not (Test-Path $modelPath)) {
        throw "Model not found: $modelPath"
    }

    Remove-OldService -ServiceName $LlamaCppServiceName

    # Build arguments for embedding server
    $arguments = @(
        "--model", "`"$modelPath`"",
        "--host", "127.0.0.1",       # Only listen on localhost (wrapper will proxy)
        "--port", $LlamaCppPort,
        "--embedding",               # Enable embedding mode
        "--pooling", "mean",         # Mean pooling for sentence embeddings
        "--ctx-size", 8192,
        "--threads", 4,
        "--batch-size", 512,
        "--parallel", 4
    ) -join " "

    Write-Status "  Arguments: $arguments"

    & $NssmPath install $LlamaCppServiceName $serverExe $arguments
    & $NssmPath set $LlamaCppServiceName DisplayName "LlamaCpp Embedding (nomic-embed-text-v1.5)"
    & $NssmPath set $LlamaCppServiceName Description "llama.cpp embedding server with nomic-embed-text-v1.5"
    & $NssmPath set $LlamaCppServiceName Start SERVICE_AUTO_START
    & $NssmPath set $LlamaCppServiceName AppDirectory $InstallDir
    & $NssmPath set $LlamaCppServiceName AppStdout "$LogsDir\llamacpp-embedding.log"
    & $NssmPath set $LlamaCppServiceName AppStderr "$LogsDir\llamacpp-embedding.error.log"
    & $NssmPath set $LlamaCppServiceName AppRotateFiles 1
    & $NssmPath set $LlamaCppServiceName AppRotateBytes 10485760

    Write-Status "  Service $LlamaCppServiceName created"
}

function Create-WrapperService {
    Write-Status "Creating API wrapper service..."

    Remove-OldService -ServiceName $WrapperServiceName

    # Build command for uvicorn
    $arguments = "-m uvicorn embedding_wrapper:app --host 0.0.0.0 --port $ApiPort"

    & $NssmPath install $WrapperServiceName $PythonPath $arguments
    & $NssmPath set $WrapperServiceName DisplayName "Embedding API Wrapper"
    & $NssmPath set $WrapperServiceName Description "FastAPI wrapper for llama.cpp embedding service"
    & $NssmPath set $WrapperServiceName Start SERVICE_AUTO_START
    & $NssmPath set $WrapperServiceName AppDirectory $WrapperDir
    & $NssmPath set $WrapperServiceName AppStdout "$LogsDir\embedding-wrapper.log"
    & $NssmPath set $WrapperServiceName AppStderr "$LogsDir\embedding-wrapper.error.log"
    & $NssmPath set $WrapperServiceName AppRotateFiles 1
    & $NssmPath set $WrapperServiceName AppRotateBytes 10485760
    & $NssmPath set $WrapperServiceName AppEnvironmentExtra "LLAMACPP_URL=http://localhost:$LlamaCppPort"

    Write-Status "  Service $WrapperServiceName created"
}

function Start-Services {
    Write-Status "Starting services..."

    # Start llama.cpp first
    Write-Status "  Starting $LlamaCppServiceName..."
    Start-Service -Name $LlamaCppServiceName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 10  # Give model time to load

    # Then start wrapper
    Write-Status "  Starting $WrapperServiceName..."
    Start-Service -Name $WrapperServiceName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

function Test-Services {
    Write-Status "Testing services..."

    $maxRetries = 30

    # Test llama.cpp
    $healthy = $false
    for ($i = 0; $i -lt $maxRetries; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$LlamaCppPort/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            $healthy = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    if ($healthy) {
        Write-Status "  llama.cpp (port $LlamaCppPort): OK" -Color Green
    } else {
        Write-Status "  llama.cpp (port $LlamaCppPort): Not responding" -Color Red
        return $false
    }

    # Test wrapper
    $healthy = $false
    for ($i = 0; $i -lt 15; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$ApiPort/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            $healthy = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    if ($healthy) {
        Write-Status "  API Wrapper (port $ApiPort): OK" -Color Green
    } else {
        Write-Status "  API Wrapper (port $ApiPort): Not responding" -Color Red
        return $false
    }

    # Test actual embedding
    try {
        $testPayload = '{"text": "This is a test sentence."}'
        $response = Invoke-RestMethod -Uri "http://localhost:$ApiPort/embed" `
            -Method POST `
            -ContentType "application/json" `
            -Body $testPayload `
            -TimeoutSec 30

        if ($response.embedding -and $response.embedding.Count -eq 768) {
            Write-Status "  Embedding test: OK (768 dimensions)" -Color Green
        } else {
            Write-Status "  Embedding test: Unexpected dimensions ($($response.embedding.Count))" -Color Yellow
        }
    } catch {
        Write-Status "  Embedding test failed: $_" -Color Yellow
    }

    return $true
}

# Main execution
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Embedding Service Setup" -ForegroundColor Cyan
Write-Host "  Model: nomic-embed-text-v1.5" -ForegroundColor Cyan
Write-Host "  llama.cpp Port: $LlamaCppPort (internal)" -ForegroundColor Cyan
Write-Host "  API Port: $ApiPort (external)" -ForegroundColor Cyan
Write-Host "  Host: $env:COMPUTERNAME" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify prerequisites
if (-not (Test-Path $NssmPath)) {
    throw "NSSM not found at $NssmPath"
}

if (-not (Test-Path $PythonPath)) {
    # Try to find Python
    $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PythonPath) {
        throw "Python not found. Please install Python 3.11+ and update `$PythonPath in script."
    }
    Write-Status "Found Python at: $PythonPath"
}

# Execute setup steps
Stop-Services
Install-LlamaCpp
Download-Model
Install-Wrapper
Create-LlamaCppService
Create-WrapperService
Start-Services
$success = Test-Services

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($success) {
    Write-Host "  Setup Complete!" -ForegroundColor Cyan
} else {
    Write-Host "  Setup Complete (with warnings)" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Status "Services:"
Write-Host "  ${LlamaCppServiceName}: http://localhost:${LlamaCppPort} (internal)" -ForegroundColor White
Write-Host "  ${WrapperServiceName}: http://localhost:${ApiPort} (external)" -ForegroundColor White
Write-Host ""

Write-Status "Management Commands:"
Write-Host "  Status:   Get-Service *Embedding*" -ForegroundColor Gray
Write-Host "  Start:    Start-Service $LlamaCppServiceName, $WrapperServiceName" -ForegroundColor Gray
Write-Host "  Stop:     Stop-Service $WrapperServiceName, $LlamaCppServiceName" -ForegroundColor Gray
Write-Host "  Logs:     Get-Content $LogsDir\*.log -Tail 50" -ForegroundColor Gray
Write-Host ""

Write-Status "Configuration for your project (.env):" -Color Yellow
Write-Host "  EMBEDDING_SERVICE_URL=http://$($env:COMPUTERNAME):$ApiPort" -ForegroundColor Yellow
Write-Host "  EMBEDDING_DIMENSIONS=768" -ForegroundColor Yellow
Write-Host ""

Write-Status "Test Commands:"
Write-Host "  Health:   curl http://$($env:COMPUTERNAME):$ApiPort/health" -ForegroundColor Gray
Write-Host "  Embed:    curl -X POST http://$($env:COMPUTERNAME):$ApiPort/embed -H 'Content-Type: application/json' -d '{\"text\":\"test\"}'" -ForegroundColor Gray
Write-Host ""
