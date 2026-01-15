#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Register RAG Server components as Windows Services using NSSM

.DESCRIPTION
    This script registers all RAG Server components as Windows Services:
    - RAGServer (Node.js web server)
    - RAGPython (FastAPI backend)
    - LlamaCppSQL (SQL model llama.cpp server)
    - LlamaCppGeneral (General model llama.cpp server)

    Services will auto-start on boot and restart on failure.

.PARAMETER Uninstall
    Remove all registered services

.PARAMETER InstallPath
    Installation directory (default: C:\Projects\LLM_Website)

.EXAMPLE
    .\register-windows-services.ps1

.EXAMPLE
    .\register-windows-services.ps1 -Uninstall

.NOTES
    Requires NSSM (Non-Sucking Service Manager)
    Install with: choco install nssm -y
#>

param(
    [switch]$Uninstall,
    [string]$InstallPath = "C:\Projects\LLM_Website"
)

$ErrorActionPreference = "Stop"

# Service definitions
# Note: MongoDB runs in Docker on port 27018, not as a Windows service
$services = @{
    RAGServer = @{
        DisplayName = "RAG Web Server"
        Description = "Node.js web server for RAG application"
        Executable = "C:\Program Files\nodejs\node.exe"
        Arguments = "$InstallPath\rag-server.js"
        WorkingDir = $InstallPath
        Dependencies = @()  # MongoDB is in Docker, not a Windows service
    }
    RAGPython = @{
        DisplayName = "RAG Python API"
        Description = "FastAPI backend for RAG application"
        Executable = "$InstallPath\python_services\venv\Scripts\python.exe"
        Arguments = "-m uvicorn main:app --host 0.0.0.0 --port 8001"
        WorkingDir = "$InstallPath\python_services"
        Dependencies = @()  # MongoDB is in Docker, not a Windows service
    }
    LlamaCppSQL = @{
        DisplayName = "LlamaCpp SQL Model"
        Description = "llama.cpp server for SQL generation (port 8080)"
        Executable = "$InstallPath\llamacpp_venv\Scripts\python.exe"
        Arguments = "-m llama_cpp.server --model $InstallPath\models\llamacpp\sqlcoder2.Q4_K_M.gguf --host 0.0.0.0 --port 8080 --n_gpu_layers 0 --n_threads 8 --n_batch 512"
        WorkingDir = $InstallPath
        Dependencies = @()
        FallbackModel = "$InstallPath\models\llamacpp\nsql-llama-2-7b.Q4_K_M.gguf"
    }
    LlamaCppGeneral = @{
        DisplayName = "LlamaCpp General Model"
        Description = "llama.cpp server for general chat (port 8081)"
        Executable = "$InstallPath\llamacpp_venv\Scripts\python.exe"
        Arguments = "-m llama_cpp.server --model $InstallPath\models\llamacpp\Qwen2.5-7B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8081 --n_gpu_layers 0 --n_threads 8 --n_batch 512"
        WorkingDir = $InstallPath
        Dependencies = @()
        FallbackModel = "$InstallPath\models\llamacpp\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    }
}

function Write-Header { param($Text) Write-Host "`n$("=" * 70)" -ForegroundColor Cyan; Write-Host $Text -ForegroundColor Cyan; Write-Host "$("=" * 70)" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Warn { param($Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Err { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Check for NSSM
$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) {
    Write-Err "NSSM not found. Please install it first:"
    Write-Host "    choco install nssm -y" -ForegroundColor Yellow
    exit 1
}

Write-Header "RAG Server Windows Service Manager"

# Pre-flight check: Verify MongoDB is running in Docker on port 27018
Write-Host "Checking MongoDB (Docker on port 27018)..." -ForegroundColor Gray
$mongoRunning = Get-NetTCPConnection -LocalPort 27018 -State Listen -ErrorAction SilentlyContinue
if ($mongoRunning) {
    Write-Success "MongoDB is running on port 27018"
} else {
    Write-Warn "MongoDB not detected on port 27018"
    Write-Host "    Ensure Docker container is running: docker ps | findstr mongo" -ForegroundColor Yellow
    Write-Host "    Services will start but may fail to connect to database" -ForegroundColor Yellow
    Write-Host ""
}

if ($Uninstall) {
    Write-Host "`nUninstalling services..." -ForegroundColor Yellow
    
    foreach ($name in $services.Keys) {
        $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
        if ($svc) {
            Write-Host "Stopping and removing $name..."
            if ($svc.Status -eq "Running") {
                Stop-Service -Name $name -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 2
            }
            & nssm remove $name confirm
            Write-Success "$name removed"
        } else {
            Write-Host "$name not installed, skipping" -ForegroundColor Gray
        }
    }
    
    Write-Host "`nAll services uninstalled" -ForegroundColor Green
    exit 0
}

# Install services
Write-Host "`nInstalling services..." -ForegroundColor Yellow

foreach ($name in $services.Keys) {
    $config = $services[$name]
    
    Write-Host "`nConfiguring $name..." -ForegroundColor White
    
    # Check if service already exists
    $existing = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Warn "$name already exists. Stopping and reconfiguring..."
        if ($existing.Status -eq "Running") {
            Stop-Service -Name $name -Force
            Start-Sleep -Seconds 2
        }
        & nssm remove $name confirm 2>$null
    }
    
    # Check executable exists
    if (-not (Test-Path $config.Executable)) {
        Write-Err "Executable not found: $($config.Executable)"
        continue
    }
    
    # For llama.cpp services, check model exists (use fallback if needed)
    if ($name -like "LlamaCpp*") {
        $modelPath = ($config.Arguments -split "--model ")[1] -split " " | Select-Object -First 1
        if (-not (Test-Path $modelPath)) {
            if ($config.FallbackModel -and (Test-Path $config.FallbackModel)) {
                Write-Warn "Primary model not found, using fallback: $($config.FallbackModel)"
                $config.Arguments = $config.Arguments -replace [regex]::Escape($modelPath), $config.FallbackModel
            } else {
                Write-Err "Model not found: $modelPath"
                continue
            }
        }
    }
    
    # Install service
    & nssm install $name $config.Executable
    
    # Configure service
    & nssm set $name AppParameters $config.Arguments
    & nssm set $name AppDirectory $config.WorkingDir
    & nssm set $name DisplayName $config.DisplayName
    & nssm set $name Description $config.Description
    & nssm set $name Start SERVICE_AUTO_START
    
    # Configure restart on failure
    & nssm set $name AppExit Default Restart
    & nssm set $name AppRestartDelay 5000
    
    # Configure logging
    $logDir = "$InstallPath\logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    & nssm set $name AppStdout "$logDir\$name.log"
    & nssm set $name AppStderr "$logDir\$name.error.log"
    & nssm set $name AppRotateFiles 1
    & nssm set $name AppRotateBytes 10485760  # 10MB
    
    # Set dependencies
    if ($config.Dependencies.Count -gt 0) {
        $deps = $config.Dependencies -join "/"
        & nssm set $name DependOnService $deps
    }
    
    Write-Success "$name installed"
}

# Start services
Write-Host "`nStarting services..." -ForegroundColor Yellow

# Start llama.cpp services first (they take time to load models)
$llamaServices = $services.Keys | Where-Object { $_ -like "LlamaCpp*" }
foreach ($name in $llamaServices) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Host "Starting $name (this may take 30-60 seconds to load model)..."
        Start-Service -Name $name
    }
}

# Wait for llama.cpp to load
Write-Host "Waiting for models to load..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Start other services
$otherServices = $services.Keys | Where-Object { $_ -notlike "LlamaCpp*" }
foreach ($name in $otherServices) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Host "Starting $name..."
        Start-Service -Name $name
    }
}

Start-Sleep -Seconds 5

# Show status
Write-Header "Service Status"

foreach ($name in $services.Keys) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($svc) {
        $status = $svc.Status
        $color = if ($status -eq "Running") { "Green" } else { "Red" }
        Write-Host "  $name : " -NoNewline
        Write-Host $status -ForegroundColor $color
    }
}

Write-Host ""
Write-Host "Services registered and started!" -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  Get-Service RAG*, LlamaCpp*           # Check status"
Write-Host "  Restart-Service RAGServer             # Restart a service"
Write-Host "  Get-Content $InstallPath\logs\*.log   # View logs"
Write-Host "  .\register-windows-services.ps1 -Uninstall  # Remove all services"
