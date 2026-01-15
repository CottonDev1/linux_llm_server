# Start Multiple llama.cpp Servers for Multi-Model Support
# 
# This script starts separate llama.cpp instances for different model types,
# enabling concurrent multi-user access to specialized models.
#
# Architecture:
#   Port 8080 - SQL Model (nsql-llama-2-7b or sqlcoder2)
#   Port 8081 - General Model (Qwen2.5-7B-Instruct)
#   Port 8082 - Code Model (qwen2.5-coder) [Optional]
#
# Usage:
#   .\start-llamacpp-multimodel.ps1              # Start all model servers
#   .\start-llamacpp-multimodel.ps1 -SqlOnly     # Start only SQL model
#   .\start-llamacpp-multimodel.ps1 -Stop        # Stop all llama.cpp servers

param(
    [switch]$SqlOnly,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = "Continue"

# Add CUDA to PATH for GPU acceleration
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if (Test-Path $cudaPath) {
    $env:PATH = "$cudaPath;$env:PATH"
}

# Configuration
$scriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$modelsDir = Join-Path $scriptDir "models\llamacpp"
$logsDir = Join-Path $scriptDir "logs"
$llamaVenv = Join-Path $scriptDir "llamacpp_venv\Scripts\python.exe"

# Model configurations
# Server specs: 64GB RAM, NVIDIA T400 4GB VRAM
# Strategy: Use GPU for acceleration (90% VRAM = ~3.6GB usable)
$models = @{
    sql = @{
        port = 8080
        modelPriority = @(
            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",  # Primary: qwen2.5-coder (1.1GB) - fits in GPU
            "sqlcoder2.Q4_K_M.gguf",                    # Fallback: sqlcoder2 (10GB) - partial GPU
            "nsql-llama-2-7b.Q4_K_M.gguf"               # Fallback: nsql-llama
        )
        contextSize = 8192
        description = "SQL Model (qwen2.5-coder)"
        gpuLayers = 99                                  # All layers for 1.5B model
        nThreads = 4
    }
    general = @{
        port = 8081
        modelPriority = @(
            "qwen2.5-7b-instruct-q4_k_m.gguf",          # Primary: Qwen2.5-7B
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf"           # Fallback: alternate casing
        )
        contextSize = 8192
        description = "General Model (Qwen2.5-7B)"
        gpuLayers = 20                                  # Partial GPU (7B has 28 layers, ~3GB for 20)
        nThreads = 4
    }
    code = @{
        port = 8082
        modelPriority = @(
            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"   # Code model
        )
        contextSize = 8192
        description = "Code Model (qwen2.5-coder)"
        gpuLayers = 99                                  # All layers (1.1GB fits easily)
        nThreads = 4
    }
}

# Ensure logs directory exists
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

function Get-ModelPath {
    param([array]$ModelPriority)
    
    foreach ($model in $ModelPriority) {
        $path = Join-Path $modelsDir $model
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Stop-LlamaCppServers {
    Write-Host "`n[STOP] Stopping all llama.cpp servers..." -ForegroundColor Yellow
    
    foreach ($name in $models.Keys) {
        $config = $models[$name]
        $port = $config.port
        
        if (Test-PortInUse -Port $port) {
            Write-Host "   Stopping $name server on port $port..." -ForegroundColor Gray
            
            # Find and kill the process using this port
            $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
            foreach ($conn in $connections) {
                $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                if ($process) {
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                    Write-Host "   Stopped process $($process.Id) ($($process.ProcessName))" -ForegroundColor Gray
                }
            }
        } else {
            Write-Host "   $name server not running on port $port" -ForegroundColor Gray
        }
    }
    
    Write-Host "   All llama.cpp servers stopped" -ForegroundColor Green
}

function Get-ServerStatus {
    Write-Host "`n[STATUS] llama.cpp Server Status" -ForegroundColor Cyan
    Write-Host "=" * 60
    
    foreach ($name in $models.Keys) {
        $config = $models[$name]
        $port = $config.port
        
        $status = if (Test-PortInUse -Port $port) { "RUNNING" } else { "STOPPED" }
        $statusColor = if ($status -eq "RUNNING") { "Green" } else { "Red" }
        
        $modelPath = Get-ModelPath -ModelPriority $config.modelPriority
        $modelName = if ($modelPath) { Split-Path $modelPath -Leaf } else { "NOT FOUND" }
        
        Write-Host "`n  $($name.ToUpper()) Server" -ForegroundColor White
        Write-Host "    Status:      " -NoNewline; Write-Host $status -ForegroundColor $statusColor
        Write-Host "    Port:        $port" -ForegroundColor Gray
        Write-Host "    Model:       $modelName" -ForegroundColor Gray
        Write-Host "    Description: $($config.description)" -ForegroundColor Gray
        
        # Try to get model info if running
        if ($status -eq "RUNNING") {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:$port/v1/models" -TimeoutSec 3 -ErrorAction Stop
                $loadedModel = $response.data[0].id
                Write-Host "    Loaded:      $loadedModel" -ForegroundColor Green
            } catch {
                Write-Host "    Loaded:      (API not responding)" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host "`n" + "=" * 60
}

function Start-ModelServer {
    param(
        [string]$Name,
        [hashtable]$Config
    )
    
    $port = $Config.port
    $modelPath = Get-ModelPath -ModelPriority $Config.modelPriority
    
    if (-not $modelPath) {
        Write-Host "   [SKIP] No model found for $Name server" -ForegroundColor Yellow
        Write-Host "         Expected one of: $($Config.modelPriority -join ', ')" -ForegroundColor Gray
        return $false
    }
    
    if (Test-PortInUse -Port $port) {
        Write-Host "   [OK] $Name server already running on port $port" -ForegroundColor Green
        return $true
    }
    
    $modelName = Split-Path $modelPath -Leaf
    Write-Host "   Starting $Name server on port $port..." -ForegroundColor Gray
    Write-Host "      Model: $modelName" -ForegroundColor Gray
    Write-Host "      Context: $($Config.contextSize) tokens" -ForegroundColor Gray
    
    $logFile = Join-Path $logsDir "llamacpp-$Name.log"
    
    # Build the command arguments
    $serverArgs = @(
        "-m", "llama_cpp.server",
        "--model", $modelPath,
        "--host", "0.0.0.0",
        "--port", $port,
        "--n_ctx", $Config.contextSize,
        "--n_gpu_layers", $Config.gpuLayers,
        "--n_batch", 512
    )

    # Add thread count if specified
    if ($Config.nThreads) {
        $serverArgs += @("--n_threads", $Config.nThreads)
    }

    # Start the server process
    $process = Start-Process -FilePath $llamaVenv `
        -ArgumentList $serverArgs `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -PassThru
    
    # Wait for server to be ready
    $maxRetries = 60  # 60 seconds max wait
    $retryCount = 0
    $serverReady = $false
    
    Write-Host "      Waiting for model to load..." -ForegroundColor Gray
    
    while ($retryCount -lt $maxRetries -and -not $serverReady) {
        Start-Sleep -Seconds 1
        $retryCount++
        
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:$port/v1/models" -TimeoutSec 2 -ErrorAction Stop
            $serverReady = $true
        } catch {
            if ($retryCount % 10 -eq 0) {
                Write-Host "      Still loading... ($retryCount seconds)" -ForegroundColor Gray
            }
        }
    }
    
    if ($serverReady) {
        Write-Host "   [OK] $Name server ready on port $port" -ForegroundColor Green
        return $true
    } else {
        Write-Host "   [WARN] $Name server may still be loading (check $logFile)" -ForegroundColor Yellow
        return $false
    }
}

# Main execution
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "llama.cpp Multi-Model Server Manager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($Stop) {
    Stop-LlamaCppServers
    exit 0
}

if ($Status) {
    Get-ServerStatus
    exit 0
}

# Check for llamacpp venv
if (-not (Test-Path $llamaVenv)) {
    Write-Host "`n[ERROR] llama.cpp virtual environment not found at:" -ForegroundColor Red
    Write-Host "        $llamaVenv" -ForegroundColor Gray
    Write-Host "`n        Run install-all.ps1 to set up the environment" -ForegroundColor Yellow
    exit 1
}

# Start servers
Write-Host "`n[START] Starting llama.cpp servers..." -ForegroundColor Yellow

# Always start SQL server
Start-ModelServer -Name "sql" -Config $models.sql

if (-not $SqlOnly) {
    # Start general server
    Start-ModelServer -Name "general" -Config $models.general
    
    # Code server is optional - only start if model exists
    $codeModel = Get-ModelPath -ModelPriority $models.code.modelPriority
    if ($codeModel) {
        Start-ModelServer -Name "code" -Config $models.code
    }
}

# Show final status
Get-ServerStatus

# Environment variable hint
Write-Host "`nEnvironment Variables for your .env file:" -ForegroundColor Cyan
Write-Host "  LLAMACPP_SQL_HOST=http://localhost:8080" -ForegroundColor White
Write-Host "  LLAMACPP_GENERAL_HOST=http://localhost:8081" -ForegroundColor White
Write-Host "  LLAMACPP_CODE_HOST=http://localhost:8082" -ForegroundColor White
Write-Host ""
