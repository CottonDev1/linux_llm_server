# LlamaCpp CUDA Setup Script
# Downloads pre-built llama.cpp with CUDA support and configures Windows services
#
# Requirements:
#   - NVIDIA GPU with CUDA support
#   - CUDA Toolkit v13.1 installed
#   - NSSM for Windows service management
#
# GPU: NVIDIA T400 4GB - Strategy: Use GPU layers intelligently to fit in VRAM

$ErrorActionPreference = "Continue"

# Configuration
$ProjectDir = "C:\Projects\llm_website"
$LlamaCppDir = "$ProjectDir\llamacpp-cuda"
$ModelsDir = "$ProjectDir\models\llamacpp"
$LogsDir = "$ProjectDir\logs"
$NssmDir = "C:\Tools\nssm"

# Latest release with CUDA 12.4 support (compatible with most NVIDIA drivers)
$ReleaseTag = "b7640"
$BaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$ReleaseTag"
$LlamaBinary = "llama-$ReleaseTag-bin-win-cuda-12.4-x64.zip"
$CudaRuntime = "cudart-llama-bin-win-cuda-12.4-x64.zip"

# Model configurations optimized for T400 4GB GPU
# Total VRAM: 4096 MiB, usable ~3800 MiB
$Models = @(
    @{
        Name = "LlamaCpp-SQL"
        Port = 8080
        Model = "sqlcoder2.Q4_K_M.gguf"  # ~9.3GB 7B model - partial GPU
        GpuLayers = 20   # ~3.5GB in GPU, rest in RAM
        ContextSize = 8192  # Increased for large schema prompts
        Threads = 6
        Description = "SQL Generation (sqlcoder2-7B)"
    },
    @{
        Name = "LlamaCpp-General"
        Port = 8081
        Model = "qwen2.5-7b-instruct-q4_k_m.gguf"          # ~4.7GB - partial GPU
        GpuLayers = 15   # ~2GB in GPU, rest in RAM
        ContextSize = 4096
        Threads = 6
        Description = "General Purpose (Qwen2.5-7B)"
    }
    # LlamaCpp-Code disabled - resources allocated to SQL model
)

function Write-Status {
    param([string]$Message, [string]$Color = "Green")
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $Message" -ForegroundColor $Color
}

function Stop-ExistingServices {
    Write-Status "Stopping existing LLM services..."

    foreach ($model in $Models) {
        $serviceName = $model.Name
        $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue

        if ($service -and $service.Status -eq "Running") {
            Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
            Write-Status "  Stopped $serviceName" -Color Yellow
        }
    }

    # Kill any orphaned processes
    foreach ($model in $Models) {
        $port = $model.Port
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }

    Start-Sleep -Seconds 3
}

function Install-NSSM {
    if (Test-Path "$NssmDir\nssm.exe") {
        Write-Status "NSSM already installed"
        return
    }

    Write-Status "Installing NSSM..."
    $NssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $zipPath = "$env:TEMP\nssm.zip"

    Invoke-WebRequest -Uri $NssmUrl -OutFile $zipPath
    New-Item -ItemType Directory -Force -Path $NssmDir | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath "$env:TEMP\nssm" -Force
    Copy-Item "$env:TEMP\nssm\nssm-2.24\win64\nssm.exe" -Destination $NssmDir

    Remove-Item $zipPath -Force
    Remove-Item "$env:TEMP\nssm" -Recurse -Force

    Write-Status "NSSM installed to $NssmDir"
}

function Download-LlamaCppCuda {
    Write-Status "Checking for llama.cpp CUDA binaries..."

    if (-not (Test-Path $LlamaCppDir)) {
        New-Item -ItemType Directory -Force -Path $LlamaCppDir | Out-Null
    }

    $serverExe = "$LlamaCppDir\llama-server.exe"

    if (Test-Path $serverExe) {
        Write-Status "llama-server.exe already exists"
        return
    }

    Write-Status "Downloading llama.cpp CUDA binaries ($ReleaseTag)..."

    # Download main binaries
    $llamaZip = "$env:TEMP\$LlamaBinary"
    Write-Status "  Downloading $LlamaBinary..."
    Invoke-WebRequest -Uri "$BaseUrl/$LlamaBinary" -OutFile $llamaZip

    # Download CUDA runtime
    $cudaZip = "$env:TEMP\$CudaRuntime"
    Write-Status "  Downloading $CudaRuntime..."
    Invoke-WebRequest -Uri "$BaseUrl/$CudaRuntime" -OutFile $cudaZip

    # Extract
    Write-Status "  Extracting binaries..."
    Expand-Archive -Path $llamaZip -DestinationPath $LlamaCppDir -Force
    Expand-Archive -Path $cudaZip -DestinationPath $LlamaCppDir -Force

    # Cleanup
    Remove-Item $llamaZip -Force
    Remove-Item $cudaZip -Force

    # Verify
    if (Test-Path $serverExe) {
        Write-Status "llama.cpp CUDA binaries installed successfully"
    } else {
        Write-Status "ERROR: llama-server.exe not found after extraction!" -Color Red
        exit 1
    }
}

function Remove-OldService {
    param([string]$ServiceName)

    $nssm = "$NssmDir\nssm.exe"
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue

    if ($service) {
        Write-Status "  Removing old service: $ServiceName" -Color Yellow
        & $nssm stop $ServiceName 2>$null
        Start-Sleep -Seconds 2
        & $nssm remove $ServiceName confirm 2>$null
        Start-Sleep -Seconds 2
    }
}

function Create-LlamaCppService {
    param(
        [string]$ServiceName,
        [int]$Port,
        [string]$ModelFile,
        [int]$GpuLayers,
        [int]$ContextSize,
        [int]$Threads,
        [string]$Description
    )

    $nssm = "$NssmDir\nssm.exe"
    $serverExe = "$LlamaCppDir\llama-server.exe"
    $modelPath = "$ModelsDir\$ModelFile"

    # Check model exists
    if (-not (Test-Path $modelPath)) {
        Write-Status "  WARNING: Model not found: $ModelFile" -Color Yellow
        return $false
    }

    $modelSize = [math]::Round((Get-Item $modelPath).Length / 1GB, 2)
    Write-Status "Creating service: $ServiceName"
    Write-Status "  Model: $ModelFile ($modelSize GB)"
    Write-Status "  GPU Layers: $GpuLayers, Context: $ContextSize, Threads: $Threads"

    # Remove old service if exists
    Remove-OldService -ServiceName $ServiceName

    # Build arguments for llama-server.exe
    # Using native executable - no Python overhead!
    $arguments = @(
        "--model", "`"$modelPath`"",
        "--host", "0.0.0.0",
        "--port", $Port,
        "--ctx-size", $ContextSize,
        "--n-gpu-layers", $GpuLayers,
        "--threads", $Threads,
        "--batch-size", 512,
        "--parallel", 2,           # Allow 2 concurrent requests
        "--cont-batching"          # Enable continuous batching
    ) -join " "

    # Create service with NSSM
    & $nssm install $ServiceName $serverExe $arguments
    & $nssm set $ServiceName Description $Description
    & $nssm set $ServiceName DisplayName "LlamaCpp CUDA - $Description"
    & $nssm set $ServiceName Start SERVICE_AUTO_START
    & $nssm set $ServiceName AppDirectory $LlamaCppDir
    & $nssm set $ServiceName AppStdout "$LogsDir\$ServiceName.log"
    & $nssm set $ServiceName AppStderr "$LogsDir\$ServiceName.error.log"
    & $nssm set $ServiceName AppRotateFiles 1
    & $nssm set $ServiceName AppRotateBytes 10485760

    # Set environment for CUDA (binaries include required DLLs)
    & $nssm set $ServiceName AppEnvironmentExtra "PATH=$LlamaCppDir;%PATH%"

    # CRITICAL: Run as current user to enable GPU access
    # LocalSystem (default) cannot access NVIDIA GPU!
    $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    Write-Status "  Setting service to run as: $currentUser" -Color Yellow
    Write-Status "  NOTE: You will need to update the service password manually:" -Color Yellow
    Write-Status "    Run: C:\Tools\nssm\nssm.exe edit $ServiceName" -Color Gray
    Write-Status "    Go to 'Log on' tab and enter your Windows password" -Color Gray
    & $nssm set $ServiceName ObjectName $currentUser

    Write-Status "  Service $ServiceName created" -Color Green
    return $true
}

function Start-Services {
    Write-Status "Starting LLM services..."

    foreach ($model in $Models) {
        $serviceName = $model.Name

        Write-Status "  Starting $serviceName..."
        Start-Service -Name $serviceName -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }

    # Wait for models to load
    Write-Status "Waiting for models to load (this may take 30-60 seconds)..."
    Start-Sleep -Seconds 15
}

function Test-Services {
    Write-Status "Testing service endpoints..."

    $allHealthy = $true

    foreach ($model in $Models) {
        $port = $model.Port
        $name = $model.Name

        $maxRetries = 30
        $healthy = $false

        for ($i = 0; $i -lt $maxRetries; $i++) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
                $healthy = $true
                break
            } catch {
                Start-Sleep -Seconds 2
            }
        }

        if ($healthy) {
            Write-Status "  $name (port $port): OK" -Color Green
        } else {
            Write-Status "  $name (port $port): Not responding (check logs)" -Color Yellow
            $allHealthy = $false
        }
    }

    return $allHealthy
}

function Show-GpuStatus {
    Write-Host "`n" -NoNewline
    Write-Status "GPU Status:"
    $gpu = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    Write-Host "  $gpu" -ForegroundColor Cyan
}

# Main execution
Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LlamaCpp CUDA Setup" -ForegroundColor Cyan
Write-Host "  GPU: NVIDIA T400 4GB" -ForegroundColor Cyan
Write-Host "  CUDA: v13.1" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LlamaCppDir | Out-Null

# Check GPU
Write-Status "Checking NVIDIA GPU..."
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Status "  Found: $gpuInfo"
} catch {
    Write-Status "ERROR: nvidia-smi not found. Is CUDA installed?" -Color Red
    exit 1
}

# Stop existing services
Stop-ExistingServices

# Install components
Install-NSSM
Download-LlamaCppCuda

# Check models
Write-Host ""
Write-Status "Checking model files..."
$modelsFound = 0
foreach ($model in $Models) {
    $modelPath = "$ModelsDir\$($model.Model)"
    if (Test-Path $modelPath) {
        $size = [math]::Round((Get-Item $modelPath).Length / 1GB, 2)
        Write-Status "  Found: $($model.Model) ($size GB)" -Color Green
        $modelsFound++
    } else {
        Write-Status "  Missing: $($model.Model)" -Color Red
    }
}

if ($modelsFound -eq 0) {
    Write-Status "ERROR: No model files found in $ModelsDir" -Color Red
    exit 1
}

# Create services
Write-Host ""
Write-Status "Creating Windows services with CUDA support..."
foreach ($model in $Models) {
    Create-LlamaCppService `
        -ServiceName $model.Name `
        -Port $model.Port `
        -ModelFile $model.Model `
        -GpuLayers $model.GpuLayers `
        -ContextSize $model.ContextSize `
        -Threads $model.Threads `
        -Description $model.Description
}

# Start services
Write-Host ""
Start-Services

# Test services
Write-Host ""
$healthy = Test-Services

# Show GPU utilization
Show-GpuStatus

# Summary
Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Status "Services created with CUDA GPU acceleration:"
foreach ($model in $Models) {
    Write-Host "  - $($model.Name): http://localhost:$($model.Port)" -ForegroundColor White
    Write-Host "    GPU Layers: $($model.GpuLayers), Context: $($model.ContextSize)" -ForegroundColor Gray
}

Write-Host ""
Write-Status "Management commands:"
Write-Host "  Status:  Get-Service LlamaCpp-*" -ForegroundColor White
Write-Host "  Stop:    .\scripts\stop-llm-services.ps1" -ForegroundColor White
Write-Host "  Logs:    $LogsDir" -ForegroundColor White
Write-Host ""
