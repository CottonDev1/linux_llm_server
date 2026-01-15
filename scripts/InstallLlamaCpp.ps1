# LlamaCpp Installation Script for Windows
# This script downloads llama.cpp, installs models, and configures Windows services
#
# Usage: Run as Administrator
#   .\install-llamacpp.ps1
#
# Prerequisites:
#   - Windows 10/11 or Windows Server 2019+
#   - PowerShell 5.1+
#   - Administrator privileges
#   - Internet connection

#Requires -RunAsAdministrator

param(
    [string]$InstallDir = "C:\llama.cpp",
    [string]$ModelsDir = "C:\Projects\llm_website\models\llamacpp",
    [switch]$SkipModelDownload,
    [switch]$SkipServiceInstall,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$LlamaCppVersion = "b4547"  # Latest stable release
$LlamaCppRelease = "https://github.com/ggerganov/llama.cpp/releases/download/$LlamaCppVersion"
$LlamaCppBinary = "llama-$LlamaCppVersion-bin-win-cuda-cu12.2.0-x64.zip"
$LlamaCppUrl = "$LlamaCppRelease/$LlamaCppBinary"

# Alternative: CPU-only version (smaller, no CUDA required)
$LlamaCppCpuBinary = "llama-$LlamaCppVersion-bin-win-avx2-x64.zip"
$LlamaCppCpuUrl = "$LlamaCppRelease/$LlamaCppCpuBinary"

# NSSM (Non-Sucking Service Manager) for Windows services
$NssmUrl = "https://nssm.cc/release/nssm-2.24.zip"

# Model configurations
$Models = @(
    @{
        Name = "SQL Model"
        File = "nsql-llama-2-7b.Q4_K_M.gguf"
        Url = "https://huggingface.co/TheBloke/nsql-llama-2-7B-GGUF/resolve/main/nsql-llama-2-7b.Q4_K_M.gguf"
        ServiceName = "LlamaCppSQL"
        Port = 8080
        ContextSize = 4096
    },
    @{
        Name = "General Model"
        File = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        Url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        ServiceName = "LlamaCppGeneral"
        Port = 8081
        ContextSize = 8192
    },
    @{
        Name = "Code Model"
        File = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        Url = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
        ServiceName = "LlamaCppCode"
        Port = 8082
        ContextSize = 8192
    }
)

# Helper functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " $Message" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "[*] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Test-CudaAvailable {
    try {
        $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $null = & nvidia-smi 2>&1
            return $LASTEXITCODE -eq 0
        }
    } catch {}
    return $false
}

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile
    )

    Write-Step "Downloading: $Url"
    Write-Step "To: $OutFile"

    $ProgressPreference = 'SilentlyContinue'  # Faster downloads
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
        Write-Step "Download complete: $OutFile"
    } catch {
        Write-Error "Failed to download: $_"
        throw
    }
}

# Main installation
Write-Header "LlamaCpp Installation Script"

# Check prerequisites
Write-Step "Checking prerequisites..."

if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
    Write-Step "Created models directory: $ModelsDir"
}

# Detect CUDA
$UseCuda = Test-CudaAvailable
if ($UseCuda) {
    Write-Step "CUDA detected - using GPU-accelerated version"
    $SelectedBinary = $LlamaCppBinary
    $SelectedUrl = $LlamaCppUrl
} else {
    Write-Warning "CUDA not detected - using CPU-only version"
    $SelectedBinary = $LlamaCppCpuBinary
    $SelectedUrl = $LlamaCppCpuUrl
}

# Step 1: Download and install llama.cpp
Write-Header "Step 1: Installing llama.cpp"

if ((Test-Path "$InstallDir\llama-server.exe") -and -not $Force) {
    Write-Step "llama.cpp already installed at $InstallDir"
    Write-Step "Use -Force to reinstall"
} else {
    # Create install directory
    if (Test-Path $InstallDir) {
        if ($Force) {
            Remove-Item -Path $InstallDir -Recurse -Force
        }
    }
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

    # Download llama.cpp
    $zipPath = "$env:TEMP\$SelectedBinary"
    Download-File -Url $SelectedUrl -OutFile $zipPath

    # Extract
    Write-Step "Extracting to $InstallDir..."
    Expand-Archive -Path $zipPath -DestinationPath $InstallDir -Force

    # Move files from nested directory if needed
    $nestedDir = Get-ChildItem -Path $InstallDir -Directory | Select-Object -First 1
    if ($nestedDir) {
        Get-ChildItem -Path $nestedDir.FullName | Move-Item -Destination $InstallDir -Force
        Remove-Item -Path $nestedDir.FullName -Force
    }

    # Verify installation
    if (Test-Path "$InstallDir\llama-server.exe") {
        Write-Step "llama.cpp installed successfully!"
    } else {
        # Try alternative path
        $serverExe = Get-ChildItem -Path $InstallDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if ($serverExe) {
            Write-Step "Found llama-server.exe at: $($serverExe.FullName)"
        } else {
            Write-Error "llama-server.exe not found after extraction"
            Write-Step "Contents of ${InstallDir}:"
            Get-ChildItem $InstallDir | ForEach-Object { Write-Host "  $_" }
            exit 1
        }
    }

    # Cleanup
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
}

# Find llama-server.exe
$llamaServer = Get-ChildItem -Path $InstallDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $llamaServer) {
    Write-Error "llama-server.exe not found in $InstallDir"
    exit 1
}
$LlamaServerPath = $llamaServer.FullName
Write-Step "Using llama-server at: $LlamaServerPath"

# Step 2: Download models
Write-Header "Step 2: Downloading Models"

if ($SkipModelDownload) {
    Write-Step "Skipping model download (--SkipModelDownload specified)"
} else {
    foreach ($model in $Models) {
        $modelPath = Join-Path $ModelsDir $model.File

        if ((Test-Path $modelPath) -and -not $Force) {
            $size = (Get-Item $modelPath).Length / 1GB
            Write-Step "$($model.Name) already exists ({0:N2} GB): $($model.File)" -f $size
        } else {
            Write-Step "Downloading $($model.Name)..."
            Download-File -Url $model.Url -OutFile $modelPath

            $size = (Get-Item $modelPath).Length / 1GB
            Write-Step "Downloaded: $($model.File) ({0:N2} GB)" -f $size
        }
    }
}

# Step 3: Install NSSM for Windows services
Write-Header "Step 3: Installing NSSM (Service Manager)"

$nssmPath = "$InstallDir\nssm.exe"
if ((Test-Path $nssmPath) -and -not $Force) {
    Write-Step "NSSM already installed"
} else {
    $nssmZip = "$env:TEMP\nssm.zip"
    Download-File -Url $NssmUrl -OutFile $nssmZip

    $nssmExtract = "$env:TEMP\nssm"
    Expand-Archive -Path $nssmZip -DestinationPath $nssmExtract -Force

    # Find nssm.exe (64-bit)
    $nssmExe = Get-ChildItem -Path $nssmExtract -Recurse -Filter "nssm.exe" |
               Where-Object { $_.FullName -like "*win64*" } |
               Select-Object -First 1

    if ($nssmExe) {
        Copy-Item -Path $nssmExe.FullName -Destination $nssmPath -Force
        Write-Step "NSSM installed to: $nssmPath"
    } else {
        Write-Error "Could not find nssm.exe in download"
        exit 1
    }

    # Cleanup
    Remove-Item $nssmZip -Force -ErrorAction SilentlyContinue
    Remove-Item $nssmExtract -Recurse -Force -ErrorAction SilentlyContinue
}

# Step 4: Create Windows services
Write-Header "Step 4: Configuring Windows Services"

if ($SkipServiceInstall) {
    Write-Step "Skipping service installation (--SkipServiceInstall specified)"
} else {
    foreach ($model in $Models) {
        $serviceName = $model.ServiceName
        $modelPath = Join-Path $ModelsDir $model.File
        $port = $model.Port
        $contextSize = $model.ContextSize

        Write-Step "Configuring service: $serviceName (port $port)"

        # Check if service exists
        $existingService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        if ($existingService) {
            if ($Force) {
                Write-Step "Removing existing service: $serviceName"
                & $nssmPath stop $serviceName 2>&1 | Out-Null
                & $nssmPath remove $serviceName confirm 2>&1 | Out-Null
                Start-Sleep -Seconds 2
            } else {
                Write-Step "Service $serviceName already exists (use -Force to reinstall)"
                continue
            }
        }

        # Verify model exists
        if (-not (Test-Path $modelPath)) {
            Write-Warning "Model not found: $modelPath - skipping service creation"
            continue
        }

        # Install service with NSSM
        $serverArgs = "-m `"$modelPath`" --host 0.0.0.0 --port $port -c $contextSize --threads 4"

        Write-Step "Installing service with args: $serverArgs"

        & $nssmPath install $serviceName $LlamaServerPath $serverArgs
        & $nssmPath set $serviceName DisplayName "LlamaCpp - $($model.Name)"
        & $nssmPath set $serviceName Description "llama.cpp server for $($model.Name) on port $port"
        & $nssmPath set $serviceName Start SERVICE_AUTO_START
        & $nssmPath set $serviceName AppStdout "$InstallDir\logs\$serviceName.log"
        & $nssmPath set $serviceName AppStderr "$InstallDir\logs\$serviceName-error.log"
        & $nssmPath set $serviceName AppRotateFiles 1
        & $nssmPath set $serviceName AppRotateBytes 10485760  # 10MB

        Write-Step "Service $serviceName installed"
    }

    # Create logs directory
    New-Item -ItemType Directory -Path "$InstallDir\logs" -Force | Out-Null
}

# Step 5: Start services
Write-Header "Step 5: Starting Services"

if ($SkipServiceInstall) {
    Write-Step "Skipping service start (--SkipServiceInstall specified)"
} else {
    foreach ($model in $Models) {
        $serviceName = $model.ServiceName
        $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue

        if ($service) {
            Write-Step "Starting $serviceName..."
            Start-Service -Name $serviceName -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3

            $service = Get-Service -Name $serviceName
            if ($service.Status -eq "Running") {
                Write-Step "$serviceName is running on port $($model.Port)"
            } else {
                Write-Warning "$serviceName failed to start - check logs at $InstallDir\logs\"
            }
        }
    }
}

# Step 6: Verify installation
Write-Header "Step 6: Verification"

Write-Step "Testing endpoints..."
Start-Sleep -Seconds 5  # Give services time to fully start

foreach ($model in $Models) {
    $port = $model.Port
    $serviceName = $model.ServiceName

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$port/health" -TimeoutSec 10 -ErrorAction Stop
        Write-Step "$serviceName (port $port): OK - Status: $($response.status)"
    } catch {
        Write-Warning "$serviceName (port $port): Not responding yet (may need more startup time)"
    }
}

# Summary
Write-Header "Installation Complete!"

Write-Host @"

LlamaCpp Installation Summary:
==============================
Install Directory: $InstallDir
Models Directory:  $ModelsDir
CUDA Enabled:      $UseCuda

Services Installed:
"@ -ForegroundColor White

foreach ($model in $Models) {
    $service = Get-Service -Name $model.ServiceName -ErrorAction SilentlyContinue
    $status = if ($service) { $service.Status } else { "Not Installed" }
    Write-Host "  - $($model.ServiceName): Port $($model.Port) - $status" -ForegroundColor White
}

Write-Host @"

Management Commands:
  Start:   Start-Service LlamaCppSQL, LlamaCppGeneral, LlamaCppCode
  Stop:    Stop-Service LlamaCppSQL, LlamaCppGeneral, LlamaCppCode
  Status:  Get-Service LlamaCpp*
  Logs:    Get-Content $InstallDir\logs\*.log -Tail 50

Test Endpoints:
  curl http://localhost:8080/health  # SQL Model
  curl http://localhost:8081/health  # General Model
  curl http://localhost:8082/health  # Code Model

"@ -ForegroundColor Gray
