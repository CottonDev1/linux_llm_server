# Deploy Embedding Service to EWRSPT-AI
# Just llama.cpp with nomic-embed-text-v1.5 - no Python wrapper needed

param(
    [switch]$Force,
    [int]$Port = 8083
)

$ErrorActionPreference = "Stop"

$LocalSetupDir = "C:\temp\embedding-setup"
$RemoteHost = "EWRSPT-AI"

# Remote paths
$InstallDir = "C:\tools\llama.cpp"
$ModelsDir = "C:\tools\llama.cpp\models"
$LogsDir = "C:\tools\llama.cpp\logs"
$NssmPath = "C:\tools\nssm\nssm.exe"

$ModelName = "nomic-embed-text-v1.5.f16.gguf"
$ServiceName = "LlamaCpp-Embedding"

function Write-Status {
    param([string]$Message, [string]$Color = "Green")
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $Message" -ForegroundColor $Color
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deploying Embedding Service" -ForegroundColor Cyan
Write-Host "  Target: $RemoteHost" -ForegroundColor Cyan
Write-Host "  Port: $Port" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check local files exist
if (-not (Test-Path "$LocalSetupDir\llama-cpp.zip")) {
    throw "llama-cpp.zip not found in $LocalSetupDir"
}
if (-not (Test-Path "$LocalSetupDir\$ModelName")) {
    throw "$ModelName not found in $LocalSetupDir"
}

# Create remote directories
Write-Status "Creating remote directories..."
Invoke-Command -ComputerName $RemoteHost -ScriptBlock {
    param($InstallDir, $ModelsDir, $LogsDir)
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
} -ArgumentList $InstallDir, $ModelsDir, $LogsDir

# Copy files to remote
Write-Status "Copying llama.cpp zip..."
Copy-Item "$LocalSetupDir\llama-cpp.zip" -Destination "\\$RemoteHost\C`$\tools\llama.cpp\llama-cpp.zip" -Force

Write-Status "Copying model (261 MB)..."
Copy-Item "$LocalSetupDir\$ModelName" -Destination "\\$RemoteHost\C`$\tools\llama.cpp\models\$ModelName" -Force

# Install on remote
Write-Status "Installing service..."
Invoke-Command -ComputerName $RemoteHost -ScriptBlock {
    param($InstallDir, $ModelsDir, $LogsDir, $NssmPath, $ModelName, $ServiceName, $Port, $Force)

    $ErrorActionPreference = "Stop"

    # Stop existing service
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -eq "Running") {
            Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3
        }
        & $NssmPath remove $ServiceName confirm 2>$null
        Start-Sleep -Seconds 2
    }

    # Extract llama.cpp
    $zipPath = "$InstallDir\llama-cpp.zip"
    $serverExe = "$InstallDir\llama-server.exe"

    if (-not (Test-Path $serverExe) -or $Force) {
        Write-Host "Extracting llama.cpp..."
        Expand-Archive -Path $zipPath -DestinationPath $InstallDir -Force

        # Move files from nested directory
        $nestedDir = Get-ChildItem -Path $InstallDir -Directory | Where-Object { $_.Name -like "llama-*" } | Select-Object -First 1
        if ($nestedDir -and (Test-Path "$($nestedDir.FullName)\llama-server.exe")) {
            Get-ChildItem -Path $nestedDir.FullName | Move-Item -Destination $InstallDir -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $nestedDir.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not (Test-Path $serverExe)) {
        throw "llama-server.exe not found"
    }

    # Create service
    Write-Host "Creating service..."
    $modelPath = "$ModelsDir\$ModelName"
    $arguments = "--model `"$modelPath`" --host 0.0.0.0 --port $Port --embedding --pooling mean --ctx-size 8192 --threads 4 --batch-size 512 --parallel 4"

    & $NssmPath install $ServiceName "$InstallDir\llama-server.exe" $arguments
    & $NssmPath set $ServiceName DisplayName "LlamaCpp Embedding (nomic-embed-text-v1.5)"
    & $NssmPath set $ServiceName Description "llama.cpp embedding server"
    & $NssmPath set $ServiceName Start SERVICE_AUTO_START
    & $NssmPath set $ServiceName AppDirectory $InstallDir
    & $NssmPath set $ServiceName AppStdout "$LogsDir\embedding.log"
    & $NssmPath set $ServiceName AppStderr "$LogsDir\embedding.error.log"
    & $NssmPath set $ServiceName AppRotateFiles 1
    & $NssmPath set $ServiceName AppRotateBytes 10485760

    # Start service
    Write-Host "Starting service..."
    Start-Service -Name $ServiceName
    Start-Sleep -Seconds 10

    # Test
    Write-Host "Testing..."
    $healthy = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $null = Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            $healthy = $true
            break
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    if ($healthy) {
        Write-Host "Service is running!" -ForegroundColor Green
    } else {
        Write-Host "Service not responding - check logs" -ForegroundColor Red
    }

} -ArgumentList $InstallDir, $ModelsDir, $LogsDir, $NssmPath, $ModelName, $ServiceName, $Port, $Force

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deployment Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "llama.cpp embedding server: http://${RemoteHost}:$Port" -ForegroundColor White
Write-Host ""
Write-Host "API endpoints:" -ForegroundColor Gray
Write-Host "  GET  /health" -ForegroundColor Gray
Write-Host "  POST /embedding  {`"content`": `"text`"}" -ForegroundColor Gray
Write-Host ""
