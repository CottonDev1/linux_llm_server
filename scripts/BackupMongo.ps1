# MongoDB Backup Script
# Backs up the rag_server database from EWRSPT-AI to local C:\temp folder
#
# Usage:
#   .\backup-mongodb.ps1                    # Backup to C:\temp\mongodb_backup_<timestamp>
#   .\backup-mongodb.ps1 -OutputPath "D:\backups"  # Custom output path
#   .\backup-mongodb.ps1 -KeepRemote        # Keep backup on remote server after copy

param(
    [string]$OutputPath = "C:\temp",
    [string]$RemoteHost = "EWRSPT-AI",
    [int]$MongoPort = 27018,
    [string]$Database = "rag_server",
    [switch]$KeepRemote
)

$ErrorActionPreference = "Stop"

# Configuration
$mongodumpPath = "C:\Program Files\MongoDB\Tools\100\bin\mongodump.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupName = "mongodb_backup_$timestamp"
$remoteBackupPath = "C:\temp\$backupName"
$localBackupPath = Join-Path $OutputPath $backupName

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MongoDB Backup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Remote Host:    $RemoteHost"
Write-Host "Database:       $Database"
Write-Host "MongoDB Port:   $MongoPort"
Write-Host "Output Path:    $localBackupPath"
Write-Host "========================================`n" -ForegroundColor Cyan

# Ensure local output directory exists
if (-not (Test-Path $OutputPath)) {
    Write-Host "Creating output directory: $OutputPath" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
}

# Step 1: Create backup on remote server
Write-Host "[1/3] Creating backup on $RemoteHost..." -ForegroundColor Green

$sshCommand = @"
if not exist "C:\temp" mkdir "C:\temp"
"$mongodumpPath" --db $Database --port $MongoPort --out "$remoteBackupPath"
"@

try {
    $result = wsl ssh chad.walker@$RemoteHost $sshCommand 2>&1
    Write-Host $result
    Write-Host "      Backup created on remote server" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Failed to create backup on remote server" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Step 2: Copy backup to local machine
Write-Host "`n[2/3] Copying backup to local machine..." -ForegroundColor Green

try {
    # Use SCP to copy the backup
    $scpSource = "chad.walker@${RemoteHost}:$remoteBackupPath"
    $scpDest = $OutputPath

    Write-Host "      From: $scpSource"
    Write-Host "      To:   $scpDest"

    wsl scp -r $scpSource $scpDest

    Write-Host "      Backup copied successfully" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Failed to copy backup to local machine" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Step 3: Clean up remote backup (unless -KeepRemote specified)
if (-not $KeepRemote) {
    Write-Host "`n[3/3] Cleaning up remote backup..." -ForegroundColor Green
    try {
        wsl ssh chad.walker@$RemoteHost "rmdir /s /q `"$remoteBackupPath`""
        Write-Host "      Remote backup removed" -ForegroundColor Green
    }
    catch {
        Write-Host "WARNING: Failed to remove remote backup" -ForegroundColor Yellow
    }
}
else {
    Write-Host "`n[3/3] Keeping remote backup at: $remoteBackupPath" -ForegroundColor Yellow
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Backup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Get backup size
$backupSize = (Get-ChildItem -Path $localBackupPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
$fileCount = (Get-ChildItem -Path $localBackupPath -Recurse -File).Count

Write-Host "Location:   $localBackupPath"
Write-Host "Size:       $([math]::Round($backupSize, 2)) MB"
Write-Host "Files:      $fileCount"
Write-Host "========================================`n" -ForegroundColor Cyan

# List backed up collections
Write-Host "Backed up collections:" -ForegroundColor Cyan
Get-ChildItem -Path (Join-Path $localBackupPath $Database) -Filter "*.bson" | ForEach-Object {
    $collName = $_.BaseName
    $size = [math]::Round($_.Length / 1KB, 1)
    Write-Host "  - $collName ($size KB)"
}

Write-Host "`nTo restore this backup, run:" -ForegroundColor Yellow
Write-Host "  mongorestore --db $Database `"$localBackupPath\$Database`"" -ForegroundColor White
