# Backup MongoDB on EWRSPT-AI
$backupDir = "C:\backups\mongodb"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupPath = "$backupDir\rag_server_$timestamp"

# Create backup directory
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

# Run mongodump with URI connection string
Write-Host "Starting MongoDB backup to $backupPath..."
$mongodump = "C:\Program Files\MongoDB\Tools\100\bin\mongodump.exe"
$uri = "mongodb://localhost:27018/?directConnection=true"
& $mongodump --uri=$uri --db rag_server --out $backupPath 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Backup completed successfully: $backupPath" -ForegroundColor Green
    $size = (Get-ChildItem $backupPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "Total size: $([math]::Round($size, 2)) MB"
} else {
    Write-Host "Backup failed!" -ForegroundColor Red
    exit 1
}
