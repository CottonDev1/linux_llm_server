# Test MongoDB connection and backup on EWRSPT-AI
$mongodump = "C:\Program Files\MongoDB\Tools\100\bin\mongodump.exe"

Write-Host "mongodump version:"
& $mongodump --version

Write-Host "`nTrying backup with explicit timeout..."
$backupPath = "C:\backups\mongodb\test_backup"
New-Item -ItemType Directory -Force -Path "C:\backups\mongodb" | Out-Null

# Try with explicit server selection timeout
& $mongodump --uri="mongodb://127.0.0.1:27018/?directConnection=true&serverSelectionTimeoutMS=60000" --db rag_server --out $backupPath

Write-Host "`nExit code: $LASTEXITCODE"
