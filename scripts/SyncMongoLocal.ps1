# Sync MongoDB data from EWRSPT-AI (VM) to local Atlas deployment
# Usage: .\sync-atlas-local.ps1
#
# Prerequisites:
#   - Docker Desktop running with local27019 container
#   - MongoDB Database Tools installed (mongodump, mongorestore)
#   - Network access to EWRSPT-AI:27018

param(
    [string]$SourceHost = "EWRSPT-AI",
    [int]$SourcePort = 27018,
    [int]$LocalPort = 27019,
    [string]$Database = "rag_server"
)

$SOURCE_URI = "mongodb://${SourceHost}:${SourcePort}/?directConnection=true"
$LOCAL_URI = "mongodb://localhost:${LocalPort}/?directConnection=true"
$BACKUP_DIR = "$env:TEMP\mongodb_sync_$Database"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MongoDB Atlas Local Sync Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Source: $SOURCE_URI" -ForegroundColor Gray
Write-Host "Target: $LOCAL_URI" -ForegroundColor Gray
Write-Host "Database: $Database" -ForegroundColor Gray
Write-Host ""

# Check if local container is running
$container = docker ps --filter "name=local$LocalPort" --format "{{.Names}}" 2>$null
if (-not $container) {
    Write-Host "ERROR: Docker container 'local$LocalPort' is not running!" -ForegroundColor Red
    Write-Host "Start it with: atlas deployments start local$LocalPort" -ForegroundColor Yellow
    exit 1
}

# Step 1: Export from VM
Write-Host "[1/3] Exporting from $SourceHost..." -ForegroundColor Yellow
$dumpResult = mongodump --uri="$SOURCE_URI" --db=$Database --out="$BACKUP_DIR" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Export failed!" -ForegroundColor Red
    Write-Host $dumpResult -ForegroundColor Red
    exit 1
}
Write-Host "      Export complete!" -ForegroundColor Green

# Step 2: Import to local
Write-Host "[2/3] Importing to localhost:$LocalPort..." -ForegroundColor Yellow
$restoreResult = mongorestore --uri="$LOCAL_URI" --db=$Database --drop "$BACKUP_DIR\$Database" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Import failed!" -ForegroundColor Red
    Write-Host $restoreResult -ForegroundColor Red
    exit 1
}
Write-Host "      Import complete!" -ForegroundColor Green

# Step 3: Create vector search indexes
Write-Host "[3/3] Creating vector search indexes..." -ForegroundColor Yellow

$indexScript = @'
use("rag_server");

const collections = [
    "documents",
    "code_context",
    "sql_examples",
    "sql_schema_context",
    "sql_stored_procedures",
    "code_methods",
    "code_classes",
    "code_callgraph",
    "code_eventhandlers",
    "code_dboperations",
    "sql_knowledge",
    "sql_failed_queries",
    "audio_analysis"
];

let created = 0;
collections.forEach(col => {
    const indexName = col + "_vector_index";
    try {
        db.getCollection(col).dropSearchIndex(indexName);
    } catch(e) {}

    try {
        db.getCollection(col).createSearchIndex({
            name: indexName,
            type: "vectorSearch",
            definition: {
                fields: [{
                    type: "vector",
                    path: "vector",
                    numDimensions: 384,
                    similarity: "cosine"
                }]
            }
        });
        created++;
        print("  Created: " + indexName);
    } catch(e) {
        print("  Skipped: " + indexName + " (collection may be empty)");
    }
});

print("");
print("Created " + created + " vector search indexes");
'@

$indexScript | docker exec -i "local$LocalPort" mongosh --quiet 2>&1 | ForEach-Object {
    if ($_ -match "Created:|Skipped:") {
        Write-Host $_ -ForegroundColor Gray
    }
}

# Cleanup
Remove-Item -Recurse -Force "$BACKUP_DIR" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Sync Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Connection string: mongodb://localhost:${LocalPort}/?directConnection=true" -ForegroundColor Cyan
Write-Host ""
