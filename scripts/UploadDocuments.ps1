# Upload documents from Desktop/AI folder
# Uses Python service directly for file processing and embedding
param(
    [string]$SourcePath = "C:\Users\chad.walker\Desktop\AI",
    [string]$Department = "customer_support",
    [string]$DocType = "documentation"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Document Upload Script (Python Service)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Source: $SourcePath"
Write-Host "Department: $Department"
Write-Host "Type: $DocType"
Write-Host "Endpoint: http://localhost:8001/documents/upload"
Write-Host ""

# Get list of files
$files = Get-ChildItem -Path $SourcePath -File -Recurse | Where-Object { $_.Extension -match '\.(pdf|docx?|xlsx?|txt|md)$' }
Write-Host "Found $($files.Count) documents to upload:" -ForegroundColor Yellow
$files | ForEach-Object { Write-Host "  - $($_.Name) ($([math]::Round($_.Length / 1KB, 1)) KB)" }
Write-Host ""

# Upload each file to Python service directly
$successCount = 0
$failCount = 0

foreach ($file in $files) {
    Write-Host "Uploading: $($file.Name)..." -NoNewline

    $curlArgs = @(
        '-s',
        '-X', 'POST',
        'http://localhost:8001/documents/upload',
        '-F', "file=@$($file.FullName)",
        '-F', "department=$Department",
        '-F', "type=$DocType"
    )

    try {
        $result = & curl.exe @curlArgs 2>&1
        $jsonResult = $result | ConvertFrom-Json

        if ($jsonResult.success) {
            $chunks = $jsonResult.chunks_created
            $chars = $jsonResult.text_extracted
            Write-Host " SUCCESS ($chunks chunks, $chars chars)" -ForegroundColor Green
            $successCount++
        } else {
            $error = $jsonResult.detail
            if (-not $error) { $error = $result }
            Write-Host " FAILED: $error" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host " ERROR: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   Raw: $result" -ForegroundColor DarkGray
        $failCount++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Upload Complete" -ForegroundColor Cyan
Write-Host "  Success: $successCount" -ForegroundColor Green
Write-Host "  Failed:  $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host "========================================" -ForegroundColor Cyan
