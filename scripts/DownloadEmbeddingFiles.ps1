# Download embedding service files locally
$setupDir = "C:\temp\embedding-setup"
New-Item -ItemType Directory -Force -Path $setupDir | Out-Null

# Download model
$modelPath = "$setupDir\nomic-embed-text-v1.5.f16.gguf"
if (-not (Test-Path $modelPath)) {
    Write-Host "Downloading nomic-embed-text-v1.5 model (~270 MB)..."
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf" -OutFile $modelPath
    Write-Host "Model downloaded"
} else {
    Write-Host "Model already exists"
}

# Show files
Write-Host "`nDownloaded files:"
Get-ChildItem $setupDir | ForEach-Object {
    $sizeMB = [math]::Round($_.Length/1MB, 2)
    Write-Host "  $($_.Name): $sizeMB MB"
}
