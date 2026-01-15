<#
.SYNOPSIS
    Pull latest git changes and analyze C# code for all configured repositories.

.DESCRIPTION
    This script:
    1. Pulls the latest changes from each Git repository
    2. Runs Roslyn C# code analysis on each repository
    3. Stores the analysis results in MongoDB with vector embeddings

    Prerequisites:
    - Python FastAPI server running on port 8100 (or configured port)
    - MongoDB running
    - Git repositories accessible

.PARAMETER ApiUrl
    Base URL for the Python API. Default: http://localhost:8100

.PARAMETER SkipPull
    Skip the git pull step and only run analysis

.PARAMETER Verbose
    Show detailed output

.EXAMPLE
    .\update-and-analyze-all.ps1

.EXAMPLE
    .\update-and-analyze-all.ps1 -ApiUrl "http://localhost:8001" -SkipPull
#>

param(
    [string]$ApiUrl = "http://localhost:8100",
    [switch]$SkipPull,
    [switch]$VerboseOutput
)

# Repository configurations
$repositories = @(
    @{
        Name = "Marketing"
        Path = "C:\Projects\Git\Marketing"
        Project = "Marketing"
    },
    @{
        Name = "Gin"
        Path = "C:\Projects\Git\Gin"
        Project = "Gin"
    },
    @{
        Name = "Warehouse"
        Path = "C:\Projects\Git\Warehouse"
        Project = "Warehouse"
    },
    @{
        Name = "Provider"
        Path = "C:\Projects\Git\Provider\Provider%20Systems"
        Project = "Provider"
    }
)

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("-" * 50) -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor DarkCyan
    Write-Host ("-" * 50) -ForegroundColor DarkCyan
}

function Test-ApiAvailable {
    param([string]$Url)

    try {
        $response = Invoke-RestMethod -Uri "$Url/status" -Method Get -TimeoutSec 10
        return $response.mongodb.connected -eq $true
    }
    catch {
        return $false
    }
}

function Invoke-GitPull {
    param(
        [string]$RepoName,
        [string]$RepoPath
    )

    Write-Section "Pulling: $RepoName"

    # Check if path exists
    if (-not (Test-Path $RepoPath)) {
        Write-ColorOutput "  ERROR: Path does not exist: $RepoPath" "Red"
        return @{ Success = $false; Error = "Path not found" }
    }

    # Check if it's a git repository
    $gitDir = Join-Path $RepoPath ".git"
    if (-not (Test-Path $gitDir)) {
        Write-ColorOutput "  ERROR: Not a git repository: $RepoPath" "Red"
        return @{ Success = $false; Error = "Not a git repository" }
    }

    try {
        # Use the Git API endpoint
        $body = @{
            repo = $RepoName.ToLower()
            analyzeChanges = $true
            maxFilesToAnalyze = 50
            includeCodeAnalysis = $false
        } | ConvertTo-Json

        $response = Invoke-RestMethod -Uri "$ApiUrl/git/pull" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120

        if ($response.success) {
            if ($response.hasChanges) {
                Write-ColorOutput "  SUCCESS: Pulled changes" "Green"
                Write-ColorOutput "  Changed files: $($response.changedFiles.Count)" "Gray"
            }
            else {
                Write-ColorOutput "  Already up to date" "Yellow"
            }
            return @{ Success = $true; HasChanges = $response.hasChanges; ChangedFiles = $response.changedFiles }
        }
        else {
            Write-ColorOutput "  FAILED: $($response.message)" "Red"
            return @{ Success = $false; Error = $response.message }
        }
    }
    catch {
        # Fallback to direct git pull if API fails
        Write-ColorOutput "  API call failed, using direct git pull..." "Yellow"

        Push-Location $RepoPath
        try {
            $output = git pull 2>&1
            $exitCode = $LASTEXITCODE

            if ($exitCode -eq 0) {
                if ($output -match "Already up to date" -or $output -match "Already up-to-date") {
                    Write-ColorOutput "  Already up to date" "Yellow"
                    return @{ Success = $true; HasChanges = $false }
                }
                else {
                    Write-ColorOutput "  SUCCESS: Pulled changes" "Green"
                    if ($VerboseOutput) {
                        Write-ColorOutput "  $output" "Gray"
                    }
                    return @{ Success = $true; HasChanges = $true }
                }
            }
            else {
                Write-ColorOutput "  FAILED: $output" "Red"
                return @{ Success = $false; Error = $output }
            }
        }
        finally {
            Pop-Location
        }
    }
}

function Invoke-RoslynAnalysis {
    param(
        [string]$RepoName,
        [string]$RepoPath,
        [string]$ProjectName
    )

    Write-Section "Analyzing: $RepoName"

    # Check if path exists
    if (-not (Test-Path $RepoPath)) {
        Write-ColorOutput "  ERROR: Path does not exist: $RepoPath" "Red"
        return @{ Success = $false; Error = "Path not found" }
    }

    # Find C# files to confirm there's something to analyze
    $csFiles = Get-ChildItem -Path $RepoPath -Filter "*.cs" -Recurse -ErrorAction SilentlyContinue
    if ($csFiles.Count -eq 0) {
        Write-ColorOutput "  No C# files found in $RepoPath" "Yellow"
        return @{ Success = $true; Classes = 0; Methods = 0; Message = "No C# files" }
    }

    Write-ColorOutput "  Found $($csFiles.Count) C# files" "Gray"

    try {
        $body = @{
            input_path = $RepoPath
            project = $ProjectName
            store = $true
        } | ConvertTo-Json

        Write-ColorOutput "  Running Roslyn analysis (this may take a few minutes)..." "Gray"

        $response = Invoke-RestMethod -Uri "$ApiUrl/roslyn/analyze" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600

        if ($response.success) {
            $stats = $response.storage_stats
            $summary = $response.data_summary

            Write-ColorOutput "  SUCCESS: Analysis complete" "Green"
            Write-ColorOutput "    Classes:        $($summary.classes)" "White"
            Write-ColorOutput "    Methods:        $($summary.methods)" "White"
            Write-ColorOutput "    Call Graph:     $($summary.call_graph)" "White"
            Write-ColorOutput "    Event Handlers: $($summary.event_handlers)" "White"
            Write-ColorOutput "    DB Operations:  $($summary.db_operations)" "White"

            return @{
                Success = $true
                Classes = $summary.classes
                Methods = $summary.methods
                CallGraph = $summary.call_graph
                EventHandlers = $summary.event_handlers
                DbOperations = $summary.db_operations
            }
        }
        else {
            Write-ColorOutput "  FAILED: Analysis returned unsuccessful" "Red"
            return @{ Success = $false; Error = "Analysis failed" }
        }
    }
    catch {
        $errorMessage = $_.Exception.Message
        Write-ColorOutput "  ERROR: $errorMessage" "Red"

        # Try to get more details from the response
        if ($_.Exception.Response) {
            try {
                $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $responseBody = $reader.ReadToEnd()
                Write-ColorOutput "  Details: $responseBody" "Red"
            }
            catch {}
        }

        return @{ Success = $false; Error = $errorMessage }
    }
}

function Get-RoslynStats {
    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/roslyn/stats" -Method Get -TimeoutSec 30
        return $response
    }
    catch {
        return $null
    }
}

# ============================================================================
# Main Script
# ============================================================================

Write-Header "Git Pull & Roslyn Analysis Script"
Write-Host "API URL: $ApiUrl"
Write-Host "Skip Pull: $SkipPull"
Write-Host "Repositories: $($repositories.Count)"
Write-Host ""

# Check API availability
Write-Host "Checking API availability..." -NoNewline
if (-not (Test-ApiAvailable -Url $ApiUrl)) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "The Python API server is not available at $ApiUrl" -ForegroundColor Red
    Write-Host "Please start it with:" -ForegroundColor Yellow
    Write-Host "  cd C:\Projects\Python_NodeServer\python_services" -ForegroundColor Gray
    Write-Host "  python -m uvicorn main:app --host 0.0.0.0 --port 8100" -ForegroundColor Gray
    Write-Host ""
    exit 1
}
Write-Host " OK" -ForegroundColor Green

# Get initial stats
$initialStats = Get-RoslynStats
if ($initialStats) {
    Write-Host ""
    Write-Host "Current MongoDB Stats:" -ForegroundColor Cyan
    Write-Host "  Classes: $($initialStats.classes)"
    Write-Host "  Methods: $($initialStats.methods)"
    Write-Host "  Call Graph: $($initialStats.callgraph)"
    Write-Host "  Event Handlers: $($initialStats.eventhandlers)"
}

# Track results
$results = @{
    Pull = @()
    Analysis = @()
    StartTime = Get-Date
}

# Process each repository
foreach ($repo in $repositories) {
    Write-Header "Processing: $($repo.Name)"

    # Step 1: Git Pull (unless skipped)
    if (-not $SkipPull) {
        $pullResult = Invoke-GitPull -RepoName $repo.Name -RepoPath $repo.Path
        $results.Pull += @{
            Name = $repo.Name
            Success = $pullResult.Success
            HasChanges = $pullResult.HasChanges
            Error = $pullResult.Error
        }
    }
    else {
        Write-ColorOutput "  Skipping git pull" "Yellow"
    }

    # Step 2: Roslyn Analysis
    $analysisResult = Invoke-RoslynAnalysis -RepoName $repo.Name -RepoPath $repo.Path -ProjectName $repo.Project
    $results.Analysis += @{
        Name = $repo.Name
        Success = $analysisResult.Success
        Classes = $analysisResult.Classes
        Methods = $analysisResult.Methods
        CallGraph = $analysisResult.CallGraph
        EventHandlers = $analysisResult.EventHandlers
        DbOperations = $analysisResult.DbOperations
        Error = $analysisResult.Error
    }
}

# ============================================================================
# Summary
# ============================================================================

$endTime = Get-Date
$duration = $endTime - $results.StartTime

Write-Header "Summary"

# Git Pull Summary
if (-not $SkipPull) {
    Write-Host "Git Pull Results:" -ForegroundColor Cyan
    foreach ($pull in $results.Pull) {
        $status = if ($pull.Success) { "[OK]" } else { "[FAIL]" }
        $color = if ($pull.Success) { "Green" } else { "Red" }
        $changes = if ($pull.HasChanges) { "(has changes)" } else { "(no changes)" }
        Write-Host "  $status $($pull.Name) $changes" -ForegroundColor $color
    }
    Write-Host ""
}

# Analysis Summary
Write-Host "Analysis Results:" -ForegroundColor Cyan
$totalClasses = 0
$totalMethods = 0
$totalCallGraph = 0
$totalEventHandlers = 0

foreach ($analysis in $results.Analysis) {
    $status = if ($analysis.Success) { "[OK]" } else { "[FAIL]" }
    $color = if ($analysis.Success) { "Green" } else { "Red" }

    if ($analysis.Success) {
        Write-Host "  $status $($analysis.Name): $($analysis.Classes) classes, $($analysis.Methods) methods" -ForegroundColor $color
        $totalClasses += $analysis.Classes
        $totalMethods += $analysis.Methods
        $totalCallGraph += $analysis.CallGraph
        $totalEventHandlers += $analysis.EventHandlers
    }
    else {
        Write-Host "  $status $($analysis.Name): $($analysis.Error)" -ForegroundColor $color
    }
}

Write-Host ""
Write-Host "Totals:" -ForegroundColor Cyan
Write-Host "  Classes:        $totalClasses"
Write-Host "  Methods:        $totalMethods"
Write-Host "  Call Graph:     $totalCallGraph"
Write-Host "  Event Handlers: $totalEventHandlers"

# Final stats from MongoDB
$finalStats = Get-RoslynStats
if ($finalStats) {
    Write-Host ""
    Write-Host "Final MongoDB Stats:" -ForegroundColor Cyan
    Write-Host "  Classes: $($finalStats.classes)"
    Write-Host "  Methods: $($finalStats.methods)"
    Write-Host "  Call Graph: $($finalStats.callgraph)"
    Write-Host "  Event Handlers: $($finalStats.eventhandlers)"
    Write-Host "  Total Documents: $($finalStats.total)"
}

Write-Host ""
Write-Host "Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor Cyan
Write-Host ""
Write-Host "Done!" -ForegroundColor Green
