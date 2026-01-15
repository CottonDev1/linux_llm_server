# ============================================================================
# Run Playwright Tests with Allure Reporting
# ============================================================================
# Usage:
#   .\scripts\run-tests-allure.ps1                    # Run all tests
#   .\scripts\run-tests-allure.ps1 -TestFile "audio"  # Run specific test file
#   .\scripts\run-tests-allure.ps1 -Headed            # Run in headed mode
#   .\scripts\run-tests-allure.ps1 -OpenReport        # Open report after tests
#   .\scripts\run-tests-allure.ps1 -Install           # Install dependencies first
# ============================================================================

param(
    [string]$TestFile = "",        # Specific test file pattern (e.g., "audio", "admin")
    [switch]$Headed = $false,      # Run tests in headed mode (visible browser)
    [switch]$OpenReport = $false,  # Open Allure report after tests complete
    [switch]$Install = $false,     # Install/update dependencies before running
    [switch]$CleanResults = $true, # Clean previous results before running
    [string]$BaseUrl = "",         # Override base URL (default: http://localhost:3000)
    [int]$Workers = 1              # Number of parallel workers
)

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AllureResultsDir = Join-Path $ProjectRoot "allure-results"
$AllureReportDir = Join-Path $ProjectRoot "allure-report"
$PlaywrightReportDir = Join-Path $ProjectRoot "playwright-report"
$TestResultsDir = Join-Path $ProjectRoot "test-results"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Step($text) {
    Write-Host "[*] $text" -ForegroundColor Yellow
}

function Write-Success($text) {
    Write-Host "[+] $text" -ForegroundColor Green
}

function Write-Error($text) {
    Write-Host "[!] $text" -ForegroundColor Red
}

# Navigate to project root
Set-Location $ProjectRoot

Write-Header "Playwright Test Runner with Allure Reporting"
Write-Host "Project Root: $ProjectRoot"
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ============================================================================
# Step 1: Install dependencies if requested
# ============================================================================
if ($Install) {
    Write-Header "Installing Dependencies"

    Write-Step "Installing npm packages..."
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Error "npm install failed!"
        exit 1
    }

    Write-Step "Installing Playwright browsers..."
    npx playwright install chromium
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Playwright browser install failed!"
        exit 1
    }

    Write-Success "Dependencies installed successfully"
}

# ============================================================================
# Step 2: Clean previous results
# ============================================================================
if ($CleanResults) {
    Write-Header "Cleaning Previous Results"

    if (Test-Path $AllureResultsDir) {
        Write-Step "Removing $AllureResultsDir..."
        Remove-Item -Recurse -Force $AllureResultsDir
    }

    if (Test-Path $AllureReportDir) {
        Write-Step "Removing $AllureReportDir..."
        Remove-Item -Recurse -Force $AllureReportDir
    }

    if (Test-Path $TestResultsDir) {
        Write-Step "Removing $TestResultsDir..."
        Remove-Item -Recurse -Force $TestResultsDir
    }

    Write-Success "Previous results cleaned"
}

# ============================================================================
# Step 3: Check if Java and Allure CLI are available
# ============================================================================
Write-Header "Checking Java and Allure CLI"

$JavaAvailable = $false
$AllureCommand = $null

# Check for Java (required by Allure)
try {
    $null = java -version 2>&1
    $JavaAvailable = $true
    Write-Success "Java is available"
}
catch {
    Write-Error "Java not found - required for Allure reports!"
    Write-Host ""
    Write-Host "Install Java from one of these options:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://adoptium.net/" -ForegroundColor White
    Write-Host "  2. Via winget: winget install EclipseAdoptium.Temurin.17.JRE" -ForegroundColor White
    Write-Host "  3. Via Chocolatey: choco install temurin17jre" -ForegroundColor White
    Write-Host ""
    Write-Host "Tests will still run, but Allure report won't be generated." -ForegroundColor Yellow
    Write-Host "Playwright HTML report will still be available." -ForegroundColor Yellow
    Write-Host ""
}

if ($JavaAvailable) {
    # Check for allure-commandline in node_modules
    $NodeAllure = Join-Path $ProjectRoot "node_modules\.bin\allure.cmd"
    if (Test-Path $NodeAllure) {
        $AllureCommand = $NodeAllure
        Write-Success "Found Allure CLI in node_modules"
    }
    else {
        # Check for global allure
        $GlobalAllure = Get-Command allure -ErrorAction SilentlyContinue
        if ($GlobalAllure) {
            $AllureCommand = "allure"
            Write-Success "Found global Allure CLI"
        }
        else {
            Write-Error "Allure CLI not found!"
            Write-Host "Install it with: npm install --save-dev allure-commandline"
        }
    }
}

# ============================================================================
# Step 4: Build test command
# ============================================================================
Write-Header "Running Playwright Tests"

# Build the command arguments
$TestArgs = @()

# Add specific test file if provided
if ($TestFile) {
    $TestArgs += "--grep"
    $TestArgs += $TestFile
    Write-Step "Running tests matching: $TestFile"
}

# Add headed mode
if ($Headed) {
    $env:HEADLESS = "false"
    Write-Step "Running in headed mode (browser visible)"
}
else {
    $env:HEADLESS = "true"
}

# Override base URL
if ($BaseUrl) {
    $env:BASE_URL = $BaseUrl
    Write-Step "Using base URL: $BaseUrl"
}

# Set workers
$TestArgs += "--workers=$Workers"
Write-Step "Using $Workers worker(s)"

# ============================================================================
# Step 5: Run tests
# ============================================================================
Write-Step "Executing tests..."
Write-Host ""

$StartTime = Get-Date

# Run Playwright tests
$TestCommand = "npx playwright test $($TestArgs -join ' ')"
Write-Host "Command: $TestCommand" -ForegroundColor DarkGray
Write-Host ""

Invoke-Expression $TestCommand
$TestExitCode = $LASTEXITCODE

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Step "Tests completed in $($Duration.TotalSeconds.ToString('F2')) seconds"

if ($TestExitCode -eq 0) {
    Write-Success "All tests passed!"
}
else {
    Write-Error "Some tests failed (exit code: $TestExitCode)"
}

# ============================================================================
# Step 6: Generate Allure Report
# ============================================================================
Write-Header "Generating Allure Report"

if (!(Test-Path $AllureResultsDir)) {
    Write-Error "No Allure results found at $AllureResultsDir"
    Write-Host "Make sure tests ran and generated results"
}
elseif ($AllureCommand) {
    Write-Step "Generating report from results..."

    & $AllureCommand generate $AllureResultsDir -o $AllureReportDir --clean
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to generate Allure report!"
    }
    else {
        Write-Success "Allure report generated at: $AllureReportDir"
    }
}
else {
    Write-Host ""
    Write-Host "Skipping Allure report generation (Java or Allure CLI not available)" -ForegroundColor Yellow
    Write-Host "Allure results saved to: $AllureResultsDir" -ForegroundColor White
    Write-Host ""
}

# ============================================================================
# Step 7: Open report if requested
# ============================================================================
if ($OpenReport -and $AllureCommand) {
    Write-Header "Opening Allure Report"

    Write-Step "Starting Allure server..."
    Write-Host "Press Ctrl+C to stop the server"
    Write-Host ""

    & $AllureCommand open $AllureReportDir
}
elseif ($OpenReport) {
    Write-Header "Opening Playwright Report"

    # Fall back to Playwright HTML report if Allure not available
    $PlaywrightIndex = Join-Path $PlaywrightReportDir "index.html"
    if (Test-Path $PlaywrightIndex) {
        Write-Step "Opening Playwright HTML report..."
        Start-Process $PlaywrightIndex
    }
    else {
        Write-Error "No reports available to open"
    }
}
else {
    Write-Host ""
    Write-Host "To view reports:" -ForegroundColor White

    if ($AllureCommand) {
        Write-Host ""
        Write-Host "  Allure Report (recommended):" -ForegroundColor Cyan
        Write-Host "    .\scripts\run-tests-allure.ps1 -OpenReport" -ForegroundColor White
        Write-Host "    npx allure open allure-report" -ForegroundColor White
        Write-Host "    npx allure serve allure-results" -ForegroundColor White
    }

    Write-Host ""
    Write-Host "  Playwright HTML Report:" -ForegroundColor Cyan
    Write-Host "    npx playwright show-report" -ForegroundColor White
    Write-Host "    Start-Process playwright-report\index.html" -ForegroundColor White
}

# ============================================================================
# Summary
# ============================================================================
Write-Header "Summary"

Write-Host "Test Exit Code: $TestExitCode"
Write-Host "Duration: $($Duration.TotalSeconds.ToString('F2')) seconds"
Write-Host ""
Write-Host "Reports generated:"
Write-Host "  - Allure Report: $AllureReportDir\index.html"
Write-Host "  - Playwright Report: $PlaywrightReportDir\index.html"
Write-Host ""

# Exit with test exit code
exit $TestExitCode
