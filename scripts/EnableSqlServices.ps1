# Enable and Start SQL Server 2019 Services
# Run as Administrator

Write-Host "=== SQL Server 2019 Service Manager ===" -ForegroundColor Cyan
Write-Host ""

# Get all SQL Server related services
$sqlServices = Get-Service | Where-Object {
    $_.Name -like "*SQL*" -or
    $_.Name -like "*MSSQL*" -or
    $_.DisplayName -like "*SQL Server*"
}

if ($sqlServices.Count -eq 0) {
    Write-Host "No SQL Server services found!" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($sqlServices.Count) SQL Server services:" -ForegroundColor Yellow
Write-Host ""

foreach ($service in $sqlServices) {
    Write-Host "Service: $($service.DisplayName)" -ForegroundColor White
    Write-Host "  Name: $($service.Name)"
    Write-Host "  Status: $($service.Status)"
    Write-Host "  StartType: $($service.StartType)"

    # Set service to Automatic startup if it's Disabled or Manual
    if ($service.StartType -eq 'Disabled' -or $service.StartType -eq 'Manual') {
        try {
            Set-Service -Name $service.Name -StartupType Automatic -ErrorAction Stop
            Write-Host "  -> Changed StartupType to Automatic" -ForegroundColor Green
        } catch {
            Write-Host "  -> Failed to change StartupType: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    # Start the service if it's not running
    if ($service.Status -ne 'Running') {
        try {
            Start-Service -Name $service.Name -ErrorAction Stop
            Write-Host "  -> Started service" -ForegroundColor Green
        } catch {
            Write-Host "  -> Failed to start: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  -> Already running" -ForegroundColor Green
    }

    Write-Host ""
}

Write-Host "=== Final Status ===" -ForegroundColor Cyan
Get-Service | Where-Object {
    $_.Name -like "*SQL*" -or
    $_.Name -like "*MSSQL*" -or
    $_.DisplayName -like "*SQL Server*"
} | Format-Table Name, DisplayName, Status, StartType -AutoSize

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
