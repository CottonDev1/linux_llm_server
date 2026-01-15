# Check embedding service status
Write-Host "Service Status:"
Get-Service LlamaCpp-Embedding | Select-Object Name, Status

Write-Host "`nPort 8083 Listeners:"
Get-NetTCPConnection -LocalPort 8083 -State Listen -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, OwningProcess

Write-Host "`nLocal Health Check:"
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8083/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "Status: $($response.StatusCode)"
    Write-Host "Content: $($response.Content)"
} catch {
    Write-Host "Failed: $_"
}

Write-Host "`nFirewall Rules for 8083:"
Get-NetFirewallRule -Direction Inbound |
    Where-Object { $_.Enabled -eq "True" } |
    Get-NetFirewallPortFilter -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -eq 8083 } |
    Select-Object @{N="Rule";E={(Get-NetFirewallRule -AssociatedNetFirewallPortFilter $_).DisplayName}}, LocalPort
