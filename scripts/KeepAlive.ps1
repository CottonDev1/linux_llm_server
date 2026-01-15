# KeepAlive.ps1 - Prevents session timeout by simulating mouse activity
# Run in a separate PowerShell window while migration is running
#
# Usage: .\KeepAlive.ps1
# Stop:  Ctrl+C

param(
    [int]$IntervalSeconds = 60,  # How often to move the mouse
    [int]$MovePixels = 1         # How many pixels to move
)

Add-Type -AssemblyName System.Windows.Forms

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Session Keep-Alive Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Interval: Every $IntervalSeconds seconds"
Write-Host "  Movement: $MovePixels pixel(s)"
Write-Host "  Press Ctrl+C to stop"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$count = 0
while ($true) {
    $pos = [System.Windows.Forms.Cursor]::Position

    # Move mouse slightly right then back
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(($pos.X + $MovePixels), $pos.Y)
    Start-Sleep -Milliseconds 50
    [System.Windows.Forms.Cursor]::Position = $pos

    $count++
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] Mouse nudged (count: $count)" -ForegroundColor Gray

    Start-Sleep -Seconds $IntervalSeconds
}
