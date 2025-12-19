# ============================================================================
# OVERNIGHT DEVICE PROCESSING SCRIPT
# Processes all remaining devices sequentially
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT BATCH PROCESSING" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# List of devices to process
$devices = @("A1", "A2", "A3", "B1")

# Activate virtual environment
& "C:\Users\au784422\Documents\Shared_Env\.env\Scripts\Activate.ps1"

# Navigate to project directory
Set-Location "C:\Users\au784422\OneDrive - Aarhus universitet\Lemming\Lemming_TIME_CORRECTION_CODE"

# Process each device
$totalDevices = $devices.Count
$completed = 0
$failed = @()

foreach ($device in $devices) {
    $completed++
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Processing Device $completed/$totalDevices : $device" -ForegroundColor Green
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    
    # Update config.py with current device
    $configPath = ".\config.py"
    $configContent = Get-Content $configPath -Raw
    $configContent = $configContent -replace 'DEVICE_ID = "[^"]*"', "DEVICE_ID = `"$device`""
    Set-Content -Path $configPath -Value $configContent
    
    # Run pipeline
    $startTime = Get-Date
    python main_pipeline.py
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] $device completed successfully" -ForegroundColor Green
        Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
        
        # Run validation
        Write-Host "`nRunning validation for $device..." -ForegroundColor Yellow
        python True_Validation_Time_Correction.py
    }
    else {
        Write-Host "`n[ERROR] $device failed!" -ForegroundColor Red
        $failed += $device
    }
    
    Write-Host "`n========================================`n" -ForegroundColor Gray
}

# Final summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BATCH PROCESSING COMPLETE!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Total devices: $totalDevices" -ForegroundColor White
Write-Host "Completed successfully: $($totalDevices - $failed.Count)" -ForegroundColor Green
Write-Host "Failed: $($failed.Count)" -ForegroundColor Red

if ($failed.Count -gt 0) {
    Write-Host "`nFailed devices:" -ForegroundColor Red
    foreach ($dev in $failed) {
        Write-Host "  - $dev" -ForegroundColor Red
    }
}

Write-Host "`nFinished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "`n"

# Keep window open
Read-Host "Press Enter to exit"