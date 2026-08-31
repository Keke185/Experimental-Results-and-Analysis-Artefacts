falling on different absolute scales—not suitable for merging into the same table)
$cpuLevels = @("0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0")
$containerName = "vcse_edge_sandboxed"
$settleSeconds = 3
$outFile = "cpu_sensitivity_clean_rerun.csv"

# Save the original CPU limit for later restoration
$originalNanoCpus = docker inspect $containerName --format="{{.HostConfig.NanoCpus}}"
Write-Host "Current $containerName NanoCpus setting: $originalNanoCpus" -ForegroundColor Cyan
if ($originalNanoCpus -and $originalNanoCpus -ne "0") {
    $originalCores = [double]$originalNanoCpus / 1000000000
    Write-Host "(equivalent to --cpus=$originalCores -- will restore this when done)`n" -ForegroundColor Cyan
} else {
    $originalCores = $null
    Write-Host "(no CPU cap currently set -- will remove the cap when done)`n" -ForegroundColor Cyan
}

#Run using the new file output
if (Test-Path $outFile) {
    Write-Host "Removing existing $outFile so this is a clean single-batch file" -ForegroundColor Yellow
    Remove-Item $outFile
}

Write-Host "Re-running the full 10-level CPU sweep (0.1 to 6.0 cores) against the frozen dataset" -ForegroundColor Cyan

foreach ($cpu in $cpuLevels) {
    Write-Host "`n=== cpu=$cpu ===" -ForegroundColor Yellow

    docker update --cpus=$cpu $containerName | Out-Null
    Start-Sleep -Seconds $settleSeconds

    python cpu_sweep_client.py $cpu $outFile
}

#Restore original CPU limits
if ($originalCores) {
    docker update --cpus=$originalCores $containerName | Out-Null
    Write-Host "`nRestored $containerName to --cpus=$originalCores" -ForegroundColor Cyan
} else {
    docker update --cpus=0 $containerName | Out-Null
    Write-Host "`nRestored $containerName to unlimited (no cap, as it was before this script ran)" -ForegroundColor Cyan
}

Write-Host "`nDone. All 10 levels saved in one clean batch to $outFile -- send this file back and I'll rebuild the report's steady-state section from it (dropping the old two-table story)" -ForegroundColor Green
