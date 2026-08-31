$cpuLevels = @("0.1", "0.25", "0.5", "1.0", "3.0", "6.0")
$containerName = "vcse_edge_sandboxed"
$steadySettleSeconds = 20
$outFile = "loss_rate_clean_rerun.csv"

# Preserve original CPU limits
$originalNanoCpus = docker inspect $containerName --format="{{.HostConfig.NanoCpus}}"

Write-Host "Current $containerName NanoCpus setting: $originalNanoCpus" -ForegroundColor Cyan
if ($originalNanoCpus -and $originalNanoCpus -ne "0") {
    $originalCores = [double]$originalNanoCpus / 1000000000
} else {
    $originalCores = $null
}

if (Test-Path $outFile) {
    Write-Host "Removing existing $outFile so this is a clean single-batch file..." -ForegroundColor Yellow
    Remove-Item $outFile
}

Write-Host "Loss-rate re-test: 6 CPU levels x (immediate + steady) phases..." -ForegroundColor Cyan

foreach ($cpu in $cpuLevels) {
    Write-Host "`n=== cpu=$cpu ===" -ForegroundColor Yellow

    docker update --cpus=$cpu $containerName | Out-Null

    Write-Host "-- immediate phase (0s settle) --" -ForegroundColor Magenta
    python loss_rate_client.py $cpu immediate

    Write-Host "-- waiting ${steadySettleSeconds}s for genuine steady state --" -ForegroundColor Magenta
    Start-Sleep -Seconds $steadySettleSeconds

    Write-Host "-- steady phase (${steadySettleSeconds}s settle) --" -ForegroundColor Magenta
    python loss_rate_client.py $cpu steady
}

#restore original CPU limit
if ($originalCores) {
    docker update --cpus=$originalCores $containerName | Out-Null
    Write-Host "`nRestored $containerName to --cpus=$originalCores" -ForegroundColor Cyan
} else {
    docker update --cpus=0 $containerName | Out-Null
    Write-Host "`nRestored $containerName to unlimited" -ForegroundColor Cyan
}

Write-Host "`nDone. Saved to $outFile -- send this back and I'll rebuild the availability section, including whether the warmup-tax effect is real or a measurement artifact." -ForegroundColor Green
