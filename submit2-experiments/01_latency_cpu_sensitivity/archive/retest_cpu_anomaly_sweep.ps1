# Targeted re-test of the 0.5 -> 0.75 core steady-state latency anomaly.
#
# v2: uses `docker update --cpus` on the EXISTING persistent sandboxed
# container (vcse_edge_sandboxed, confirmed via `docker ps` to already be
# bound to port 8002) instead of trying to `docker run` a brand-new
# container on that same port.
#
# The script records the container's CURRENT cpu limit before touching
# anything, and restores it at the end (or removes the cap entirely if none
# was set beforehand).
#
# Usage: powershell -ExecutionPolicy Bypass -File .\retest_cpu_anomaly_sweep.ps1

$cpuLevels = @("0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
$containerName = "vcse_edge_sandboxed"
$settleSeconds = 3   # Pause briefly after each `docker update` command before proceeding with the test.

#Record the original CPU limits for later recovery.
$originalNanoCpus = docker inspect $containerName --format="{{.HostConfig.NanoCpus}}"
Write-Host "Current $containerName NanoCpus setting: $originalNanoCpus" -ForegroundColor Cyan
if ($originalNanoCpus -and $originalNanoCpus -ne "0") {
    $originalCores = [double]$originalNanoCpus / 1000000000
    Write-Host "(equivalent to --cpus=$originalCores -- will restore this when done)`n" -ForegroundColor Cyan
} else {
    $originalCores = $null
    Write-Host "(no CPU cap currently set -- will remove the cap when done)`n" -ForegroundColor Cyan
}

Write-Host "Re-testing 0.5~1.0 core range (6 levels) against the frozen dataset" -ForegroundColor Cyan

foreach ($cpu in $cpuLevels) {
    Write-Host "`n=== cpu=$cpu ===" -ForegroundColor Yellow

    docker update --cpus=$cpu $containerName | Out-Null
    Start-Sleep -Seconds $settleSeconds

    python retest_cpu_anomaly_client.py $cpu
}

#restore original CPU limit
if ($originalCores) {
    docker update --cpus=$originalCores $containerName | Out-Null
    Write-Host "`nRestored $containerName to --cpus=$originalCores" -ForegroundColor Cyan
} else {
    docker update --cpus=0 $containerName | Out-Null
    Write-Host "`nRestored $containerName to unlimited (no cap, as it was before this script ran)" -ForegroundColor Cyan
}

Write-Host "`nDone. Results appended to retest_0.5to1.0_results.csv -- send this file back and I'll compare it against the original 0.5/0.75-core numbers." -ForegroundColor Green
