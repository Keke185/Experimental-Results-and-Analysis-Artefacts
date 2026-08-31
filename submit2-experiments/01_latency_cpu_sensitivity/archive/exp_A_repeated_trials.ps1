# Supplementary Experiment A: Repeated-trials noise quantification.
# Usage: powershell -ExecutionPolicy Bypass -File .\exp_A_repeated_trials.ps1

$repeats = 5
$results = @()

function Wait-Healthy($url, $timeoutSec) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
    return $false
}

Write-Host "Resetting all containers" -ForegroundColor Cyan
docker compose down 2>$null | Out-Null
docker rm -f vcse_edge_sandboxed 2>$null | Out-Null

for ($i = 1; $i -le $repeats; $i++) {
    Write-Host "`n--- Trial $i / $repeats ---" -ForegroundColor Yellow

    #Cloud
    docker compose up -d cloud | Out-Null
    if (Wait-Healthy "http://localhost:8000/health" 90) {
        $resp = curl.exe -s http://localhost:8000/startup_timing | ConvertFrom-Json
        $results += [PSCustomObject]@{ config = "cloud"; trial = $i; total_ms = $resp.total_cold_start_ms }
        Write-Host "cloud trial=$i -> $($resp.total_cold_start_ms) ms"
    } else {
        Write-Host "cloud trial=$i -> TIMEOUT" -ForegroundColor Red
    }
    docker compose stop cloud | Out-Null
    docker compose rm -f cloud | Out-Null

    #Edge, unsandboxed
    docker compose up -d edge_unsandboxed | Out-Null
    if (Wait-Healthy "http://localhost:8001/health" 90) {
        $resp = curl.exe -s http://localhost:8001/startup_timing | ConvertFrom-Json
        $results += [PSCustomObject]@{ config = "edge_unsandboxed"; trial = $i; total_ms = $resp.total_cold_start_ms }
        Write-Host "edge_unsandboxed trial=$i -> $($resp.total_cold_start_ms) ms"
    } else {
        Write-Host "edge_unsandboxed trial=$i -> TIMEOUT" -ForegroundColor Red
    }
    docker compose stop edge_unsandboxed | Out-Null
    docker compose rm -f edge_unsandboxed | Out-Null

    #Edge, sandboxed
    docker run -d --name vcse_edge_sandboxed --cpus="1.0" --memory="512m" --network implement_vcse-net -p 8002:8000 -v implement_artifact-store:/data -e NODE_ROLE=edge-sandboxed implement-edge_unsandboxed | Out-Null
    if (Wait-Healthy "http://localhost:8002/health" 90) {
        $resp = curl.exe -s http://localhost:8002/startup_timing | ConvertFrom-Json
        $results += [PSCustomObject]@{ config = "edge_sandboxed_1.0cpu"; trial = $i; total_ms = $resp.total_cold_start_ms }
        Write-Host "edge_sandboxed trial=$i -> $($resp.total_cold_start_ms) ms"
    } else {
        Write-Host "edge_sandboxed trial=$i -> TIMEOUT" -ForegroundColor Red
    }
    docker rm -f vcse_edge_sandboxed | Out-Null
}

$results | Export-Csv -Path expA_repeated_trials_results.csv -NoTypeInformation
Write-Host "`nSaved expA_repeated_trials_results.csv ($($results.Count) rows)" -ForegroundColor Green
$results | Format-Table -AutoSize
