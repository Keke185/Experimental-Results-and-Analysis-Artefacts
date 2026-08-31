# Supplementary Experiment B: 10-level cold-start CPU sweep (cold cache).
# Usage: powershell -ExecutionPolicy Bypass -File .\exp_B_coldstart_cpu_sweep.ps1

$cpuLevels = @("0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0")
$repeats = 2
$results = @()
$port = 8011
$containerName = "coldstart_cpu_sweep"

function Wait-Healthy($port, $timeoutSec) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$port/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
    return $false
}

foreach ($cpu in $cpuLevels) {
    for ($i = 1; $i -le $repeats; $i++) {
        docker rm -f $containerName 2>$null | Out-Null
        docker run -d --name $containerName --cpus=$cpu --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -e NODE_ROLE="cold-$cpu" implement-edge_unsandboxed | Out-Null
        $ok = Wait-Healthy $port 150
        if ($ok) {
            $resp = curl.exe -s http://localhost:$port/startup_timing | ConvertFrom-Json
            $results += [PSCustomObject]@{ cpu = $cpu; trial = $i; total_ms = $resp.total_cold_start_ms; model_load_ms = $resp.model_load_ms }
            Write-Host "cpu=$cpu trial=$i -> $($resp.total_cold_start_ms) ms"
        } else {
            Write-Host "cpu=$cpu trial=$i -> TIMEOUT" -ForegroundColor Red
            $results += [PSCustomObject]@{ cpu = $cpu; trial = $i; total_ms = "TIMEOUT"; model_load_ms = "TIMEOUT" }
        }
        docker rm -f $containerName | Out-Null
    }
}

$results | Export-Csv -Path expB_coldstart_cpu_sweep_results.csv -NoTypeInformation
Write-Host "`nSaved expB_coldstart_cpu_sweep_results.csv ($($results.Count) rows)" -ForegroundColor Green
$results | Format-Table -AutoSize
