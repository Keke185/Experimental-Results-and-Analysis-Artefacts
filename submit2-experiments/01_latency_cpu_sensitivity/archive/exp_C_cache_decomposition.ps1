# Supplementary Experiment C: cold-cache vs warm-cache decomposition.
#
# Splits cold-start time into its two components: network-download time

# Usage: powershell -ExecutionPolicy Bypass -File .\exp_C_cache_decomposition.ps1

$cpuLevels = @("0.5", "1.0", "4.0")
$repeats = 3
$results = @()
$port = 8012
$containerName = "coldstart_decomp"

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

docker volume create hf-cache | Out-Null

#warm the shared cache once ,so warm trials never hit the network
Write-Host "Priming hf-cache volume" -ForegroundColor Cyan
docker rm -f $containerName 2>$null | Out-Null
docker run -d --name $containerName --cpus="1.0" --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface -e NODE_ROLE=warm-prime implement-edge_unsandboxed | Out-Null
Wait-Healthy $port 90 | Out-Null
docker rm -f $containerName | Out-Null
Write-Host "Cache warmed`n" -ForegroundColor Cyan

foreach ($cpu in $cpuLevels) {
    for ($i = 1; $i -le $repeats; $i++) {

        #COLD: fresh download, no cache mount
        docker rm -f $containerName 2>$null | Out-Null
        docker run -d --name $containerName --cpus=$cpu --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -e NODE_ROLE="cold-$cpu" implement-edge_unsandboxed | Out-Null
        $ok = Wait-Healthy $port 120

        if ($ok) {
            $resp = curl.exe -s http://localhost:$port/startup_timing | ConvertFrom-Json
            $results += [PSCustomObject]@{ cpu = $cpu; cache = "cold"; trial = $i; total_ms = $resp.total_cold_start_ms; model_load_ms = $resp.model_load_ms }
            Write-Host "cold  cpu=$cpu trial=$i -> $($resp.total_cold_start_ms) ms"
        } else {
            Write-Host "cold  cpu=$cpu trial=$i -> TIMEOUT" -ForegroundColor Red
        }

        docker rm -f $containerName | Out-Null

        docker run -d --name $containerName --cpus=$cpu --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface -e NODE_ROLE="warm-$cpu" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 implement-edge_unsandboxed | Out-Null
        $ok = Wait-Healthy $port 90
        if ($ok) {
            $resp = curl.exe -s http://localhost:$port/startup_timing | ConvertFrom-Json
            $results += [PSCustomObject]@{ cpu = $cpu; cache = "warm"; trial = $i; total_ms = $resp.total_cold_start_ms; model_load_ms = $resp.model_load_ms }
            Write-Host "warm  cpu=$cpu trial=$i -> $($resp.total_cold_start_ms) ms"
        } else {
            Write-Host "warm  cpu=$cpu trial=$i -> TIMEOUT" -ForegroundColor Red
        }
        docker rm -f $containerName | Out-Null
    }
}

$results | Export-Csv -Path expC_cache_decomposition_results.csv -NoTypeInformation
Write-Host "`nSaved expC_cache_decomposition_results.csv ($($results.Count) rows)" -ForegroundColor Green
$results | Format-Table -AutoSize
