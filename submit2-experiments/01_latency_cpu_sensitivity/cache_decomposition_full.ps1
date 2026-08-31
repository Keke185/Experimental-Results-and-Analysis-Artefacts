# Clean, single-session re-run of the cold-cache vs warm-cache decomposition
# (0.5 / 1.0 / 4.0 cores x cold vs warm x 3 repeats = 18 trials). Matches the
# methodology behind the original expC_cache_decomposition_results.csv
# exactly. Run this independently of / after cold_sweep_full.ps1 -- both use
#the same temporary container name and port, so don't run them at the same
# time, but the order doesn't matter otherwise.

$cpuLevels = @("0.5", "1.0", "4.0")
$repeats = 3
$port = 8010
$containerName = "coldstart_retest"
$imageName = "implement-edge_unsandboxed"
$outFile = "cache_decomposition_clean_rerun.csv"

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

if (Test-Path $outFile) {
    Write-Host "Removing existing $outFile so this is a clean single-batch file" -ForegroundColor Yellow
    Remove-Item $outFile
}

"cpu,cache,trial,total_ms,model_load_ms" | Out-File -FilePath $outFile -Encoding utf8

docker volume create hf-cache | Out-Null

Write-Host "Priming hf-cache volume" -ForegroundColor Cyan
docker rm -f $containerName 2>$null | Out-Null
docker run -d --name $containerName --cpus="1.0" --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface -e NODE_ROLE=warm-prime $imageName | Out-Null
Wait-Healthy $port 90 | Out-Null
docker rm -f $containerName | Out-Null
Write-Host "Cache warmed.`n" -ForegroundColor Cyan

foreach ($cpu in $cpuLevels) {
    for ($i = 1; $i -le $repeats; $i++) {

        # Cold Start Test: download, no cache mounting
        docker rm -f $containerName 2>$null | Out-Null
        docker run -d --name $containerName --cpus=$cpu --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -e NODE_ROLE="cold-$cpu" $imageName | Out-Null
        $ok = Wait-Healthy $port 120
        if ($ok) {
            $resp = curl.exe -s http://localhost:$port/startup_timing | ConvertFrom-Json
            "$cpu,cold,$i,$($resp.total_cold_start_ms),$($resp.model_load_ms)" | Out-File -FilePath $outFile -Append -Encoding utf8
            Write-Host "cold  cpu=$cpu trial=$i -> $($resp.total_cold_start_ms) ms"
        } else {
            Write-Host "cold  cpu=$cpu trial=$i -> TIMEOUT" -ForegroundColor Red
        }
        docker rm -f $containerName | Out-Null

        # WARM trial: pre-warmed cache, offline mode,
        docker run -d --name $containerName --cpus=$cpu --memory="512m" --network implement_vcse-net -p ${port}:8000 -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface -e NODE_ROLE="warm-$cpu" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 $imageName | Out-Null
        $ok = Wait-Healthy $port 90

        if ($ok) {
            $resp = curl.exe -s http://localhost:$port/startup_timing | ConvertFrom-Json
            "$cpu,warm,$i,$($resp.total_cold_start_ms),$($resp.model_load_ms)" | Out-File -FilePath $outFile -Append -Encoding utf8
            Write-Host "warm  cpu=$cpu trial=$i -> $($resp.total_cold_start_ms) ms"
        } else {
            Write-Host "warm  cpu=$cpu trial=$i -> TIMEOUT" -ForegroundColor Red
        }
        docker rm -f $containerName | Out-Null
    }
}

Write-Host "`nDone. Saved to $outFile -- send this back too and I'll rebuild the cache-decomposition half of the report section." -ForegroundColor Green
