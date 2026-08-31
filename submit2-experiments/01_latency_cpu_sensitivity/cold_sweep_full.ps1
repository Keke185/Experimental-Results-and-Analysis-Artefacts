

$cpuLevels = @("0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0")
$repeats = 2
$port = 8010
$containerName = "coldstart_retest"
$imageName = "implement-edge_unsandboxed"
$outFile = "cold_sweep_clean_rerun.csv"

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
"cpu,trial,total_ms,model_load_ms" | Out-File -FilePath $outFile -Encoding utf8

Write-Host "Cold-start sweep: 10 levels x $repeats repeats, cold cache (real download) each time" -ForegroundColor Cyan

foreach ($cpu in $cpuLevels) {
    for ($i = 1; $i -le $repeats; $i++) {
        docker rm -f $containerName 2>$null | Out-Null
        docker run -d --name $containerName --cpus=$cpu --memory="512m" `
            --network implement_vcse-net -p ${port}:8000 `
            -v implement_artifact-store:/data -e NODE_ROLE="cold-$cpu" `
            $imageName | Out-Null

        $ok = Wait-Healthy $port 120
        if ($ok) {
            $resp = curl.exe -s http://localhost:$port/startup_timing | ConvertFrom-Json
            "$cpu,$i,$($resp.total_cold_start_ms),$($resp.model_load_ms)" | Out-File -FilePath $outFile -Append -Encoding utf8
            Write-Host "cold  cpu=$cpu trial=$i -> $($resp.total_cold_start_ms) ms"
        } else {
            Write-Host "cold  cpu=$cpu trial=$i -> TIMEOUT (container never became healthy)" -ForegroundColor Red
        }
        docker rm -f $containerName | Out-Null
    }
}

Write-Host "`nDone. Saved to $outFile -- send this back and I'll rebuild the cold-start-sweep half of the report section from it." -ForegroundColor Green
