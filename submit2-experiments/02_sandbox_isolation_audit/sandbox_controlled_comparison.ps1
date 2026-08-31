# Controlled sandboxed-vs-unsandboxed latency comparison, same CPU quota.

$cpuLevels = @("0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0")
$sandboxedContainer = "vcse_edge_sandboxed"
$sandboxedPort = 8002
$unsandboxedImage = "implement-edge_unsandboxed"
$unsandboxedPort = 8011
$outFile = "sandbox_comparison_clean_rerun.csv"

# Obtain the raw CPU limit of the sandbox container for later restoration.
$originalNanoCpus = docker inspect $sandboxedContainer --format="{{.HostConfig.NanoCpus}}"
Write-Host "Current $sandboxedContainer NanoCpus setting: $originalNanoCpus" -ForegroundColor Cyan
if ($originalNanoCpus -and $originalNanoCpus -ne "0") {
    $originalCores = [double]$originalNanoCpus / 1000000000
} else {
    $originalCores = $null
}

if (Test-Path $outFile) {
    Write-Host "Removing existing $outFile so this is a clean single-batch file..." -ForegroundColor Yellow
    Remove-Item $outFile
}

function Wait-Healthy($port, $timeoutSec = 90) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 2
    }
    return $false
}

foreach ($cpu in $cpuLevels) {
    Write-Host "`n=== cpu=$cpu ===" -ForegroundColor Yellow

    #sandboxed side
    Write-Host "-- sandboxed ($sandboxedContainer, port $sandboxedPort) --" -ForegroundColor Magenta
    docker update --cpus=$cpu $sandboxedContainer | Out-Null
    Start-Sleep -Seconds 3
    python sandbox_comparison_client.py "sandboxed_$cpu" $sandboxedPort $outFile

    #unsandboxed side
    Write-Host "-- unsandboxed (temp container, port $unsandboxedPort) --" -ForegroundColor Magenta
    $tempName = "edge_unsandboxed_ctrl_$($cpu -replace '\.','_')"
    docker rm -f $tempName 2>$null | Out-Null
    docker run -d --name $tempName --cpus=$cpu --memory="512m" `
        --network implement_vcse-net -p ${unsandboxedPort}:8000 `
        -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface `
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NODE_ROLE="sandbox-ctrl-$cpu" `
        $unsandboxedImage | Out-Null

    if (Wait-Healthy -port $unsandboxedPort -timeoutSec 90) {
        python sandbox_comparison_client.py "unsandboxed_$cpu" $unsandboxedPort $outFile
    } else {
        Write-Host "[WARN] $tempName did not become healthy within timeout -- skipping cpu=$cpu unsandboxed measurement" -ForegroundColor Red
    }

    docker rm -f $tempName | Out-Null
}


#Restore the original CPU limits of the sandbox container
if ($originalCores) {
    docker update --cpus=$originalCores $sandboxedContainer | Out-Null
    Write-Host "`nRestored $sandboxedContainer to --cpus=$originalCores" -ForegroundColor Cyan
} else {
    docker update --cpus=0 $sandboxedContainer | Out-Null
    Write-Host "`nRestored $sandboxedContainer to unlimited" -ForegroundColor Cyan
}

Write-Host "`nDone. Saved to $outFile -- send this back and I'll compute the clean, same-CPU-level sandboxing cost curve and update Section 3." -ForegroundColor Green
