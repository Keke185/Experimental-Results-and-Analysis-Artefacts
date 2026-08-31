function WaitHealthy($port, $maxTries) {
    for ($i = 0; $i -lt $maxTries; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$port/health" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

$cpuLevels = @("1.0", "4.0")

foreach ($cpu in $cpuLevels) {
    Write-Host "=== CPU=$cpu OLD logic ==="
    docker rm -f temp_old_edge 2>$null | Out-Null
    docker run -d --name temp_old_edge -p 8021:8000 --cpus=$cpu --memory=512m `
        --network implement_vcse-net `
        -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface `
        -e NODE_ROLE=edge-old-temp implement-edge_unsandboxed | Out-Null
    if (WaitHealthy 8021 60) {
        python decision_logic_e2e_client.py "cpu${cpu}_old" 8021 30
    } else {
        Write-Host "temp_old_edge did not become healthy"
    }
    docker rm -f temp_old_edge | Out-Null

    Write-Host "=== CPU=$cpu NEW logic ==="
    docker rm -f temp_new_edge 2>$null | Out-Null
    docker run -d --name temp_new_edge -p 8022:8000 --cpus=$cpu --memory=512m `
        --network implement_vcse-net `
        -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface `
        -e NODE_ROLE=edge-new-temp vcse-edge-newlogic | Out-Null
    if (WaitHealthy 8022 60) {
        python decision_logic_e2e_client.py "cpu${cpu}_new" 8022 30
    } else {
        Write-Host "temp_new_edge did not become healthy"
    }
    docker rm -f temp_new_edge | Out-Null
}

Write-Host "done"
