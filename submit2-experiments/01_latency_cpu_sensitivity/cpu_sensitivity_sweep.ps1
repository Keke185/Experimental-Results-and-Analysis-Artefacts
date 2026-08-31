# Perform multiple `--cpus` limit scans on the sandboxed edge container,
# After each restarted container recovers, run
# the cpu_sensitivity_test.py test (and append the tagged results to
# cpu_sensitivity_results.csv). Memory is always fixed at 512MB, therefore
# CPU is the only variable.

$cpuLevels = @("0.1", "0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "4.0", "6.0")
$containerName = "vcse_edge_sandboxed"
$networkName = "implement_vcse-net"
$volumeName = "implement_artifact-store"
$imageName = "implement-edge_unsandboxed"
$hostPort = 8002

foreach ($cpu in $cpuLevels) {
    Write-Host "=== Testing --cpus=$cpu ===" -ForegroundColor Cyan

    docker rm -f $containerName 2>$null | Out-Null

    $runArgs = @(
        "run", "-d",
        "--name", $containerName,
        "--cpus=$cpu",
        "--memory=512m",
        "--network", $networkName,
        "-p", "${hostPort}:8000",
        "-v", "${volumeName}:/data:ro",
        "-e", "NODE_ROLE=edge-sandboxed",
        $imageName
    )
    docker @runArgs | Out-Null

    Write-Host "  waiting for service to become healthy"
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:$hostPort/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                $ready = $true
                break
            }
        }
        catch {
            # not ready yet, keep waiting
        }
    }

    if (-not $ready) {
        Write-Host "  ! service did not become healthy within timeout, skipping this level" -ForegroundColor Red
        continue
    }

    Write-Host "  service ready, running test"
    $env:CPU_LABEL = $cpu
    python cpu_sensitivity_test.py
    Write-Host ""
}

Write-Host "=== Sweep complete. See cpu_sensitivity_results.csv for all raw rows. ===" -ForegroundColor Green
