Write-Host "===== Docker daemon status =====" -ForegroundColor Cyan
docker version
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[!] Docker daemon is not reachable. Open Docker Desktop, wait for it to fully start, then re-run this script." -ForegroundColor Red
    exit 1
}

Write-Host "`n===== vcse_edge_sandboxed container status =====" -ForegroundColor Cyan
docker ps -a --filter "name=vcse_edge_sandboxed" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"

Write-Host "`n===== Current CPU/memory config (if it exists) =====" -ForegroundColor Cyan
docker inspect vcse_edge_sandboxed --format="NanoCpus={{.HostConfig.NanoCpus}}  Memory={{.HostConfig.Memory}}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Container 'vcse_edge_sandboxed' not found -- see note below." -ForegroundColor Red
}

Write-Host "`n===== Health endpoint (only meaningful if container is Up) =====" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8002/health" -TimeoutSec 5 -UseBasicParsing
    Write-Host "Health check: $($r.StatusCode) $($r.StatusDescription)" -ForegroundColor Green
} catch {
    Write-Host "Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n===== Supporting volumes =====" -ForegroundColor Cyan
docker volume ls --filter "name=implement_artifact-store"
docker volume ls --filter "name=hf-cache"

Write-Host "`n===== Supporting network =====" -ForegroundColor Cyan
docker network ls --filter "name=implement_vcse-net"

Write-Host "`n===== Python environment =====" -ForegroundColor Cyan
python --version
pip show requests pandas 2>$null | Select-String "Name|Version"
