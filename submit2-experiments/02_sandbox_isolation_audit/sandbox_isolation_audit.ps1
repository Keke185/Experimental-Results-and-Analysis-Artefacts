# Sandbox-isolation audit for the "sandboxed edge" configuration


$SandboxedContainer = "vcse_edge_sandboxed"
$UnsandboxedContainer = (docker ps --format "{{.Names}}" | Select-String "edge_unsandboxed" | Select-Object -First 1).Line
if (-not $UnsandboxedContainer) {
    Write-Host "Could not auto-detect the edge_unsandboxed container name from 'docker ps'." -ForegroundColor Yellow
    Write-Host "    Edit `$UnsandboxedContainer at the top of this script if steps 1b/2b fail."
    $UnsandboxedContainer = "implement-edge_unsandboxed-1"
}

Write-Host "Sandboxed container:   $SandboxedContainer"
Write-Host "Unsandboxed container: $UnsandboxedContainer"
Write-Host ""

Write-Host "=== 1a. Resource limits configured on SANDBOXED container ===" -ForegroundColor Cyan
docker inspect --format='NanoCpus={{.HostConfig.NanoCpus}}  Memory(bytes)={{.HostConfig.Memory}}' $SandboxedContainer
Write-Host "  expected: NanoCpus > 0 and Memory(bytes) > 0 (whatever --cpus/--memory you set)"
Write-Host ""

Write-Host "=== 1b. Resource limits configured on UNSANDBOXED container (contrast) ===" -ForegroundColor Cyan
docker inspect --format='NanoCpus={{.HostConfig.NanoCpus}}  Memory(bytes)={{.HostConfig.Memory}}' $UnsandboxedContainer
Write-Host "  expected: NanoCpus=0  Memory(bytes)=0   (both mean 'unlimited')"
Write-Host ""

Write-Host "=== 1c. CPU/memory usage under a short stress load (SANDBOXED) ===" -ForegroundColor Cyan
Write-Host "    Launching a 5s CPU-bound loop inside the sandboxed container (detached)..."


docker exec -d $SandboxedContainer python -c "import time`nt = time.time() + 5`nx = 0`nwhile time.time() < t: x += 1"
Start-Sleep -Seconds 2
Write-Host "    docker stats snapshot while under load (CPU% should sit near/at your --cpus cap):"
docker stats --no-stream $SandboxedContainer
Start-Sleep -Seconds 3
Write-Host ""

Write-Host "=== 2a. Network isolation: outbound request from SANDBOXED container (expect BLOCKED) ===" -ForegroundColor Cyan
docker exec $SandboxedContainer python -c "
import urllib.request
try:
    urllib.request.urlopen('https://www.google.com', timeout=3)
    print('UNEXPECTED: outbound request succeeded -- network isolation NOT effective')
except Exception as e:
    print('OK: outbound request blocked as expected ->', type(e).__name__, str(e)[:120])
"
Write-Host ""

Write-Host "=== 2b. Network isolation: outbound request from UNSANDBOXED container (expect it WORKS, for contrast) ===" -ForegroundColor Cyan
docker exec $UnsandboxedContainer python -c "
import urllib.request
try:
    r = urllib.request.urlopen('https://www.google.com', timeout=3)
    print('OK (as expected for unsandboxed): outbound request succeeded, status =', r.status)
except Exception as e:
    print('unexpected: outbound request blocked ->', type(e).__name__, str(e)[:120])
"
Write-Host ""

Write-Host "=== 2c. Inbound still works: host -> SANDBOXED container's published port ===" -ForegroundColor Cyan
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8002/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    Write-Host "OK: /health reachable from host, status = $($resp.StatusCode)"
}
catch {
    Write-Host "FAILED: /health not reachable from host -> $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "=== 3. Exposed surface inside SANDBOXED container (only :8000/uvicorn expected) ===" -ForegroundColor Cyan
docker exec $SandboxedContainer python -c "
import subprocess
for cmd in (['ss', '-tulnp'], ['netstat', '-tulnp']):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print('command:', ' '.join(cmd))
        print(out.stdout or out.stderr)
        break
    except FileNotFoundError:
        continue
else:
    print('(neither ss nor netstat installed in this minimal image -- itself a small')
    print(' positive data point for a minimal attack surface; note it as such in the audit.)')
"

Write-Host ""
Write-Host "=== Audit complete. Record the outputs above (especially 1a/1b and 2a/2b side-by-side) ===" -ForegroundColor Green
Write-Host "    in the dissertation's sandbox-isolation audit results."
