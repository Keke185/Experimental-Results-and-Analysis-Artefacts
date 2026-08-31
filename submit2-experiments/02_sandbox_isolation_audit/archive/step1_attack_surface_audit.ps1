
$sandboxed = "vcse_edge_sandboxed"
$unsandboxedImage = "implement-edge_unsandboxed"

Write-Host "`n===== 0. Image identity =====" -ForegroundColor Cyan
Write-Host "-- $sandboxed is running from image: --" -ForegroundColor Yellow
docker inspect $sandboxed --format="{{.Config.Image}}"
Write-Host "-- Image ID (for exact match comparison): --" -ForegroundColor Yellow
docker inspect $sandboxed --format="{{.Image}}"
Write-Host "-- Locally available image tagged '$unsandboxedImage': --" -ForegroundColor Yellow
docker images $unsandboxedImage --format "{{.Repository}}:{{.Tag}}  ID={{.ID}}"

Write-Host "`n===== C1. Network-probing tools (present/absent) =====" -ForegroundColor Cyan
docker exec $sandboxed sh -c "for t in ss netstat nc curl wget nmap; do if command -v \$t >/dev/null 2>&1; then echo \"\$t: PRESENT\"; else echo \"\$t: absent\"; fi; done"

Write-Host "`n===== C2. Exposed ports =====" -ForegroundColor Cyan
Write-Host "-- Declared in image config: --" -ForegroundColor Yellow
docker inspect $sandboxed --format="{{.Config.ExposedPorts}}"
Write-Host "-- Actually published to host: --" -ForegroundColor Yellow
docker port $sandboxed

Write-Host "`n===== C3. Running user, capabilities, privilege mode =====" -ForegroundColor Cyan
Write-Host "-- Process owner inside container: --" -ForegroundColor Yellow
docker exec $sandboxed whoami
docker exec $sandboxed id
Write-Host "-- Docker-level security config: --" -ForegroundColor Yellow
docker inspect $sandboxed --format="Privileged={{.HostConfig.Privileged}}  ReadonlyRootfs={{.HostConfig.ReadonlyRootfs}}  CapAdd={{.HostConfig.CapAdd}}  CapDrop={{.HostConfig.CapDrop}}"
Write-Host "-- Effective kernel capabilities of PID 1 (raw bitmask, CapEff line): --" -ForegroundColor Yellow
docker exec $sandboxed sh -c "cat /proc/1/status | grep -i cap"

Write-Host "`n===== C4. Dependency inventory =====" -ForegroundColor Cyan
Write-Host "-- Python packages (pip): --" -ForegroundColor Yellow
docker exec $sandboxed pip list
Write-Host "-- OS package count (Debian/apt-based slim image): --" -ForegroundColor Yellow
docker exec $sandboxed sh -c "dpkg -l 2>/dev/null | wc -l || echo 'dpkg not available in this image'"

