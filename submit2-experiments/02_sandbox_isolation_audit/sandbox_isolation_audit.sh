#!/usr/bin/env bash

# Sandbox isolation auditing for the "sandbox edge" configuration

set -uo pipefail

SANDBOXED_CONTAINER="vcse_edge_sandboxed"

UNSANDBOXED_CONTAINER="$(docker ps --format '{{.Names}}' | grep -i edge_unsandboxed | head -n1)"
if [ -z "${UNSANDBOXED_CONTAINER}" ]; then
  echo " Could not auto-detect the edge_unsandboxed container name from 'docker ps'."
  echo "    Set UNSANDBOXED_CONTAINER manually at the top of this script if step 1b/2b fail."
  UNSANDBOXED_CONTAINER="edge_unsandboxed"
fi

echo "Sandboxed container:   ${SANDBOXED_CONTAINER}"
echo "Unsandboxed container: ${UNSANDBOXED_CONTAINER}"
echo

echo "=== 1a. Resource limits configured on SANDBOXED container ==="
docker inspect --format='NanoCpus={{.HostConfig.NanoCpus}}  Memory(bytes)={{.HostConfig.Memory}}' \
  "${SANDBOXED_CONTAINER}" 2>/dev/null || echo "  (container not found -- run run_sandboxed_edge.sh first)"
echo "  expected: NanoCpus=500000000 (0.5 CPU), Memory(bytes)=268435456 (256 MB)"
echo

echo "=== 1b. Resource limits configured on UNSANDBOXED container (contrast) ==="
docker inspect --format='NanoCpus={{.HostConfig.NanoCpus}}  Memory(bytes)={{.HostConfig.Memory}}' \
  "${UNSANDBOXED_CONTAINER}" 2>/dev/null || echo "  (container not found)"
echo "  expected: NanoCpus=0  Memory(bytes)=0   (both mean 'unlimited')"
echo

echo "=== 1c. CPU/memory usage under a short stress load (SANDBOXED) ==="
echo "    Running a 5s CPU-bound loop inside the sandboxed container..."
docker exec "${SANDBOXED_CONTAINER}" python -c "
import time
end = time.time() + 5
x = 0
while time.time() < end:
    x += 1
print('stress loop finished, iterations =', x)
" 2>/dev/null &
STRESS_PID=$!
sleep 1
echo "    docker stats snapshot while under load (CPU% should sit near/at the 0.5-core cap):"
docker stats --no-stream "${SANDBOXED_CONTAINER}" 2>/dev/null
wait "${STRESS_PID}" 2>/dev/null
echo

echo "=== 2a. Network isolation: outbound request from SANDBOXED container (expect BLOCKED) ==="
docker exec "${SANDBOXED_CONTAINER}" python -c "
import urllib.request
try:
    urllib.request.urlopen('https://www.google.com', timeout=3)
    print('UNEXPECTED: outbound request succeeded -- network isolation NOT effective')
except Exception as e:
    print('OK:outbound request blocked as expected ->', type(e).__name__, str(e)[:120])
" 2>/dev/null || echo "  (container not found or python not reachable in it)"
echo

echo "=== 2b. Network isolation: outbound request from UNSANDBOXED container (expect it WORKS, for contrast) ==="
docker exec "${UNSANDBOXED_CONTAINER}" python -c "
import urllib.request
try:
    r = urllib.request.urlopen('https://www.google.com', timeout=3)
    print('OK (as expected for unsandboxed): outbound request succeeded, status =', r.status)
except Exception as e:
    print('unexpected: outbound request blocked ->', type(e).__name__, str(e)[:120])
" 2>/dev/null || echo "  (container not found or python not reachable in it)"
echo

echo "=== 2c. Inbound still works: host -> SANDBOXED container's published port ==="
python3 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://localhost:8002/health', timeout=3)
    print('OK: /health reachable from host, status =', r.status)
except Exception as e:
    print('FAILED: /health not reachable from host ->', type(e).__name__, e)
"
echo

echo "=== 3. Exposed surface inside SANDBOXED container (only :8000/uvicorn expected) ==="
docker exec "${SANDBOXED_CONTAINER}" python -c "
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
    print('(neither ss nor netstat installed in this minimal image -- that itself is a small ')
    print(' positive data point for a minimal attack surface; note it as such in the audit.)')
" 2>/dev/null || echo "  (container not found)"

echo
echo "=== Audit complete. Record the outputs above (especially 1a/1b and 2a/2b side-by-side) ==="
echo "    in the dissertation's sandbox-isolation audit results."
