$Container = "vcse_edge_sandboxed"

$CanaryMarker = "CANARY_MARKER_9f3e7a2b1c"

$CanaryFull = "CANARY_MARKER_9f3e7a2b1c_DO_NOT_PERSIST_THIS_STRING"

$Endpoint = "http://localhost:8002/match/check"

Write-Host "Step 1: Send canary request, check response body"

Write-Host "POST $Endpoint item_id=CANARY_TEST_01"

python -c "import requests, json; r = requests.post('$Endpoint', json={'item_id': 'CANARY_TEST_01', 'item_text': '$CanaryFull'}, timeout=15); print(r.status_code); print(json.dumps(r.json(), indent=2, ensure_ascii=False)); print('CANARY_IN_RESPONSE:', '$CanaryFull' in r.text)"

Write-Host "`n Step 2: Search container stdout / uvicorn access logs"

Write-Host "docker logs $Container 2>&1 | Select-String `"$CanaryMarker`""

docker logs $Container 2>&1 | Select-String $CanaryMarker

Write-Host "(No output above indicates that Select-String did not find a match - this is the expected "clean" result, not an error)"

Write-Host "`n Step 3: Search container filesystem (readable files)"

Write-Host "docker exec $Container grep -rI `"$CanaryMarker`" / --exclude-dir=proc --exclude-dir=sys"

docker exec $Container `grep -rI $CanaryMarker / --exclude-dir=proc --exclude-dir=sys`

`Write-Host "(No output means grep found no matches in any readable files, which is the expected "clean" result, not an error)"

`Write-Host "`n Audit complete"

`Write-Host "Passed criteria: CANARY_IN_RESPONSE was False in step 1, and there was no output in steps 2 and 3."

`Write-Host "The -I parameter in step 3 skips binary files (such as model weight files) because the audit targets observable plaintext leaks, not binary files."

`Write-Host "/proc and /sys were excluded because they are virtual filesystems unrelated to application-level data persistence and could otherwise cause the scan to stall or become disruptive."`