$lines = Get-Content edge_service\main.py
for ($i = 1; $i -le 54; $i++) {
    Write-Host "$($i): $($lines[$i-1])"
}
Write-Host "-----"
for ($i = 115; $i -le 147; $i++) {
    Write-Host "$($i): $($lines[$i-1])"
}
