Write-Host "=== edge_service/Dockerfile ==="
Get-Content edge_service\Dockerfile

Write-Host "=== edge_service dir ==="
Get-ChildItem edge_service -Recurse | Select-Object FullName

Write-Host "=== docker-compose.yml (edge sections) ==="
Select-String -Path docker-compose.yml -Pattern "edge|image:|build:|context:|dockerfile:|ports:|volumes:|environment:" -Context 0,1
