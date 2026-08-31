Get-ChildItem -Recurse -Include Dockerfile,docker-compose.yml,docker-compose.yaml,*.dockerfile -ErrorAction SilentlyContinue | Select-Object FullName

Write-Host "---edge_service dir listing---"

Get-ChildItem edge_service | Select-Object Name, Length

Write-Host "---current image for sandboxed container---"

docker inspect vcse_edge_sandboxed --format "{{.Config.Image}}"
