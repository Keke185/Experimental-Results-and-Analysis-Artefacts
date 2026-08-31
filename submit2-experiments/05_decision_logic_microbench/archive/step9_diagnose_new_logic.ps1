docker rm -f temp_new_edge 2>$null | Out-Null
docker run -d --name temp_new_edge -p 8022:8000 --cpus=1.0 --memory=512m `
    --network implement_vcse-net `
    -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface `
    -e NODE_ROLE=edge-new-temp vcse-edge-newlogic | Out-Null

Start-Sleep -Seconds 15

Write-Host "---docker ps---"
docker ps -a --filter "name=temp_new_edge"

Write-Host "---logs---"
docker logs temp_new_edge
