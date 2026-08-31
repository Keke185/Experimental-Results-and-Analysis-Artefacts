docker rm -f temp_new_edge 2>$null | Out-Null
docker run -d --name temp_new_edge -p 8022:8000 --cpus=1.0 --memory=512m `
    --network implement_vcse-net `
    -v implement_artifact-store:/data -v hf-cache:/root/.cache/huggingface `
    -e NODE_ROLE=edge-measure vcse-edge-newlogic | Out-Null

Start-Sleep -Seconds 5

docker cp measure_new_logic_load.py temp_new_edge:/app/measure_new_logic_load.py
docker exec temp_new_edge python measure_new_logic_load.py

docker rm -f temp_new_edge | Out-Null
