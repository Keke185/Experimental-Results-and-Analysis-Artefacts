New-Item -ItemType Directory -Force -Path edge_service_new_logic | Out-Null
Copy-Item edge_service\requirements.txt edge_service_new_logic\requirements.txt -Force
Copy-Item main_new_logic.py edge_service_new_logic\main.py -Force
Copy-Item Dockerfile_new_logic edge_service_new_logic\Dockerfile -Force
Copy-Item fit_and_save_pipeline.py edge_service_new_logic\fit_and_save_pipeline.py -Force

Push-Location edge_service_new_logic
python fit_and_save_pipeline.py
docker build -t vcse-edge-newlogic .
Pop-Location

Write-Host "---network/volume check---"
docker network ls
docker volume ls
