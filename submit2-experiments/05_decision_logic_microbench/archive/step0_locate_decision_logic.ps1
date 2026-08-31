Get-ChildItem -Recurse -Include *.py | Select-String -Pattern "LogisticRegression|StandardScaler|joblib.load|pickle.load" | Group-Object Path | Select-Object Name

Write-Host "---"

Get-ChildItem -Recurse -Include *.py | Where-Object { $_.Name -match "benchmark|micro|decision|logic|classifier" } | Select-Object FullName

Write-Host "---"

Get-ChildItem -Recurse -Include *.pkl,*.joblib,*.json | Where-Object { $_.Name -match "scaler|classifier|model|threshold" } | Select-Object FullName, Length