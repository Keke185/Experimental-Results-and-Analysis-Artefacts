cd C:\Users\HK_COCO\Desktop\MSc

Get-ChildItem -Recurse -Include *.py,*.ipynb -ErrorAction SilentlyContinue | Select-String -Pattern "LogisticRegression" -List | Select-Object Path

Write-Host "---pkl/joblib---"

Get-ChildItem -Recurse -Include *.pkl,*.joblib -ErrorAction SilentlyContinue | Select-Object FullName, Length

Write-Host "---9 feature refs---"

Get-ChildItem -Recurse -Include *.py,*.ipynb -ErrorAction SilentlyContinue | Select-String -Pattern "9.?feature|feature.?9|nine.?feature" -List | Select-Object Path
