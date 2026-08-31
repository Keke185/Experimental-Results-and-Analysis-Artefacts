$lines = Get-Content edge_service\main.py
$total = $lines.Count

function DumpRange($start, $end) {
    Write-Host "===== lines $start-$end ====="
    for ($i = $start; $i -le [Math]::Min($end, $total); $i++) {
        Write-Host "$($i): $($lines[$i-1])"
    }
}

DumpRange 55 115
DumpRange 148 220
