# Rebuild candidates from last ~90 days of issuance dates after a major process change.
# Confirm API license and replace CSV path before running.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
& $Python asof.py fit --traffic 'C:\data\traffic.csv' --origin-step 7 --validation-origins 2 --models baseline,mlp --output 'asof_results'
if ($LASTEXITCODE -ne 0) { throw "Full refit command failed with exit code $LASTEXITCODE" }
