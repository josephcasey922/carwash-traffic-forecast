# Run each evening with Windows Task Scheduler after actual counts have been exported.
# Replace path. This script only forecasts; NO retraining takes place.
# Requires an authorized weather API and a compatible license.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
& $Python asof.py forecast --traffic 'C:\data\traffic.csv' --model-dir 'asof_results' --model mlp --output 'forward_results'
if ($LASTEXITCODE -ne 0) { throw "Daily forecast command failed with exit code $LASTEXITCODE" }
