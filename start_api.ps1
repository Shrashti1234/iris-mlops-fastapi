$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
docker build -t iris-api .
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
docker run -d --name iris-api -p 8000:8000 iris-api
if ($LASTEXITCODE -ne 0) { throw 'Docker start failed; check whether iris-api is already running' }
python scripts/smoke_test.py
