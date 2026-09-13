$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force artifacts | Out-Null
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'artifacts\readout.pt')) { throw 'Run train.ps1 first.' }
if (-not (Test-Path -LiteralPath 'simulation\dist\index.html')) { throw 'Build simulation first: cd simulation; npm run build' }
if (-not (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) {
    $digitProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','digitlab.server:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'artifacts\server.out.log' -RedirectStandardError 'artifacts\server.err.log' -PassThru
    $digitProcess.Id | Set-Content 'artifacts\digit-server.pid'
}
if (-not (Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue)) {
    $flyProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','http.server','5173','--bind','127.0.0.1','--directory','simulation/dist') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'artifacts\simulation.out.log' -RedirectStandardError 'artifacts\simulation.err.log' -PassThru
    $flyProcess.Id | Set-Content 'artifacts\fly-server.pid'
}
Write-Host 'Digit Lab: http://127.0.0.1:8000/'
Write-Host 'Fly simulation: http://127.0.0.1:5173/'
