param([int]$TrainSize=10000,[int]$Epochs=25,[string]$Device='cpu')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
& .\.venv\Scripts\python.exe -u -m digitlab.train --train-size $TrainSize --epochs $Epochs --device $Device
if ($LASTEXITCODE -ne 0) { throw 'Training failed. See output above.' }
Write-Host 'Saved artifacts/readout.pt and artifacts/metrics.json. Restart Digit Lab to load new weights.'
