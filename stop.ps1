$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython=(Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
foreach ($record in @(@{File='artifacts\digit-server.pid';Marker='digitlab.server:app'},@{File='artifacts\fly-server.pid';Marker='http.server 5173'})) {
    if (Test-Path -LiteralPath $record.File) {
        $taskProcessId=[int](Get-Content -LiteralPath $record.File)
        $taskProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$taskProcessId"
        if ($taskProcess -and $taskProcess.ExecutablePath -eq $taskPython -and $taskProcess.CommandLine.Contains($record.Marker)) {
            # Windows venv Python may launch a base-interpreter child. Stop only
            # this verified project's process tree, keeping unrelated servers safe.
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$taskProcessId" | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
            Stop-Process -Id $taskProcessId -ErrorAction SilentlyContinue
            Write-Host "Stopped $($record.Marker)"
        }
    }
}
