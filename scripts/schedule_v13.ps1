param([switch]$Install)
$ErrorActionPreference = 'Stop'
$v13Root = Split-Path $PSScriptRoot -Parent
$v13Python = Join-Path (Split-Path $v13Root -Parent) '.venv\Scripts\pythonw.exe'
$v13Action = New-ScheduledTaskAction -Execute $v13Python -Argument '-m v13.service' -WorkingDirectory $v13Root
$v13Trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(5)) -RepetitionInterval (New-TimeSpan -Minutes 15)
$v13Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
if ($Install) {
    if (Get-ScheduledTask -TaskName 'GradprojectV13Audit' -ErrorAction SilentlyContinue) { throw 'V13 task already exists; inspect before replacing' }
    Register-ScheduledTask -TaskName 'GradprojectV13Audit' -Action $v13Action -Trigger $v13Trigger -Settings $v13Settings -Description 'Additive V13 receipt audit, uncertainty annotations and EV evaluation; no duplicate network collection, training or model promotion' | Select-Object TaskName,State
} else { Write-Output 'Use -Install to register V13 audit-only job; existing V11/V12 tasks are preserved.' }
