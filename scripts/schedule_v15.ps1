param([switch]$Install)
$ErrorActionPreference='Stop'
$v15Root=Split-Path $PSScriptRoot -Parent
$v15Python=Join-Path (Split-Path $v15Root -Parent) '.venv\Scripts\pythonw.exe'
if ($Install) {
 if (Get-ScheduledTask -TaskName 'GradprojectV15Candidates' -ErrorAction SilentlyContinue) { throw 'Inspect existing V15 task before replacement' }
 $v15Action=New-ScheduledTaskAction -Execute $v15Python -Argument '-m v15.monitor' -WorkingDirectory $v15Root
 $v15Trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(15)) -RepetitionInterval (New-TimeSpan -Minutes 15)
 $v15Settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
 Register-ScheduledTask -TaskName 'GradprojectV15Candidates' -Action $v15Action -Trigger $v15Trigger -Settings $v15Settings -Description 'Immutable V15 candidate forecasts and frozen EV evaluations; published prices skip forecasts; no network recollection or model promotion' | Select-Object TaskName,State
} else { Write-Output 'Use -Install after checking models; original V11-V14 tasks are preserved.' }
