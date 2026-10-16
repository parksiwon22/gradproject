param([switch]$Install)
$ErrorActionPreference='Stop'
$v14Root=Split-Path $PSScriptRoot -Parent
$v14Python=Join-Path (Split-Path $v14Root -Parent) '.venv\Scripts\pythonw.exe'
if ($Install) {
 if (Get-ScheduledTask -TaskName 'GradprojectV14Candidates' -ErrorAction SilentlyContinue) { throw 'Inspect existing V14 task before replacement' }
 $v14Action=New-ScheduledTaskAction -Execute $v14Python -Argument '-m v14.forward' -WorkingDirectory $v14Root
 $v14Trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(15)) -RepetitionInterval (New-TimeSpan -Minutes 15)
 $v14Settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
 Register-ScheduledTask -TaskName 'GradprojectV14Candidates' -Action $v14Action -Trigger $v14Trigger -Settings $v14Settings -Description 'Immutable V14 candidate forecasts and frozen EV evaluations; published prices skip forecasts; no network recollection or model promotion' | Select-Object TaskName,State
} else { Write-Output 'Use -Install after checking models; original V11-V13 tasks are preserved.' }
