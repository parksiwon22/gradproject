param([switch]$Install)
$ErrorActionPreference='Stop'
$v16Root=Split-Path $PSScriptRoot -Parent
$v16Python=Join-Path (Split-Path $v16Root -Parent) '.venv\Scripts\pythonw.exe'
if ($Install) {
 if (Get-ScheduledTask -TaskName 'GradprojectV16Audit' -ErrorAction SilentlyContinue) { throw 'Inspect existing V16 task before replacement' }
 $v16Action=New-ScheduledTaskAction -Execute $v16Python -Argument '-m v16.monitor' -WorkingDirectory $v16Root
 $v16Trigger=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(15)) -RepetitionInterval (New-TimeSpan -Minutes 15)
 $v16Settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
 Register-ScheduledTask -TaskName 'GradprojectV16Audit' -Action $v16Action -Trigger $v16Trigger -Settings $v16Settings -Description 'Read-only inherited forward status aggregation to V16 only; no duplicate forecast issuance, API collection or model promotion' | Select-Object TaskName,State
} else { Write-Output 'Use -Install after checking models; original V11-V15 tasks are preserved.' }
