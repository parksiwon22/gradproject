param([switch]$Install)
$ErrorActionPreference='Stop'
$v17Root=Split-Path $PSScriptRoot -Parent
$v17Python=Join-Path (Split-Path $v17Root -Parent) '.venv\Scripts\pythonw.exe'
$v17Folder=Join-Path $v17Root 'artifacts\v17\scheduler'
if ($Install) {
 New-Item -ItemType Directory -Path $v17Folder -Force | Out-Null
 $v17Task=Get-ScheduledTask -TaskName 'GradprojectV15Candidates'
 $v17Args=($v17Task.Actions.Arguments -join ',')
 if ($v17Args -notin @('-m v15.monitor','-m v17.service')) { throw 'Unexpected task action; inspect before migration' }
 $v17Backup=Join-Path $v17Folder 'GradprojectV15Candidates_before.xml'
 if (-not (Test-Path -LiteralPath $v17Backup)) { Export-ScheduledTask -TaskName 'GradprojectV15Candidates' | Set-Content -LiteralPath $v17Backup -Encoding Unicode }
 $v17Action=New-ScheduledTaskAction -Execute $v17Python -Argument '-m v17.service' -WorkingDirectory $v17Root
 $v17Daily=New-ScheduledTaskTrigger -Daily -At '23:00'
 $v17Repeat=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(15)) -RepetitionInterval (New-TimeSpan -Minutes 15)
 $v17Settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
 Set-ScheduledTask -TaskName 'GradprojectV15Candidates' -Action $v17Action -Trigger @($v17Daily,$v17Repeat) -Settings $v17Settings | Select-Object TaskName,State
} else { Write-Output 'Use -Install after successful V17 service smoke; reuses V15 task, no duplicate seventh job. Original task XML retained privately.' }
