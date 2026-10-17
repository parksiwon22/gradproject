param([switch]$Install)
$ErrorActionPreference='Stop'
$v19Root=Split-Path $PSScriptRoot -Parent
$v19Python=Join-Path (Split-Path $v19Root -Parent) '.venv\Scripts\pythonw.exe'
$v19Folder=Join-Path $v19Root 'artifacts\v19\scheduler'
if ($Install) {
 New-Item -ItemType Directory -Path $v19Folder -Force | Out-Null
 $v19Task=Get-ScheduledTask -TaskName 'GradprojectV12ForwardEV'
 if (($v19Task.Actions.Arguments -join ',') -notin @('-m v18.service','-m v19.observer')) { throw 'Unexpected prior action' }
 $v19Backup=Join-Path $v19Folder 'GradprojectV12ForwardEV_before.xml'
 if (-not (Test-Path -LiteralPath $v19Backup)) { Export-ScheduledTask -TaskName 'GradprojectV12ForwardEV' | Set-Content -LiteralPath $v19Backup -Encoding Unicode }
 $v19Observer=New-ScheduledTaskAction -Execute $v19Python -Argument '-m v19.observer' -WorkingDirectory $v19Root
 $v19Repeat=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(15)) -RepetitionInterval (New-TimeSpan -Minutes 15)
 $v19Early=New-ScheduledTaskTrigger -Daily -At '22:50'
 $v19Fresh=New-ScheduledTaskTrigger -Daily -At '22:58'
 Set-ScheduledTask -TaskName 'GradprojectV12ForwardEV' -Action $v19Observer -Trigger @($v19Repeat,$v19Early,$v19Fresh) | Select-Object TaskName,State
 $v19PublisherName='GradprojectV19Publish'
 $v19Existing=Get-ScheduledTask -TaskName $v19PublisherName -ErrorAction SilentlyContinue
 if ($v19Existing -and ($v19Existing.Actions.Arguments -join ',') -ne '-m v19.publisher --warm') { throw 'Unexpected publisher; no replacement' }
 $v19Warm=New-ScheduledTaskTrigger -Daily -At '22:59:30'
 # Fallback cold start; IgnoreNew prevents overlap with the warm process.
 $v19Fallback=New-ScheduledTaskTrigger -Daily -At '23:00'
 $v19LateRetry=New-ScheduledTaskTrigger -Daily -At '23:02'
 $v19Settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
 $v19Action=New-ScheduledTaskAction -Execute $v19Python -Argument '-m v19.publisher --warm' -WorkingDirectory $v19Root
 Register-ScheduledTask -TaskName $v19PublisherName -Action $v19Action -Trigger @($v19Warm,$v19Fallback,$v19LateRetry) -Settings $v19Settings -Description 'Only V19 issuer: local precomputed packets; no network, inference or training at23h; skip public or unverified targets' -Force | Select-Object TaskName,State
} else { Write-Output 'Install after observer and publisher verification. Observer prepares at22:50/22:58; separate single warm publisher at22:59:30/23:00. Original task XML retained.' }
