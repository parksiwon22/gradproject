param([switch]$Install)
$ErrorActionPreference='Stop'
$v20Root=Split-Path $PSScriptRoot -Parent
if ($Install) {
 $v20Task=Get-ScheduledTask -TaskName 'GradprojectV12ForwardEV'
 if (($v20Task.Actions.Arguments -join ',') -notin @('-m v19.observer','-m v20.service')) {throw 'Unexpected collector; no modification'}
 $v20Folder=Join-Path $v20Root 'artifacts\v20\scheduler'
 New-Item -ItemType Directory -Path $v20Folder -Force | Out-Null
 $v20Backup=Join-Path $v20Folder 'collector_before.xml'
 if (-not (Test-Path -LiteralPath $v20Backup)) {Export-ScheduledTask -TaskName $v20Task.TaskName | Set-Content -LiteralPath $v20Backup -Encoding Unicode}
 $v20Python=Join-Path (Split-Path $v20Root -Parent) '.venv\Scripts\pythonw.exe'
 $v20Action=New-ScheduledTaskAction -Execute $v20Python -Argument '-m v20.service' -WorkingDirectory $v20Root
 Set-ScheduledTask -TaskName $v20Task.TaskName -Action $v20Action | Select-Object TaskName,State
 # Triggers/settings and the sole V19 publisher are deliberately retained.
} else {Write-Output 'Use -Install to add V20 audit after the existing V19 observer; no new issuer or trigger.'}
