param([switch]$Install)
$ErrorActionPreference = 'Stop'
$v12Root = Split-Path $PSScriptRoot -Parent
$v12Python = Join-Path (Split-Path $v12Root -Parent) '.venv\Scripts\pythonw.exe'
$v12Action = New-ScheduledTaskAction -Execute $v12Python -Argument '-m v12.pipeline' -WorkingDirectory $v12Root
$v12Trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(3)) -RepetitionInterval (New-TimeSpan -Minutes 15)
$v12Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
if ($Install) {
    Register-ScheduledTask -TaskName 'GradprojectV12ForwardEV' -Action $v12Action -Trigger $v12Trigger -Settings $v12Settings -Description 'Public price receipt observation and immutable forward EV research; V11 collector remains separate; no automatic model promotion or upload' -Force | Select-Object TaskName,State
} else {
    Write-Output 'Run with -Install to add the V12 local15-minute observation job; existing V11 task is preserved.'
}
