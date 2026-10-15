param([switch]$Install, [switch]$Remove)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path (Split-Path $projectRoot -Parent) '.venv\Scripts\pythonw.exe'
$taskName = 'GradprojectV11Local'
if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    exit
}
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Research Python environment missing' }
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument '-m v11.pipeline' -WorkingDirectory $projectRoot
$hourly = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(5)) -RepetitionInterval (New-TimeSpan -Hours 1)
$daily = New-ScheduledTaskTrigger -Daily -At '23:05'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
if ($Install) {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($hourly,$daily) -Settings $settings -Description 'Local KPX price collection and immutable forecast reconciliation; no model retraining or public data upload' -Force | Select-Object TaskName,State
} else {
    Write-Output 'Prepared local schedule: hourly collection and 23:05 KST forecast. Run with -Install to register; -Remove to unregister.'
}
