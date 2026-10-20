param([switch]$Install)
$ErrorActionPreference='Stop'
$v20Root=Split-Path $PSScriptRoot -Parent
if ($Install) {
 $v20Folder=Join-Path $v20Root 'artifacts\v20\scheduler'
 New-Item -ItemType Directory -Path $v20Folder -Force | Out-Null
 # Keep the previously working Task Scheduler process mode. The wrapper
 # persists stdout/stderr, so it no longer depends on a visible console.
 $v20Python=Join-Path (Split-Path $v20Root -Parent) '.venv\Scripts\pythonw.exe'
 $specs=@(
  @{Name='GradprojectV12ForwardEV';Expected=@('-m v19.observer','-m v20.service');Arguments='-u -m v20.task_runner --module v20.service'},
  @{Name='GradprojectV19Publish';Expected=@('-m v19.publisher --warm');Arguments='-u -m v20.task_runner --module v19.publisher -- --warm'}
 )
 if (-not (Test-Path -LiteralPath $v20Python)) { throw "Python executable not found: $v20Python" }
 # Validate both tasks and export timestamped full XML backups before any update.
 $prepared=@()
 foreach ($spec in $specs) {
  $task=Get-ScheduledTask -TaskName $spec.Name -ErrorAction Stop
  $args=($task.Actions.Arguments -join ',')
  if ($args -notin $spec.Expected -and $args -notlike '*v20.task_runner*') { throw "Unexpected action for $($spec.Name): $args" }
  if ($task.Principal.UserId -notmatch '박시원|S-1-5-21-2416485759-2394923871-2357612515-1001') { throw "Unexpected principal for $($spec.Name)" }
  $stamp=Get-Date -Format 'yyyyMMddTHHmmssfff'
  $backup=Join-Path $v20Folder ($spec.Name+'_before_'+$stamp+'.xml')
  Export-ScheduledTask -TaskName $spec.Name -ErrorAction Stop | Set-Content -LiteralPath $backup -Encoding Unicode
  $prepared+=@{Spec=$spec;Task=$task;Backup=$backup}
 }
 # Enable event evidence only after both XML backups are safely written.
 wevtutil.exe sl Microsoft-Windows-TaskScheduler/Operational /e:true
 $v20EventsEnabled=($LASTEXITCODE -eq 0)
 if (-not $v20EventsEnabled) { Write-Warning 'Operational log enable was denied; per-run logs will still be written by the Python wrapper.' }
 foreach ($entry in $prepared) {
  $action=New-ScheduledTaskAction -Execute $v20Python -Argument $entry.Spec.Arguments -WorkingDirectory $v20Root
  $settings=$entry.Task.Settings
  $settings.WakeToRun=$true
  $settings.StartWhenAvailable=$true
  if ($settings.IdleSettings) { $settings.IdleSettings.StopOnIdleEnd=$false }
  Set-ScheduledTask -TaskName $entry.Spec.Name -Action $action -Settings $settings -ErrorAction Stop | Select-Object TaskName,State
 }
 # Global AC/DC sleep timers are intentionally left unchanged.
 Write-Output ('Updated the two existing tasks only. XML backups are in artifacts/v20/scheduler. WakeToRun is enabled; Operational log enabled='+$v20EventsEnabled+'; global AC/DC sleep policy was not changed.')
} else {Write-Output 'Use -Install after reviewing existing task XML: backs up and updates only the existing observer and single publisher. Uses pythonw.exe with redacted file start/end logs; enables WakeToRun and Operational history; no extra triggers or tasks.'}
