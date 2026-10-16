param([switch]$Install)
$ErrorActionPreference='Stop'
$v18Root=Split-Path $PSScriptRoot -Parent
$v18Python=Join-Path (Split-Path $v18Root -Parent) '.venv\Scripts\pythonw.exe'
$v18Folder=Join-Path $v18Root 'artifacts\v18\scheduler'
if ($Install) {
 New-Item -ItemType Directory -Path $v18Folder -Force | Out-Null
 $v18Changes=@(
  @{Name='GradprojectV12ForwardEV';Old='-m v12.pipeline';New='-m v18.service'},
  @{Name='GradprojectV14Candidates';Old='-m v14.forward';New='-m v18.legacy v14'},
  @{Name='GradprojectV15Candidates';Old='-m v17.service';New='-m v18.legacy v17'}
 )
 # Validate all tasks and back up before changing any action.
 foreach ($v18Change in $v18Changes) {
  $v18Task=Get-ScheduledTask -TaskName $v18Change.Name
  if (($v18Task.Actions.Arguments -join ',') -notin @($v18Change.Old,$v18Change.New)) { throw 'Unexpected task action; migration stopped' }
  $v18Backup=Join-Path $v18Folder ($v18Change.Name+'_before.xml')
  if (-not (Test-Path -LiteralPath $v18Backup)) { Export-ScheduledTask -TaskName $v18Change.Name | Set-Content -LiteralPath $v18Backup -Encoding Unicode }
 }
 foreach ($v18Change in $v18Changes) {
  $v18Action=New-ScheduledTaskAction -Execute $v18Python -Argument $v18Change.New -WorkingDirectory $v18Root
  if ($v18Change.Name -eq 'GradprojectV12ForwardEV') {
   $v18Daily=New-ScheduledTaskTrigger -Daily -At '23:00'
   $v18Retry=New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(15)) -RepetitionInterval (New-TimeSpan -Minutes 15)
   Set-ScheduledTask -TaskName $v18Change.Name -Action $v18Action -Trigger @($v18Daily,$v18Retry) | Select-Object TaskName,State
  } else { Set-ScheduledTask -TaskName $v18Change.Name -Action $v18Action | Select-Object TaskName,State }
 }
} else { Write-Output 'Install after service verification: one V18 issuer, prior candidate jobs reconcile only, original task XML retained. V11/V13/V16 unchanged.' }
