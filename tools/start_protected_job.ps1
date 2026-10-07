param(
    [Parameter(Mandatory=$true)][string]$JobName,
    [Parameter(Mandatory=$true)][string]$Script,
    [string[]]$ScriptArguments=@(),
    [string]$PythonExecutable
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($JobName -notmatch '^[A-Za-z0-9_-]{1,64}$') {throw 'JobName must be a short basename.'}
$scriptPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Script))
if (-not $scriptPath.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
    -not (Test-Path -LiteralPath $scriptPath -PathType Leaf) -or
    [IO.Path]::GetExtension($scriptPath) -ne '.py') {throw 'Python script must exist inside this project.'}
if (-not $PythonExecutable) {$PythonExecutable=(Get-Command python -CommandType Application | Select-Object -First 1).Source}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {throw 'Python executable unavailable.'}
$pwshPath=(Get-Command pwsh -CommandType Application | Select-Object -First 1).Source
$workerPath=Join-Path $PSScriptRoot 'protected_job_worker.ps1'
$jobId=[guid]::NewGuid().ToString('N').Substring(0,12)
$taskName="CodexResearch-$JobName-$jobId"
$jobDirectory=Join-Path $projectRoot "runs/${JobName}_guard_$jobId"
if (Test-Path -LiteralPath $jobDirectory) {throw 'Guard output already exists.'}
New-Item -ItemType Directory -Path $jobDirectory | Out-Null
$planPath=Join-Path $jobDirectory 'job_plan.json'
$plan=[ordered]@{schema='independent-on-demand-job-v1';task_name=$taskName;job_directory=$jobDirectory;
    working_directory=$projectRoot;executable=$PythonExecutable;
    arguments=@('-X','utf8','-u','-B',$scriptPath)+$ScriptArguments;
    created_local=(Get-Date).ToString('o');runtime_limit_enforced=$false;
    recurring=$false;automatic_restart=$false;idle_sleep_prevention=$true}
$plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding utf8
$actionArguments='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -Plan "{1}"' -f $workerPath,$planPath
$action=New-ScheduledTaskAction -Execute $pwshPath -Argument $actionArguments -WorkingDirectory $projectRoot
$userIdentity=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal=New-ScheduledTaskPrincipal -UserId $userIdentity -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -DisallowHardTerminate
# No trigger is installed. Start once on demand; no password or persistent power-plan change.
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
try {Start-ScheduledTask -TaskName $taskName}
catch {Unregister-ScheduledTask -TaskName $taskName -Confirm:$false;throw}
[pscustomobject]@{task_name=$taskName;job_directory=$jobDirectory;plan=$planPath;
    execution_mode='windows_task_scheduler';runtime_limit_enforced=$false;recurring=$false} | ConvertTo-Json
