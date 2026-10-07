param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$planPath=[IO.Path]::GetFullPath($Plan)
if (-not $planPath.StartsWith($projectRoot+'\runs\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Plan outside runs.'}
$job=Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
$jobDirectory=[IO.Path]::GetFullPath($job.job_directory)
if ($jobDirectory -ne [IO.Path]::GetDirectoryName($planPath) -or $job.working_directory -ne $projectRoot) {throw 'Invalid job root.'}
$receiptPath=Join-Path $jobDirectory 'launch_receipt.json'
$lock=$null
$child=$null
$awake=$false
$result=[ordered]@{status='STARTING';execution_mode='windows_task_scheduler';worker_pid=$PID;
    child_pid=$null;task_name=$job.task_name;started_local=(Get-Date).ToString('o');
    updated_local=$null;finished_local=$null;executable=$job.executable;arguments=$job.arguments;
    worker_exit_code=$null;error=$null;runtime_limit_enforced=$false;
    idle_sleep_request_active=$false;idle_sleep_request_cleared=$false;heartbeat_count=0}

function Write-Receipt {
    $result.updated_local=(Get-Date).ToString('o')
    $temp="$receiptPath.$PID.tmp"
    [IO.File]::WriteAllText($temp,($result | ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temp,$receiptPath,$true)
}

function Convert-WindowsArgument([string]$Value) {
    # Standard CommandLineToArgvW escaping, including trailing backslashes.
    $builder=[Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashes=0
    foreach($char in $Value.ToCharArray()) {
        if($char -eq '\') {$slashes++;continue}
        if($char -eq '"') {
            [void]$builder.Append(('\'*(2*$slashes+1)))
            [void]$builder.Append('"')
        } else {
            [void]$builder.Append(('\'*$slashes))
            [void]$builder.Append($char)
        }
        $slashes=0
    }
    [void]$builder.Append(('\'*(2*$slashes)))
    [void]$builder.Append('"')
    return $builder.ToString()
}

try {
    $lock=[IO.File]::Open((Join-Path $jobDirectory 'worker.lock'),[IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ResearchAwake {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    $previous=[ResearchAwake]::SetThreadExecutionState([uint32]2147483649) # CONTINUOUS | SYSTEM_REQUIRED
    if($previous -eq 0) {throw 'Cannot establish temporary idle-sleep protection.'}
    $awake=$true
    $result.idle_sleep_request_active=$true
    $argumentLine=(@($job.arguments | ForEach-Object {Convert-WindowsArgument ([string]$_)}) -join ' ')
    $child=Start-Process -FilePath $job.executable -ArgumentList $argumentLine -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $jobDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $jobDirectory 'stderr.log')
    $result.child_pid=$child.Id
    $result.status='RUNNING'
    Write-Receipt
    while(-not $child.WaitForExit(1000)) {
        $result.heartbeat_count++
        Write-Receipt
    }
    # A second wait refreshes redirected handles and exit code.
    $child.WaitForExit()
    $result.worker_exit_code=$child.ExitCode
    $result.status=if($child.ExitCode -eq 0){'COMPLETE'}else{'ERROR'}
} catch {
    $result.status='ERROR'
    $result.error=$_.Exception.Message
    if($null -ne $child -and -not $child.HasExited) {$child.Kill();$child.WaitForExit()}
} finally {
    if($awake) {
        $cleared=[ResearchAwake]::SetThreadExecutionState([uint32]2147483648)
        $result.idle_sleep_request_cleared=($cleared -ne 0)
        $result.idle_sleep_request_active=$false
    }
    $result.finished_local=(Get-Date).ToString('o')
    Write-Receipt
    if($lock){$lock.Dispose()}
}
if($result.status -eq 'ERROR'){exit 1}
