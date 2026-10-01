param(
    [string]$RunName = 'inertial_20261001_seed0',
    [int]$Minutes = 25,
    [ValidateSet('inertial','masked')]
    [string]$Experiment = 'inertial'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a directory basename.' }
$runDirectory = Join-Path $projectRoot "runs/$RunName"
$launchDirectory = Join-Path $projectRoot "runs/${RunName}_launch_$(Get-Date -Format yyyyMMdd_HHmmss)"
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) {
    throw 'Refusing to reuse an existing run or launch directory.'
}
$pythonPath = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$entry = 'new/nca_inertial_wind_tunnel/run_wind_tunnel.py'
$arguments = @('-u', $entry, '--device', 'cuda', '--arms', 'all', '--seed', '0',
    '--steps', '800', '--minutes', "$Minutes", '--out', "runs/$RunName")
if ($Experiment -eq 'masked') {
    $entry = 'new/masked_medium/run_masked.py'
    $arguments = @('-u', $entry, '--device', 'cuda', '--minutes', "$Minutes", '--out', "runs/$RunName")
}
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started = Get-Date
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt = [ordered]@{
    host = [Environment]::MachineName
    pid = $process.Id
    started_local = $started.ToString('o')
    started_utc = $started.ToUniversalTime().ToString('o')
    executable = $pythonPath
    arguments = $arguments
    working_directory = $projectRoot
    run_directory = $runDirectory
    gpu = (& nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)
    maximum_minutes = $Minutes
    experiment = $Experiment
    continuous_monitoring = $false
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $launchDirectory 'launch_receipt.json') -Encoding utf8
# A bounded dispatch check, not a training monitor. Confirm the child's manifest.
$verificationDeadline = (Get-Date).AddSeconds(12)
do {
    if ($process.WaitForExit(500)) {
        Get-Content -LiteralPath (Join-Path $launchDirectory 'stderr.log')
        throw "Training child exited during dispatch with code $($process.ExitCode)."
    }
    $manifestPath = Join-Path $runDirectory 'manifest.json'
    if (Test-Path -LiteralPath $manifestPath) { break }
} while ((Get-Date) -lt $verificationDeadline)
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw 'Child is running but startup manifest is not yet available; receipt retained, do not relaunch blindly.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.pid -ne $process.Id) { throw 'Launch PID does not match child manifest.' }
$receipt['dispatch_verified'] = $true
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $launchDirectory 'launch_receipt.json') -Encoding utf8
[pscustomobject]@{ pid=$process.Id; started=$receipt.started_local; gpu=$manifest.gpu;
    run=$runDirectory; receipt=(Join-Path $launchDirectory 'launch_receipt.json'); verified=$true } | ConvertTo-Json
