param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Preflight
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a basename.' }
$runDirectory=Join-Path $projectRoot "runs/$RunName"
$launchDirectory=Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName,(Get-Date -Format yyyyMMdd_HHmmss))
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) { throw 'Output already exists.' }
$preflightDirectory=[IO.Path]::GetFullPath((Join-Path $projectRoot $Preflight))
if (-not $preflightDirectory.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Preflight outside project.' }
$status=Get-Content -LiteralPath (Join-Path $preflightDirectory 'status.json') -Raw | ConvertFrom-Json
$manifest=Get-Content -LiteralPath (Join-Path $preflightDirectory 'manifest.json') -Raw | ConvertFrom-Json
$aggregate=Get-Content -LiteralPath (Join-Path $preflightDirectory 'aggregate.json') -Raw | ConvertFrom-Json
if ($status.status -ne 'PREFLIGHT_PASSED' -or -not $aggregate.pass -or $manifest.protocol -ne 'seed4_dense_transition_v1' -or
    -not $manifest.preflight -or $manifest.updates -ne 3 -or $manifest.seed -ne 4 -or $manifest.runtime_limit_enforced -ne $false) {
    throw 'Bound preflight qualification failed.'
}
foreach ($entry in $manifest.source_sha256.PSObject.Properties) {
    $source=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if (-not $source.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside project.' }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Source changed: $($entry.Name)" }
}
$gpu=& nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect GPU.' }
$fields=$gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 3500) { throw 'Insufficient GPU headroom.' }
$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source
$arguments=@('-X','utf8','-u','-B','new/transition_100_200/run.py','--out',"runs/$RunName",'--preflight-dir',$Preflight)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started=Get-Date
$process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt=[ordered]@{host=[Environment]::MachineName;pid=$process.Id;started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o');executable=$pythonPath;arguments=$arguments;
    run_directory=$runDirectory;gpu=$gpu;preflight_directory=$preflightDirectory;
    maximum_seconds=$null;runtime_limit_enforced=$false;watchdog_enabled=$false;continuous_monitoring=$false;dispatch_verified=$false}
$receiptPath=Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$deadline=(Get-Date).AddSeconds(30)
$startup=$null
do {
    if ($process.WaitForExit(500)) { throw "Child exited, code=$($process.ExitCode); receipt retained." }
    $statusPath=Join-Path $runDirectory 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        try {$startup=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json} catch {continue}
        if ($startup.status -eq 'RUNNING' -and $startup.completed_updates -ge 1) {break}
        if ($startup.status -eq 'ERROR') {throw 'Child failed; no blind relaunch.'}
    }
} while ((Get-Date) -lt $deadline)
if ($null -eq $startup -or $startup.status -ne 'RUNNING' -or $startup.completed_updates -lt 1) {throw 'First update unverified; receipt retained.'}
$child=Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($child.pid -ne $process.Id -or $child.protocol -ne 'seed4_dense_transition_v1' -or $child.preflight -or
    $child.updates -ne 300 -or $child.seed -ne 4 -or $child.runtime_limit_enforced -ne $false -or $child.watchdog_enabled -ne $false) {throw 'Child manifest mismatch.'}
$receipt.dispatch_verified=$true
$receipt.verified_completed_updates=$startup.completed_updates
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$process.Id;verified=$true;run=$runDirectory;receipt=$receiptPath;gpu=$child.gpu;
    estimated_seconds=$aggregate.estimated_formal_seconds;maximum_seconds=$null;runtime_limit_enforced=$false} | ConvertTo-Json
