param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Qualification,
    [switch]$Verify
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') {throw 'RunName must be a basename.'}
$runDirectory=Join-Path $projectRoot "runs/$RunName"
$qualificationPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Qualification))
if (-not $qualificationPath.StartsWith($projectRoot+'\analyses\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Qualification outside analyses.'}
$checked=Get-Content -LiteralPath $qualificationPath -Raw | ConvertFrom-Json
if ($checked.status -ne 'PASS' -or $checked.protocol -ne 'continuous_execution_coverage_v1' -or
    $checked.runtime_limit_enforced -ne $false) {throw 'Qualification failed.'}
foreach ($entry in $checked.source_sha256.PSObject.Properties) {
    $source=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if (-not $source.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Source outside project.'}
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {throw "Binding changed: $($entry.Name)"}
}
if ($Verify) {
    $candidates=@(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'runs') -Directory -Filter "${RunName}_launch_*" |
        Where-Object {Test-Path -LiteralPath (Join-Path $_.FullName 'launch_receipt.json')})
    if ($candidates.Count -ne 1) {throw 'Require exactly one launch receipt for this run.'}
    $receiptPath=Join-Path $candidates[0].FullName 'launch_receipt.json'
    $receipt=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    $startup=Get-Content -LiteralPath (Join-Path $runDirectory 'status.json') -Raw | ConvertFrom-Json
    $child=Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
    if ($receipt.execution_mode -ne 'tool_owned_foreground' -or $receipt.run_directory -ne $runDirectory -or
        $receipt.qualification -ne $Qualification -or $startup.status -ne 'RUNNING' -or
        $startup.completed_updates -lt 1 -or $startup.pid -ne $child.pid -or
        $child.protocol -ne 'continuous_execution_coverage_v1' -or $child.expected_arms -ne 16 -or
        $child.runtime_limit_enforced -ne $false) {throw 'Worker/receipt mismatch or first update unverified.'}
    $process=Get-Process -Id $child.pid -ErrorAction Stop
    $launcher=Get-Process -Id $receipt.launcher_pid -ErrorAction Stop
    if ($process.HasExited -or $launcher.HasExited) {throw 'Worker or foreground launcher exited.'}
    $receipt.pid=$process.Id
    $receipt.dispatch_verified=$true
    $receipt.verified_completed_updates=$startup.completed_updates
    $receipt.verified_after_tool_yield=$true
    $receipt.verified_local=(Get-Date).ToString('o')
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    [pscustomobject]@{pid=$process.Id;launcher_pid=$launcher.Id;verified=$true;
        verified_updates=$startup.completed_updates;phase=$startup.phase;run=$runDirectory;
        receipt=$receiptPath;gpu=$child.gpu;expected_arms=16;runtime_limit_enforced=$false} | ConvertTo-Json
    exit 0
}
$launchDirectory=Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName,(Get-Date -Format yyyyMMdd_HHmmss))
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) {throw 'Output already exists.'}
$gpu=& nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect GPU.'}
$fields=$gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 4000) {throw 'Insufficient GPU headroom.'}
$drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($runDirectory))
if ($drive.AvailableFreeSpace -lt 3GB) {throw 'Insufficient disk headroom.'}
$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source
if (-not $pythonPath) {throw 'Python unavailable; activate historical Torch2.5.1 environment.'}
$arguments=@('-X','utf8','-u','-B','new/continuous_coverage/run.py','--out',"runs/$RunName",'--qualification',$Qualification)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started=Get-Date
$receipt=[ordered]@{host=[Environment]::MachineName;pid=$null;launcher_pid=$PID;started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o');executable=$pythonPath;arguments=$arguments;
    run_directory=$runDirectory;gpu=$gpu;qualification=$Qualification;expected_arms=16;
    expected_dense_records=208;maximum_seconds=$null;runtime_limit_enforced=$false;
    watchdog_enabled=$false;continuous_monitoring=$false;dispatch_verified=$false;
    execution_mode='tool_owned_foreground';verified_completed_updates=$null;
    verified_after_tool_yield=$false;verified_local=$null;worker_exit_code=$null}
$receiptPath=Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Output "Foreground launch receipt: $receiptPath"
# Keep this tool-owned shell alive for the worker's entire lifetime. A detached
# Start-Process child is reclaimed when this environment's launching shell exits.
# The tool returns a live exec session, not a completed launching command.
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath @arguments 1> (Join-Path $launchDirectory 'stdout.log') 2> (Join-Path $launchDirectory 'stderr.log')
    $workerExit=$LASTEXITCODE
} finally {
    Pop-Location
}
$receipt=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$receipt.worker_exit_code=$workerExit
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
if ($workerExit -ne 0) {throw "Worker exited with code $workerExit; artifacts retained."}
Write-Output "Worker completed; receipt: $receiptPath"
