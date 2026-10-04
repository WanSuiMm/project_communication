param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Qualification
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') {throw 'RunName must be a basename.'}
$runDirectory=Join-Path $projectRoot "runs/$RunName"
$launchDirectory=Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName,(Get-Date -Format yyyyMMdd_HHmmss))
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) {throw 'Output already exists.'}
$qualificationPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Qualification))
if (-not $qualificationPath.StartsWith($projectRoot+'\analyses\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Qualification outside analyses.'}
$checked=Get-Content -LiteralPath $qualificationPath -Raw | ConvertFrom-Json
if ($checked.status -ne 'PASS' -or $checked.protocol -ne 'binary_carrier_causal_compression_v1' -or $checked.recurrent_training_updates -ne 0) {throw 'Qualification failed.'}
foreach ($entry in $checked.source_sha256.PSObject.Properties) {
    $source=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if (-not $source.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Source outside project.'}
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {throw "Source binding changed: $($entry.Name)"}
}
$gpu=& nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect GPU.'}
$fields=$gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 3500) {throw 'Insufficient GPU headroom.'}
$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source
$arguments=@('-X','utf8','-u','-B','new/binary_carrier/run.py','--out',"runs/$RunName",'--qualification',$Qualification)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started=Get-Date
$process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt=[ordered]@{host=[Environment]::MachineName;pid=$process.Id;started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o');executable=$pythonPath;arguments=$arguments;
    run_directory=$runDirectory;gpu=$gpu;qualification=$Qualification;recurrent_training_updates=0;decoder_fits=1;
    maximum_seconds=$null;runtime_limit_enforced=$false;watchdog_enabled=$false;continuous_monitoring=$false;dispatch_verified=$false}
$receiptPath=Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$startupDeadline=(Get-Date).AddSeconds(50)
$executionPath=Join-Path $runDirectory 'startup_execution.json'
do {
    $statusPath=Join-Path $runDirectory 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $startup=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($startup.status -eq 'ERROR') {throw 'Child failed; receipt retained.'}
    }
    if (Test-Path -LiteralPath $executionPath) {break}
    if ($process.WaitForExit(500)) {throw "Child exited, code=$($process.ExitCode); receipt retained."}
} while ((Get-Date) -lt $startupDeadline)
if (-not (Test-Path -LiteralPath $executionPath)) {throw 'Actual computation unverified; inspect receipt before retrying. Child not terminated.'}
$execution=Get-Content -LiteralPath $executionPath -Raw | ConvertFrom-Json
$child=Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($child.pid -ne $process.Id -or $execution.pid -ne $process.Id -or $execution.finite_capture -ne $true -or $child.protocol -ne 'binary_carrier_causal_compression_v1' -or $child.recurrent_training_updates -ne 0 -or $child.runtime_limit_enforced -ne $false) {throw 'Child computation/manifest mismatch.'}
$capturePath=Join-Path $runDirectory $execution.capture_path
if ((Get-FileHash -LiteralPath $capturePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $execution.capture_sha256) {throw 'Executed capture binding changed.'}
$receipt.dispatch_verified=$true; $receipt.verified_phase='calibration_capture_complete'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$process.Id;verified=$true;run=$runDirectory;receipt=$receiptPath;
    gpu=$child.gpu;recurrent_training_updates=0;decoder_fits=1;runtime_limit_enforced=$false} | ConvertTo-Json
