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
if ($checked.status -ne 'PASS' -or $checked.protocol -ne 'semantic_write_source_causal_v1' -or $checked.training -ne $false) {throw 'Qualification failed.'}
foreach ($entry in $checked.source_sha256.PSObject.Properties) {
    $source=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if (-not $source.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Source outside project.'}
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {throw "Binding changed: $($entry.Name)"}
}
$gpu=& nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect GPU.'}
$fields=$gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 1500) {throw 'Insufficient GPU headroom.'}
$drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($runDirectory))
if ($drive.AvailableFreeSpace -lt 2GB) {throw 'Insufficient disk headroom.'}
$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source
if (-not $pythonPath) {throw 'Historical Python unavailable; activate the Torch2.5.1 environment.'}
$arguments=@('-X','utf8','-u','-B','new/semantic_write_audit/run.py','--out',"runs/$RunName",'--qualification',$Qualification)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started=Get-Date
$process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt=[ordered]@{host=[Environment]::MachineName;pid=$process.Id;started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o');executable=$pythonPath;arguments=$arguments;
    run_directory=$runDirectory;gpu=$gpu;qualification=$Qualification;expected_units=396;
    training=$false;optimizer_updates=0;maximum_seconds=$null;runtime_limit_enforced=$false;
    watchdog_enabled=$false;continuous_monitoring=$false;dispatch_verified=$false}
$receiptPath=Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$startupDeadline=(Get-Date).AddSeconds(50)
$startup=$null
do {
    if ($process.WaitForExit(500)) {throw "Child exited, code=$($process.ExitCode); receipt retained."}
    $statusPath=Join-Path $runDirectory 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        try {$startup=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json} catch {continue}
        if ($startup.status -eq 'RUNNING' -and $startup.phase -eq 'inference') {break}
        if ($startup.status -eq 'ERROR') {throw 'Child failed; receipt retained.'}
    }
} while ((Get-Date) -lt $startupDeadline)
if ($null -eq $startup -or $startup.status -ne 'RUNNING' -or $startup.phase -ne 'inference') {throw 'Inference dispatch unverified; inspect receipt before retrying.'}
$child=Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($child.pid -ne $process.Id -or $child.protocol -ne 'semantic_write_source_causal_v1' -or $child.expected_units -ne 396 -or $child.training -ne $false) {throw 'Child manifest mismatch.'}
$receipt.dispatch_verified=$true
$receipt.verified_phase=$startup.phase
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$process.Id;verified=$true;run=$runDirectory;receipt=$receiptPath;
    gpu=$child.gpu;expected_units=396;training=$false;runtime_limit_enforced=$false} | ConvertTo-Json
