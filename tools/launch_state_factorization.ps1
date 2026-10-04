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
if ($checked.status -ne 'PASS' -or $checked.protocol -ne 'output_preserving_state_factorization_v1' -or $checked.training_updates -ne 0) {throw 'Qualification failed.'}
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
$arguments=@('-X','utf8','-u','-B','new/state_factorization/run.py','--out',"runs/$RunName",'--qualification',$Qualification)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started=Get-Date
$process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt=[ordered]@{host=[Environment]::MachineName;pid=$process.Id;started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o');executable=$pythonPath;arguments=$arguments;
    run_directory=$runDirectory;gpu=$gpu;qualification=$Qualification;training_updates=0;
    maximum_seconds=$null;runtime_limit_enforced=$false;watchdog_enabled=$false;
    continuous_monitoring=$false;dispatch_verified=$false}
$receiptPath=Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$startupDeadline=(Get-Date).AddSeconds(50)
$startup=$null
do {
    $statusPath=Join-Path $runDirectory 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        try {$startup=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json} catch {continue}
        if ($startup.status -eq 'COMPLETE' -or ($startup.status -eq 'RUNNING' -and $startup.completed_cells -ge 1)) {break}
        if ($startup.status -eq 'ERROR') {throw 'Child failed; receipt retained.'}
    }
    if ($process.WaitForExit(500)) {throw "Child exited, code=$($process.ExitCode); receipt retained."}
} while ((Get-Date) -lt $startupDeadline)
if ($null -eq $startup -or ($startup.status -ne 'COMPLETE' -and ($startup.status -ne 'RUNNING' -or $startup.completed_cells -lt 1))) {throw 'First actual intervention unverified; inspect receipt before retrying.'}
$child=Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($child.pid -ne $process.Id -or $child.protocol -ne 'output_preserving_state_factorization_v1' -or $child.training_updates -ne 0 -or $child.runtime_limit_enforced -ne $false) {throw 'Child manifest mismatch.'}
$receipt.dispatch_verified=$true
$receipt.verified_phase=$startup.phase
$receipt.verified_completed_cells=$startup.completed_cells
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$process.Id;verified=$true;run=$runDirectory;receipt=$receiptPath;
    gpu=$child.gpu;training_updates=0;runtime_limit_enforced=$false} | ConvertTo-Json
