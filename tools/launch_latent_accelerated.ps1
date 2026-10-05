param(
    [Parameter(Mandatory=$true)][string]$OriginalRun,
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Qualification
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
foreach ($name in @($OriginalRun,$RunName)) {if ($name -notmatch '^[a-zA-Z0-9_-]+$') {throw 'Run names must be basenames.'}}
if ($OriginalRun -eq $RunName) {throw 'Migration requires a separate output.'}
$original=Join-Path $projectRoot "runs/$OriginalRun"
$output=Join-Path $projectRoot "runs/$RunName"
if (Test-Path -LiteralPath $output) {throw 'New output already exists.'}
$qualificationPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Qualification))
if (-not $qualificationPath.StartsWith($projectRoot+'\analyses\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Qualification outside analyses.'}
$qualified=Get-Content -LiteralPath $qualificationPath -Raw | ConvertFrom-Json
if ($qualified.status -ne 'PASS' -or $qualified.protocol -ne 'native_latent_graph_exact300_v1') {throw 'Full exact qualification required.'}
foreach ($field in @('source_sha256','references_sha256')) {
    foreach ($entry in $qualified.$field.PSObject.Properties) {
        $file=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
        if (-not $file.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Binding outside project.'}
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {throw "Changed binding: $($entry.Name)"}
    }
}
$originalManifest=Get-Content -LiteralPath (Join-Path $original 'manifest.json') -Raw | ConvertFrom-Json
if ($originalManifest.protocol -ne 'native_latent_width_v1' -or $originalManifest.expected_arms -ne 96) {throw 'Original suite mismatch.'}
if ((Get-FileHash -LiteralPath (Join-Path $original 'plans.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $qualified.original_plan_sha256) {throw 'Plan changed.'}
$worker=Get-Process -Id $originalManifest.pid -ErrorAction Stop
$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source
if ($worker.ProcessName -ne 'python' -or $worker.Path -ne $pythonPath) {throw 'Original process identity mismatch.'}
$startedUtc=[DateTimeOffset]::Parse($originalManifest.started_utc).UtcDateTime
if ([Math]::Abs(($worker.StartTime.ToUniversalTime()-$startedUtc).TotalSeconds) -gt 30) {throw 'PID was reused or start binding mismatched.'}
$gpu=& nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) {throw 'GPU unavailable.'}
$gpuFields=$gpu.Split(',')
if (([int]$gpuFields[1]-[int]$gpuFields[2]) -lt 2000) {throw 'Insufficient GPU headroom.'}
$launchDirectory=Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName,(Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$before=@(Get-Content -LiteralPath (Join-Path $original 'perarm.json') -Raw | ConvertFrom-Json)
$boundaryStart=Get-Date
$boundaryDeadline=$boundaryStart.AddSeconds(180)
$boundaryStatus=$null
do {
    if ($worker.HasExited) {throw 'Original exited before handover; inspect its records.'}
    $completed=@(Get-Content -LiteralPath (Join-Path $original 'perarm.json') -Raw | ConvertFrom-Json)
    $boundaryStatus=Get-Content -LiteralPath (Join-Path $original 'status.json') -Raw | ConvertFrom-Json
    if ($completed.Count -gt $before.Count -or $boundaryStatus.phase -eq 'arm_complete') {break}
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $boundaryDeadline)
# All imported rows remain fully completed even if the boundary wait expires.
# A partially executed arm is retained in the original and restarted unchanged.
$worker.Refresh()
if ($worker.HasExited -or $worker.ProcessName -ne 'python' -or $worker.Path -ne $pythonPath) {throw 'Worker identity changed.'}
Stop-Process -Id $worker.Id -ErrorAction Stop
if (-not $worker.WaitForExit(5000)) {throw 'Original worker stop unverified.'}
$completed=@(Get-Content -LiteralPath (Join-Path $original 'perarm.json') -Raw | ConvertFrom-Json)
$lastStatus=Get-Content -LiteralPath (Join-Path $original 'status.json') -Raw | ConvertFrom-Json
$completedKeys=@{}
foreach ($row in $completed) {$completedKeys[('{0}:{1}' -f $row.block,$row.arm)]=$true}
$partial=@()
foreach ($blockDirectory in Get-ChildItem -LiteralPath $original -Directory -Filter 'block*') {
    foreach ($armDirectory in Get-ChildItem -LiteralPath $blockDirectory.FullName -Directory) {
        $blockNumber=[int]($blockDirectory.Name.Substring(5))
        if (-not $completedKeys.ContainsKey(('{0}:{1}' -f $blockNumber,$armDirectory.Name))) {
            $partial += [ordered]@{block=$blockNumber;arm=$armDirectory.Name;path=('{0}/{1}' -f $blockDirectory.Name,$armDirectory.Name);retained=$true;aggregate_excluded=$true;restart_from_update=0}
        }
    }
}
$handover=[ordered]@{original_run="runs/$OriginalRun";original_pid=$worker.Id;original_worker_stopped=$true;
    stopped_utc=(Get-Date).ToUniversalTime().ToString('o');reason='User authorized local CUDA Graph acceleration';
    original_run_modified=$false;imported_completed_arms=$completed.Count;
    completed_rows_sha256=(Get-FileHash -LiteralPath (Join-Path $original 'perarm.json') -Algorithm SHA256).Hash.ToLowerInvariant();
    waited_seconds=((Get-Date)-$boundaryStart).TotalSeconds;last_status=$lastStatus;partial_attempts_excluded=$partial}
$handoverPath=Join-Path $launchDirectory 'handover_receipt.json'
$handover | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $handoverPath -Encoding utf8
$handoverRelative=[IO.Path]::GetRelativePath($projectRoot,$handoverPath).Replace('\','/')
$arguments=@('-X','utf8','-u','-B','new/latent_runtime/accelerated.py','--original',"runs/$OriginalRun",'--out',"runs/$RunName",'--qualification',$Qualification,'--handover',$handoverRelative)
$started=Get-Date
$child=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt=[ordered]@{pid=$child.Id;host=[Environment]::MachineName;started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o');executable=$pythonPath;arguments=$arguments;
    run_directory=$output;original_run_directory=$original;qualification=$Qualification;handover=$handoverRelative;
    imported_completed_arms=$completed.Count;expected_arms=96;runtime_limit_enforced=$false;continuous_monitoring=$false;dispatch_verified=$false}
$receiptPath=Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$startupDeadline=(Get-Date).AddSeconds(55)
$startup=$null
do {
    if ($child.WaitForExit(500)) {throw "Accelerated worker exited, code=$($child.ExitCode); original and receipt retained."}
    $statusPath=Join-Path $output 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        try {$startup=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json} catch {continue}
        if ($startup.status -eq 'RUNNING' -and $startup.phase -eq 'training' -and $startup.completed_updates -ge 1) {break}
        if ($startup.status -eq 'ERROR') {throw 'Accelerated child failed; inspect retained receipts.'}
    }
} while ((Get-Date) -lt $startupDeadline)
if ($null -eq $startup -or $startup.status -ne 'RUNNING' -or $startup.phase -ne 'training' -or $startup.completed_updates -lt 1) {throw 'First accelerated optimizer update unverified.'}
$manifest=Get-Content -LiteralPath (Join-Path $output 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.pid -ne $child.Id -or $manifest.execution_amendment -ne 'local_cuda_graph_k8_v1' -or $manifest.expected_arms -ne 96) {throw 'Child manifest mismatch.'}
$receipt.dispatch_verified=$true
$receipt.verified_completed_updates=$startup.completed_updates
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$child.Id;verified=$true;run=$output;receipt=$receiptPath;gpu=$manifest.gpu;
    imported_arms=$completed.Count;expected_arms=96;runtime_limit_enforced=$false} | ConvertTo-Json
