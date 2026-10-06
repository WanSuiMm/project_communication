param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Qualification,
    [Parameter(Mandatory=$true)][int]$OwnerSessionId
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') {throw 'RunName must be a basename.'}
$runDirectory=Join-Path $projectRoot "runs/$RunName"
$qualificationPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Qualification))
if (-not $qualificationPath.StartsWith($projectRoot+'\analyses\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Qualification outside analyses.'}
$checked=Get-Content -LiteralPath $qualificationPath -Raw | ConvertFrom-Json
if ($checked.status -ne 'PASS' -or $checked.protocol -ne 'continuous_execution_coverage_v1') {throw 'Qualification failed.'}
foreach ($entry in $checked.source_sha256.PSObject.Properties) {
    $source=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if (-not $source.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Source outside project.'}
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {throw "Binding changed: $($entry.Name)"}
}
$candidates=@(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'runs') -Directory -Filter "${RunName}_launch_*" |
    Where-Object {Test-Path -LiteralPath (Join-Path $_.FullName 'launch_receipt.json')})
if ($candidates.Count -ne 1) {throw 'Require exactly one launch receipt.'}
$receiptPath=Join-Path $candidates[0].FullName 'launch_receipt.json'
$receipt=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$manifest=Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($receipt.execution_mode -ne 'tool_owned_foreground' -or $receipt.run_directory -ne $runDirectory -or
    $receipt.qualification -ne $Qualification -or $manifest.protocol -ne $checked.protocol -or
    $manifest.expected_arms -ne 16 -or $manifest.runtime_limit_enforced -ne $false -or
    $manifest.qualification_sha256 -ne (Get-FileHash -LiteralPath $qualificationPath -Algorithm SHA256).Hash.ToLowerInvariant()) {
    throw 'Manifest/receipt mismatch.'
}
$statusPath=Join-Path $runDirectory 'status.json'
function Read-Status {Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json}
$baseline=Read-Status
if ($baseline.status -ne 'RUNNING' -or $baseline.pid -ne $manifest.pid -or
    ($baseline.completed_updates -lt 1 -and $baseline.completed_arms -lt 1)) {throw 'Real update unverified.'}
# Processes launched in a different isolated tool context cannot be enumerated
# by Get-Process here, even while status demonstrably advances. Verify durable
# worker progress instead. This is one bounded dispatch check, never a monitor.
$deadline=(Get-Date).AddSeconds(30)
$latest=$baseline
do {
    Start-Sleep -Milliseconds 750
    $latest=Read-Status
    if ($latest.status -ne 'RUNNING' -or $latest.pid -ne $manifest.pid) {throw 'Worker stopped during dispatch verification.'}
    if ($latest.elapsed_seconds -gt $baseline.elapsed_seconds -and
        ($latest.completed_arms -gt $baseline.completed_arms -or
         $latest.completed_dense_records -gt $baseline.completed_dense_records -or
         $latest.completed_updates -gt $baseline.completed_updates)) {break}
} while ((Get-Date) -lt $deadline)
if ($latest.elapsed_seconds -le $baseline.elapsed_seconds -or
    ($latest.completed_arms -le $baseline.completed_arms -and
     $latest.completed_dense_records -le $baseline.completed_dense_records -and
     $latest.completed_updates -le $baseline.completed_updates)) {throw 'No advancing progress observed.'}
$receipt.pid=$manifest.pid
$receipt.dispatch_verified=$true
$receipt.verified_completed_updates=$latest.completed_updates
$receipt.verified_after_tool_yield=$true
$receipt.verified_local=(Get-Date).ToString('o')
$receipt | Add-Member -NotePropertyName owner_exec_session_id -NotePropertyValue $OwnerSessionId -Force
$receipt | Add-Member -NotePropertyName verification_method -NotePropertyValue 'advancing_worker_progress_across_tool_calls' -Force
$receipt | Add-Member -NotePropertyName verification_snapshots -NotePropertyValue @($baseline,$latest) -Force
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$manifest.pid;session_id=$OwnerSessionId;verified=$true;
    before_update=$baseline.completed_updates;after_update=$latest.completed_updates;
    dense_records=$latest.completed_dense_records;receipt=$receiptPath} | ConvertTo-Json
