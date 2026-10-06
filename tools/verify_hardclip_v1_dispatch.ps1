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
if ($checked.status -ne 'PASS' -or $checked.protocol -ne 'hardclip_v1_fixed_tail_v1') {throw 'Qualification failed.'}
$candidates=@(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'runs') -Directory -Filter "${RunName}_launch_*" |
    Where-Object {Test-Path -LiteralPath (Join-Path $_.FullName 'launch_receipt.json')})
if ($candidates.Count -ne 1) {throw 'Require exactly one launch receipt.'}
$receiptPath=Join-Path $candidates[0].FullName 'launch_receipt.json'
$receipt=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if ($receipt.execution_mode -ne 'tool_owned_foreground' -or $receipt.run_directory -ne $runDirectory -or
    $receipt.qualification -ne $Qualification) {throw 'Receipt mismatch.'}
$statusPath=Join-Path $runDirectory 'status.json'
$manifestPath=Join-Path $runDirectory 'manifest.json'
$deadline=(Get-Date).AddSeconds(50)
$baseline=$null
$latest=$null
# One bounded check after tool yield. No ongoing monitor is created.
do {
    if (Test-Path -LiteralPath $statusPath) {
        $latest=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($latest.status -eq 'ERROR') {throw 'Worker failed during dispatch.'}
        if ($null -eq $baseline) {$baseline=$latest}
        $checkpoint=Join-Path $runDirectory 'block00/neural/checkpoints/u025.pt'
        if ($latest.status -eq 'RUNNING' -and $latest.completed_updates -ge 25 -and
            (Test-Path -LiteralPath $checkpoint) -and $latest.elapsed_seconds -gt $baseline.elapsed_seconds) {break}
    }
    Start-Sleep -Milliseconds 750
} while ((Get-Date) -lt $deadline)
if ($null -eq $latest -or $latest.status -ne 'RUNNING' -or $latest.completed_updates -lt 25 -or
    -not (Test-Path -LiteralPath $checkpoint) -or $latest.elapsed_seconds -le $baseline.elapsed_seconds) {throw 'Real post-yield progress unverified within bounded check.'}
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.protocol -ne $checked.protocol -or $manifest.expected_arms -ne 16 -or
    $manifest.expected_dense_records -ne 208 -or $manifest.pid -ne $latest.pid -or
    $manifest.runtime_limit_enforced -ne $false -or
    $manifest.qualification_sha256 -ne (Get-FileHash -LiteralPath $qualificationPath -Algorithm SHA256).Hash.ToLowerInvariant() -or
    $manifest.calibration_binding.summary_sha256 -ne $checked.calibration_binding.summary_sha256) {throw 'Manifest/qualification mismatch.'}
$receipt.pid=$manifest.pid
$receipt.dispatch_verified=$true
$receipt.verified_completed_updates=$latest.completed_updates
$receipt.verified_after_tool_yield=$true
$receipt | Add-Member -NotePropertyName verified_local -NotePropertyValue (Get-Date).ToString('o') -Force
$receipt | Add-Member -NotePropertyName owner_exec_session_id -NotePropertyValue $OwnerSessionId -Force
$receipt | Add-Member -NotePropertyName verification_method -NotePropertyValue 'advancing_worker_progress_and_u025_checkpoint_after_tool_yield' -Force
$receipt | Add-Member -NotePropertyName verification_snapshots -NotePropertyValue @($baseline,$latest) -Force
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{pid=$manifest.pid;session_id=$OwnerSessionId;verified=$true;
    completed_updates=$latest.completed_updates;dense_records=$latest.completed_dense_records;
    checkpoint=$checkpoint;receipt=$receiptPath} | ConvertTo-Json
