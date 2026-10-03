param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Preflight
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a directory basename.' }
$runDirectory = Join-Path $projectRoot "runs/$RunName"
$launchDirectory = Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName, (Get-Date -Format yyyyMMdd_HHmmss))
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) { throw 'Output already exists.' }
$preflightDirectory = [IO.Path]::GetFullPath((Join-Path $projectRoot $Preflight))
if (-not $preflightDirectory.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Preflight must be inside the project directory.'
}
$checked = Get-Content -LiteralPath (Join-Path $preflightDirectory 'status.json') -Raw | ConvertFrom-Json
if ($checked.status -ne 'PREFLIGHT_PASSED') { throw 'Passed preflight required.' }
$manifestBefore = Get-Content -LiteralPath (Join-Path $preflightDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($manifestBefore.protocol -ne 'seed4_independent_followup_v1' -or -not $manifestBefore.training -or
    -not $manifestBefore.preflight -or $manifestBefore.updates_per_arm -ne 3 -or
    $manifestBefore.maximum_seconds -ne 180 -or $manifestBefore.expected_formal_arms -ne 13 -or
    (@($manifestBefore.arms.experiment) -join ',') -ne 'control,A,B') {
    throw 'Preflight does not match the three-smoke/thirteen-training-arm protocol.'
}
foreach ($entry in $manifestBefore.source_sha256.PSObject.Properties) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if (-not $sourcePath.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside project.' }
    if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Source changed after preflight: $($entry.Name)"
    }
}
$aggregate = Get-Content -LiteralPath (Join-Path $preflightDirectory 'aggregate.json') -Raw | ConvertFrom-Json
if ($aggregate.decision -ne 'PREFLIGHT_PASSED' -or $aggregate.updates_per_arm -ne 300 -or
    $aggregate.expected_arms -ne 13 -or $aggregate.estimated_formal_seconds -gt 2400 -or
    -not $aggregate.identities_verified -or -not $aggregate.gradient_entry_gate -or -not $aggregate.memory_gate) {
    throw 'Fixed-budget preflight failed.'
}
$gpu = & nvidia-smi --id=0 --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify GPU availability.' }
$fields = $gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 3500) { throw 'Insufficient GPU memory.' }
$pythonPath = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$arguments = @('-u','new/seed4_followup/train.py','--out',"runs/$RunName",'--preflight-dir',$Preflight)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started = Get-Date
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt = [ordered]@{ host=[Environment]::MachineName; pid=$process.Id; started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o'); executable=$pythonPath; arguments=$arguments;
    run_directory=$runDirectory; gpu=$gpu; maximum_minutes=40; updates_per_arm=300; expected_arms=13;
    preflight_directory=$preflightDirectory; continuous_monitoring=$false; dispatch_verified=$false }
$receiptPath = Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$deadline = (Get-Date).AddSeconds(30)
$startup = $null
do {
    if ($process.WaitForExit(500)) { throw "Child exited during dispatch; receipt retained, exit=$($process.ExitCode)." }
    $statusPath = Join-Path $runDirectory 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        try { $startup = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json } catch { continue }
        if ($startup.status -eq 'RUNNING' -and $startup.phase -eq 'training' -and $startup.completed_updates -ge 1) { break }
        if ($startup.status -in @('ERROR','INCOMPLETE','TIME_BUDGET','WATCHDOG_TIMEOUT')) { throw 'Startup failed; receipt retained.' }
    }
} while ((Get-Date) -lt $deadline)
if ($null -eq $startup -or $startup.status -ne 'RUNNING' -or $startup.phase -ne 'training' -or $startup.completed_updates -lt 1) {
    throw 'First training update not verified. Receipt retained; do not relaunch blindly.'
}
$manifest = Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.pid -ne $process.Id -or $manifest.protocol -ne 'seed4_independent_followup_v1' -or
    -not $manifest.training -or $manifest.preflight -ne $false -or $manifest.updates_per_arm -ne 300 -or
    $manifest.maximum_seconds -ne 2400 -or @($manifest.arms).Count -ne 13) {
    throw 'Child manifest mismatch.'
}
$receipt.dispatch_verified = $true
$receipt.verified_completed_updates = $startup.completed_updates
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{ pid=$process.Id; started=$receipt.started_local; gpu=$manifest.gpu; run=$runDirectory;
    receipt=$receiptPath; verified=$true; updates_per_arm=300; expected_arms=13; maximum_minutes=40 } | ConvertTo-Json
