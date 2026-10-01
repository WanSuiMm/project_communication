param(
    [string]$RunName = 'workspace_revision_20261002_paired01',
    [string]$Preflight = 'runs/workspace_revision_20261002_preflight'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a directory basename.' }
$runDirectory = Join-Path $projectRoot "runs/$RunName"
$launchDirectory = Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName, (Get-Date -Format yyyyMMdd_HHmmss))
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) {
    throw 'Existing output cannot be reused.'
}
$checkedDirectory = Join-Path $projectRoot $Preflight
$checked = Get-Content -LiteralPath (Join-Path $checkedDirectory 'status.json') -Raw | ConvertFrom-Json
if ($checked.status -ne 'PREFLIGHT_PASSED') { throw 'Required two-arm preflight did not pass.' }
$sources = (Get-Content -LiteralPath (Join-Path $checkedDirectory 'manifest.json') -Raw | ConvertFrom-Json).source_sha256
foreach ($entry in $sources.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $projectRoot $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw "Source changed since preflight: $($entry.Name)" }
}
$gpu = & nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify GPU availability.' }
$fields = $gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 3500) { throw 'Insufficient free GPU memory.' }
$pythonPath = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$arguments = @('-u', 'new/workspace_revision/run_revision.py', '--out', "runs/$RunName")
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started = Get-Date
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt = [ordered]@{ host=[Environment]::MachineName; pid=$process.Id;
    started_local=$started.ToString('o'); started_utc=$started.ToUniversalTime().ToString('o');
    executable=$pythonPath; arguments=$arguments; working_directory=$projectRoot;
    run_directory=$runDirectory; gpu=$gpu; maximum_minutes=25;
    training=$true; continuous_monitoring=$false; dispatch_verified=$false }
$receiptPath = Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$deadline = (Get-Date).AddSeconds(25)
do {
    if ($process.WaitForExit(500)) {
        Get-Content -LiteralPath (Join-Path $launchDirectory 'stderr.log')
        throw "Child exited during dispatch with code $($process.ExitCode); receipt retained."
    }
    $manifestPath = Join-Path $runDirectory 'manifest.json'
    $statusPath = Join-Path $runDirectory 'status.json'
    if ((Test-Path -LiteralPath $manifestPath) -and (Test-Path -LiteralPath $statusPath)) {
        $startup = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($startup.status -eq 'RUNNING' -and $startup.phase -eq 'training' -and $startup.completed_updates -ge 1) { break }
        if ($startup.status -in @('ERROR', 'FAILED_OR_INCOMPLETE', 'TIME_BUDGET')) {
            throw "Child startup failed: $($startup.status); receipt retained."
        }
    }
} while ((Get-Date) -lt $deadline)
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Manifest pending; receipt retained. Do not relaunch blindly.' }
if ($null -eq $startup -or $startup.status -ne 'RUNNING' -or $startup.phase -ne 'training' -or $startup.completed_updates -lt 1) {
    throw 'First training update not yet verified; receipt retained. Do not relaunch blindly.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.pid -ne $process.Id -or $manifest.training -ne $true -or $manifest.preflight -ne $false) {
    throw 'Child manifest does not match authorized training dispatch.'
}
$receipt.dispatch_verified = $true
$receipt.verified_completed_updates = $startup.completed_updates
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{ pid=$process.Id; started=$receipt.started_local; gpu=$manifest.gpu;
    run=$runDirectory; receipt=$receiptPath; verified=$true; maximum_minutes=25 } | ConvertTo-Json
