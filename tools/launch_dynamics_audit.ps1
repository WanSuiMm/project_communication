param([string]$RunName = 'dynamics_audit_20261001_seed0')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a directory basename.' }
$runDirectory = Join-Path $projectRoot "runs/$RunName"
$launchDirectory = Join-Path $projectRoot "runs/${RunName}_launch_$(Get-Date -Format yyyyMMdd_HHmmss)"
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) {
    throw 'Existing output cannot be reused.'
}
$preflight = Join-Path $projectRoot 'runs/dynamics_audit_20261001_preflight'
$checked = Get-Content -LiteralPath (Join-Path $preflight 'status.json') -Raw | ConvertFrom-Json
if ($checked.status -ne 'PREFLIGHT_PASSED') { throw 'Required four-arm preflight did not pass.' }
$sources = (Get-Content -LiteralPath (Join-Path $preflight 'manifest.json') -Raw | ConvertFrom-Json).source_sha256
foreach ($entry in $sources.PSObject.Properties) {
    $current = (Get-FileHash -LiteralPath (Join-Path $projectRoot $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($current -ne $entry.Value) { throw "Source changed since preflight: $($entry.Name)" }
}
$gpu = & nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify GPU availability.' }
$fields = $gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 3500) { throw 'Insufficient free GPU memory for the frozen audit.' }
$pythonPath = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$arguments = @('-u', 'new/dynamics_audit/audit.py', '--out', "runs/$RunName")
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started = Get-Date
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt = [ordered]@{ host=[Environment]::MachineName; pid=$process.Id;
    started_local=$started.ToString('o'); started_utc=$started.ToUniversalTime().ToString('o');
    executable=$pythonPath; arguments=$arguments; working_directory=$projectRoot;
    run_directory=$runDirectory; gpu=$gpu; maximum_minutes=25; training=$false;
    continuous_monitoring=$false; dispatch_verified=$false }
$receiptPath = Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$deadline = (Get-Date).AddSeconds(12)
do {
    if ($process.WaitForExit(500)) {
        Get-Content -LiteralPath (Join-Path $launchDirectory 'stderr.log')
        throw "Audit child exited during dispatch with code $($process.ExitCode); receipt retained."
    }
    $manifestPath = Join-Path $runDirectory 'manifest.json'
    if (Test-Path -LiteralPath $manifestPath) { break }
} while ((Get-Date) -lt $deadline)
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Manifest pending; receipt retained. Do not relaunch blindly.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.pid -ne $process.Id -or $manifest.training -ne $false -or $manifest.preflight -ne $false) {
    throw 'Child manifest does not match authorized inference-only dispatch.'
}
$receipt.dispatch_verified = $true
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{ pid=$process.Id; started=$receipt.started_local; gpu=$manifest.gpu;
    run=$runDirectory; receipt=$receiptPath; verified=$true; maximum_minutes=25 } | ConvertTo-Json
