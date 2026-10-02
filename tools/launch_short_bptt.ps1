param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Preflight
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a directory basename.' }
$runDirectory = Join-Path $projectRoot "runs/$RunName"
$launchDirectory = Join-Path $projectRoot ('runs/{0}_launch_{1}' -f $RunName, (Get-Date -Format yyyyMMdd_HHmmss))
if ((Test-Path -LiteralPath $runDirectory) -or (Test-Path -LiteralPath $launchDirectory)) { throw 'Output already exists.' }
$preflightDirectory = Join-Path $projectRoot $Preflight
$checked = Get-Content -LiteralPath (Join-Path $preflightDirectory 'status.json') -Raw | ConvertFrom-Json
if ($checked.status -ne 'PREFLIGHT_PASSED') { throw 'Passed preflight required.' }
$sources = (Get-Content -LiteralPath (Join-Path $preflightDirectory 'manifest.json') -Raw | ConvertFrom-Json).source_sha256
foreach ($entry in $sources.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $projectRoot $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Source changed after preflight: $($entry.Name)"
    }
}
$aggregate = Get-Content -LiteralPath (Join-Path $preflightDirectory 'aggregate.json') -Raw | ConvertFrom-Json
if ($aggregate.recommended_updates -lt 200) { throw 'Insufficient bounded compute budget.' }
$gpu = & nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify GPU availability.' }
$fields = $gpu.Split(',')
if (([int]$fields[1]-[int]$fields[2]) -lt 3500) { throw 'Insufficient GPU memory.' }
$pythonPath = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$arguments = @('-u','new/short_bptt/run.py','--out',"runs/$RunName",'--preflight-dir',$Preflight)
New-Item -ItemType Directory -Path $launchDirectory | Out-Null
$started = Get-Date
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchDirectory 'stdout.log') -RedirectStandardError (Join-Path $launchDirectory 'stderr.log')
$receipt = [ordered]@{ host=[Environment]::MachineName; pid=$process.Id; started_local=$started.ToString('o');
    started_utc=$started.ToUniversalTime().ToString('o'); executable=$pythonPath; arguments=$arguments;
    run_directory=$runDirectory; gpu=$gpu; maximum_minutes=25; updates_per_arm=$aggregate.recommended_updates;
    preflight_directory=$preflightDirectory; continuous_monitoring=$false; dispatch_verified=$false }
$receiptPath = Join-Path $launchDirectory 'launch_receipt.json'
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$deadline = (Get-Date).AddSeconds(25)
$startup = $null
do {
    if ($process.WaitForExit(500)) { throw "Child exited during dispatch; receipt retained, exit=$($process.ExitCode)." }
    $statusPath = Join-Path $runDirectory 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $startup = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($startup.status -eq 'RUNNING' -and $startup.phase -eq 'training' -and $startup.completed_updates -ge 1) { break }
        if ($startup.status -in @('ERROR','FAILED_OR_INCOMPLETE','TIME_BUDGET')) { throw 'Startup failed; receipt retained.' }
    }
} while ((Get-Date) -lt $deadline)
if ($null -eq $startup -or $startup.status -ne 'RUNNING' -or $startup.phase -ne 'training' -or $startup.completed_updates -lt 1) {
    throw 'First training update not verified. Receipt retained; do not relaunch blindly.'
}
$manifest = Get-Content -LiteralPath (Join-Path $runDirectory 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.pid -ne $process.Id -or $manifest.preflight -ne $false -or $manifest.updates_per_arm -ne $aggregate.recommended_updates) {
    throw 'Child manifest mismatch.'
}
$receipt.dispatch_verified = $true
$receipt.verified_completed_updates = $startup.completed_updates
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
[pscustomobject]@{ pid=$process.Id; started=$receipt.started_local; gpu=$manifest.gpu; run=$runDirectory;
    receipt=$receiptPath; verified=$true; updates_per_arm=$manifest.updates_per_arm; maximum_minutes=25 } | ConvertTo-Json
