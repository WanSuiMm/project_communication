param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Qualification,
    [string]$ResumeParent
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if($RunName -notmatch '^[A-Za-z0-9_-]+$'){throw 'RunName must be a basename.'}
$target=Join-Path $projectRoot "runs/$RunName"
if(Test-Path -LiteralPath $target){throw 'New output directory required.'}
$qualificationPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Qualification))
if(-not $qualificationPath.StartsWith($projectRoot+'\analyses\',[StringComparison]::OrdinalIgnoreCase)){throw 'Qualification outside analyses.'}
$checked=Get-Content -LiteralPath $qualificationPath -Raw | ConvertFrom-Json
if($checked.status -ne 'PASS' -or $checked.protocol -ne 'rrc_v0_prestream_relation_factorization_v1'){throw 'Qualification failed.'}
foreach($entry in $checked.source_sha256.PSObject.Properties) {
    $source=[IO.Path]::GetFullPath((Join-Path $projectRoot $entry.Name))
    if(-not $source.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value){throw "Source binding changed: $($entry.Name)"}
}
$gpu=& nvidia-smi --id=0 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
if($LASTEXITCODE -ne 0){throw 'Cannot inspect GPU.'}
$parts=$gpu.Split(',')
if(([int]$parts[0]-[int]$parts[1]) -lt 4000){throw 'Insufficient GPU headroom.'}
if([IO.DriveInfo]::new([IO.Path]::GetPathRoot($target)).AvailableFreeSpace -lt 3GB){throw 'Insufficient disk headroom.'}
if($ResumeParent) {
    $parent=[IO.Path]::GetFullPath((Join-Path $projectRoot $ResumeParent))
    if(-not $parent.StartsWith($projectRoot+'\runs\',[StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath (Join-Path $parent 'manifest.json'))){throw 'Resume parent unavailable.'}
    $script='new/rrc_recovery/run.py'
    $arguments=@('--parent',$ResumeParent,'--out',"runs/$RunName")
} else {
    $script='new/rrc_v0/run.py'
    $arguments=@('--out',"runs/$RunName",'--qualification',$Qualification)
}
& (Join-Path $PSScriptRoot 'start_protected_job.ps1') -JobName $RunName -Script $script -ScriptArguments $arguments
