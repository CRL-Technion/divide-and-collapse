# Recursively clone Tang et al.'s Judgelight and pin to the SHAs frozen in
# third_party/SUBMODULE_SHAS.json. Re-run after a SHAs bump.
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Target   = Join-Path $RepoRoot 'third_party\judgelight'
$Shas     = Join-Path $RepoRoot 'third_party\SUBMODULE_SHAS.json'

if (Test-Path (Join-Path $Target '.git')) {
    Write-Host "[setup_judgelight] $Target already exists - pulling pinned SHAs"
} else {
    Write-Host "[setup_judgelight] recursive clone into $Target"
    git clone --recurse-submodules https://github.com/TachikakaMin/Judgelight.git $Target
}

$ShasObj = Get-Content $Shas -Raw | ConvertFrom-Json

git -C $Target fetch --all --tags
git -C $Target checkout $ShasObj.judgelight.sha

foreach ($name in $ShasObj.nested_submodules.PSObject.Properties.Name) {
    $info = $ShasObj.nested_submodules.$name
    $sub  = Join-Path $Target $name
    git -C $sub fetch --all --tags
    git -C $sub checkout $info.sha
}
Write-Host "[setup_judgelight] pinned SHAs applied"
