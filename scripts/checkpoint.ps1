param(
    [Parameter(Mandatory=$true)][string]$Message,
    [string]$Python = 'python',
    [switch]$NoPush
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
$branchName = & git branch --show-current
if ($LASTEXITCODE -ne 0 -or $branchName -notlike 'codex/*') { throw 'Use a codex/* development branch.' }
$alreadyStaged = & git diff --cached --name-only
if ($alreadyStaged) { throw 'Review existing staged changes before running a checkpoint.' }
Invoke-Checked -Executable $Python -Arguments @('-X','utf8','-m','pytest','-q')
Invoke-Checked -Executable 'git' -Arguments @('diff','--check')
# Explicit source directories only. Patient stores and generated artifacts are ignored.
Invoke-Checked -Executable 'git' -Arguments @('add','--','medic','scripts','tests','docs','.github','README.md','pyproject.toml','.gitignore')
$stagedPaths = & git diff --cached --name-only
foreach ($candidate in $stagedPaths) {
    if ($candidate -match '(?i)(^|/)(\.medic|medic-data|records|reports|\.env)(/|$)|\.(dcm|dicom|nrrd|nii|zip|iso|pem|key)$') {
        throw "Private/generated path staged: $candidate. Review the index before committing."
    }
}
Invoke-Checked -Executable 'git' -Arguments @('diff','--cached','--check')
Invoke-Checked -Executable 'git' -Arguments @('commit','-m',$Message)
if (-not $NoPush) { Invoke-Checked -Executable 'git' -Arguments @('push','--set-upstream','origin','HEAD') }
