param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$externalRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "external")).Path
$target = Join-Path $externalRoot "falcor2"
$remote = "https://github.com/NVlabs/falcor2.git"
$commit = "d629c967fa800af81cf5c916bfb2a825b012f473"

if ([System.IO.Path]::GetFullPath((Split-Path -Parent $target)) -ne $externalRoot) {
    throw "Refusing to install falcor2 outside external/: $target"
}

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git failed: git $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $target)) {
    Invoke-Git -Arguments @("clone", "--filter=blob:none", "--no-checkout", $remote, $target)
    Invoke-Git -Arguments @("-C", $target, "fetch", "--depth", "1", "origin", $commit)
    Invoke-Git -Arguments @("-C", $target, "checkout", "--detach", $commit)
} elseif (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
    throw "falcor2 target exists but is not a Git clone: $target"
}

$actualCommit = (& git -C $target rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $commit) {
    throw "falcor2 commit mismatch: expected=$commit actual=$actualCommit"
}
$dirty = @(& git -C $target status --porcelain --ignore-submodules=all)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
    throw "falcor2 worktree is not clean; preserve or remove local changes before continuing"
}

Invoke-Git -Arguments @("-C", $target, "submodule", "sync", "--recursive")
Invoke-Git -Arguments @("-C", $target, "submodule", "update", "--init", "--recursive")

$submoduleStatus = @(& git -C $target submodule status --recursive)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect falcor2 submodules"
}
$invalid = @($submoduleStatus | Where-Object { $_ -match '^[+-U]' })
if ($invalid.Count -ne 0) {
    throw "falcor2 submodule state mismatch: $($invalid -join '; ')"
}

$dirty = @(& git -C $target status --porcelain --ignore-submodules=none)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
    throw "falcor2 worktree changed during setup"
}
Write-Host "falcor2 oracle source ready: $target @ $commit"
