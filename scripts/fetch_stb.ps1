param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$externalRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "external")).Path
$target = Join-Path $externalRoot "stb"
$remote = "https://github.com/nothings/stb.git"
$commit = "013ac3beddff3dbffafd5177e7972067cd2b5083"
$headerSha256 = "594C2FE35D49488B4382DBFAEC8F98366DEFCA819D916AC95BECF3E75F4200B3"

if ([System.IO.Path]::GetFullPath((Split-Path -Parent $target)) -ne $externalRoot) {
    throw "Refusing to install stb outside external/: $target"
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
    throw "stb target exists but is not a Git clone: $target"
}

$actualCommit = (& git -C $target rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $commit) {
    throw "stb commit mismatch: expected=$commit actual=$actualCommit"
}
$dirty = @(& git -C $target status --porcelain --ignore-submodules=all)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
    throw "stb worktree is not clean; preserve or remove local changes before continuing"
}
$actualHeaderHash = (Get-FileHash -LiteralPath (Join-Path $target "stb_image.h") -Algorithm SHA256).Hash
if ($actualHeaderHash -ne $headerSha256) {
    throw "stb_image.h hash mismatch: expected=$headerSha256 actual=$actualHeaderHash"
}

Write-Host "stb source ready: $target @ $commit"
