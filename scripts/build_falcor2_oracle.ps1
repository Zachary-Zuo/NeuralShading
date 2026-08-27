param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$falcor2Root = Join-Path $projectRoot "external\falcor2"
$sdkArchive = Join-Path $projectRoot "external\MDL-SDK-2025.0.0-387700.1252-nt-x86-64.zip"
$builder = Join-Path $projectRoot "tools\reference\build_falcor2_oracle.py"
$commit = "d629c967fa800af81cf5c916bfb2a825b012f473"
$expectedSdkSha256 = "407464bb19371ad3dc92fb64db52af6ece2177a48d6811dc0f461de3f392b546"

if (-not (Test-Path -LiteralPath (Join-Path $falcor2Root ".git"))) {
    throw "Missing pinned falcor2 source; run scripts/fetch_falcor2_oracle.ps1"
}
$actualCommit = (& git -C $falcor2Root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $commit) {
    throw "falcor2 commit mismatch: expected=$commit actual=$actualCommit"
}
$dirty = @(& git -C $falcor2Root status --porcelain)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
    throw "falcor2 worktree must be clean before building the oracle"
}
if (-not (Test-Path -LiteralPath $sdkArchive)) {
    throw "Missing pinned MDL SDK archive; run scripts/fetch_mdl_sdk.ps1"
}
$actualSdkSha256 = (Get-FileHash -LiteralPath $sdkArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSdkSha256 -ne $expectedSdkSha256) {
    throw "MDL SDK archive SHA-256 mismatch: expected=$expectedSdkSha256 actual=$actualSdkSha256"
}

conda run -n neural-shading python $builder `
    --source $falcor2Root `
    --mdl-archive $sdkArchive `
    --configuration $Configuration
if ($LASTEXITCODE -ne 0) {
    throw "falcor2 oracle build failed"
}

$extension = Get-ChildItem -LiteralPath (Join-Path $falcor2Root "falcor2") -Filter "falcor2_ext*.pyd" | Select-Object -First 1
if ($null -eq $extension) {
    throw "falcor2 oracle extension is missing after build"
}
Write-Host "falcor2 oracle ready: $($extension.FullName)"
