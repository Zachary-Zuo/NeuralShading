param(
    [Parameter(Mandatory = $true)][string]$Package,
    [Parameter(Mandatory = $true)][string]$Material,
    [ValidateSet("Release", "Debug")][string]$Configuration = "Release",
    [ValidateSet("path-tracing", "deferred")][string]$ReferenceMode = "path-tracing",
    [ValidateSet("path-tracing", "deferred")][string]$NeuralMode = "deferred",
    [uint32]$Width = 1600,
    [uint32]$Height = 900,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$packagePath = (Resolve-Path -LiteralPath $Package).Path
$materialPath = (Resolve-Path -LiteralPath $Material).Path
$manifest = Get-Content -LiteralPath (Join-Path $packagePath "manifest.json") -Encoding UTF8 -Raw | ConvertFrom-Json
if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot "build_viewer.ps1") -Configuration $Configuration
    if (-not $?) { throw "Failed to build NclsViewer" }
}
$viewer = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\$Configuration\NclsViewer.exe"
& $viewer --material $materialPath --bundle-root $packagePath `
    --slot0-package source-reference --slot0-mode $ReferenceMode `
    --slot1-package ([string]$manifest.package_id) --slot1-mode $NeuralMode `
    --width $Width --height $Height
if ($LASTEXITCODE -ne 0) { throw "NclsViewer failed: $LASTEXITCODE" }
