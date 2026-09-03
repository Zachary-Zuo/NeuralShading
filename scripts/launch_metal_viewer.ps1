param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [uint32]$Width = 1600,
    [uint32]$Height = 900,
    [string]$CatalogRoot = "artifacts\viewer\metal-step00020000",
    [switch]$AcceptNvidiaOmniverseTerms,
    [switch]$SkipPrepare,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$catalog = Join-Path (Join-Path $projectRoot $CatalogRoot) "catalog.json"

if (-not $SkipPrepare -and -not (Test-Path -LiteralPath $catalog -PathType Leaf)) {
    & (Join-Path $PSScriptRoot "prepare_metal_viewer.ps1") `
        -OutputRoot $CatalogRoot `
        -AcceptNvidiaOmniverseTerms:$AcceptNvidiaOmniverseTerms
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the linked Metal viewer catalog" }
}
if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot "build_viewer.ps1") -Configuration $Configuration
    if ($LASTEXITCODE -ne 0) { throw "Failed to build NclsViewer" }
}

$viewer = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\$Configuration\NclsViewer.exe"
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    throw "NclsViewer executable is missing: $viewer"
}
if (-not (Test-Path -LiteralPath $catalog -PathType Leaf)) {
    throw "Linked Metal viewer catalog is missing: $catalog"
}

$arguments = @(
    "--material", $catalog,
    "--bundle-root", (Join-Path (Join-Path $projectRoot $CatalogRoot) "manual-packages"),
    "--evaluator-preview-lighting",
    "--width", $Width.ToString(),
    "--height", $Height.ToString()
)
$process = Start-Process -FilePath $viewer -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -PassThru
Write-Output "NclsViewer started: PID=$($process.Id)"
Write-Output "Linked catalog: $catalog"
