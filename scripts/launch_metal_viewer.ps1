param(
    [Parameter(Mandatory = $true)]
    [string]$Handoff,
    [ValidateSet("ReferenceVsHybrid", "ReferenceVsDirect", "HybridVsDirect")]
    [string]$Comparison = "ReferenceVsHybrid",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [uint32]$Width = 1600,
    [uint32]$Height = 900,
    [ValidateSet("path-tracing", "deferred")]
    [string]$LeftMode = "path-tracing",
    [ValidateSet("path-tracing", "deferred")]
    [string]$RightMode = "path-tracing",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([System.IO.Path]::IsPathRooted($Handoff)) {
    $handoffPath = [System.IO.Path]::GetFullPath($Handoff)
}
else {
    $handoffPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Handoff))
}
if (-not (Test-Path -LiteralPath $handoffPath -PathType Leaf)) {
    throw "Metal budgeted handoff is missing: $handoffPath"
}
$document = Get-Content -LiteralPath $handoffPath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($document.format_name -ne "ncls.metal-budgeted-viewer-handoff" -or `
    $document.format_version -ne 2 -or `
    $document.checkpoint_compatibility -ne "exact") {
    throw "Unsupported handoff; rebuild with tools/viewer/prepare_metal_catalog.py"
}
$handoffRoot = Split-Path -Parent $handoffPath
$catalog = Join-Path $handoffRoot ([string]$document.reference_catalog)
$bundleRoot = Join-Path $handoffRoot ([string]$document.bundle_root)
if (-not (Test-Path -LiteralPath $catalog -PathType Leaf)) {
    throw "Handoff reference catalog is missing: $catalog"
}
if (-not (Test-Path -LiteralPath $bundleRoot -PathType Container)) {
    throw "Handoff package root is missing: $bundleRoot"
}
$packages = @{}
foreach ($package in $document.packages) {
    $packages[[string]$package.role] = $package
}
if (-not $packages.ContainsKey("hybrid")) {
    throw "Handoff must contain the selected hybrid package"
}
if ($Comparison -ne "ReferenceVsHybrid" -and -not $packages.ContainsKey("direct")) {
    throw "This comparison requires an explicitly exported direct package"
}

if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot "build_viewer.ps1") -Configuration $Configuration
    if (-not $?) { throw "Failed to build NclsViewer" }
}
$viewer = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\$Configuration\NclsViewer.exe"
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    throw "NclsViewer executable is missing: $viewer"
}

$slot0Package = "source-reference"
$slot0Mode = $LeftMode
$slot1Package = [string]$packages["hybrid"].package_id
$slot1Mode = $RightMode
if ($Comparison -eq "ReferenceVsDirect") {
    $slot1Package = [string]$packages["direct"].package_id
}
elseif ($Comparison -eq "HybridVsDirect") {
    $slot0Package = [string]$packages["hybrid"].package_id
    $slot1Package = [string]$packages["direct"].package_id
}

$arguments = @(
    "--material", $catalog,
    "--bundle-root", $bundleRoot,
    "--slot0-package", $slot0Package,
    "--slot0-mode", $slot0Mode,
    "--slot1-package", $slot1Package,
    "--slot1-mode", $slot1Mode,
    "--width", $Width.ToString(),
    "--height", $Height.ToString()
)
$process = Start-Process -FilePath $viewer -ArgumentList $arguments `
    -WorkingDirectory $projectRoot -PassThru
Write-Output "NclsViewer started: PID=$($process.Id)"
Write-Output "Comparison: $Comparison"
Write-Output "Hybrid: $($packages['hybrid'].profile_id) / $($packages['hybrid'].package_id)"
if ($packages.ContainsKey("direct")) {
    Write-Output "Direct: $($packages['direct'].profile_id) / $($packages['direct'].package_id)"
}
Write-Output "Compatibility: $($document.checkpoint_compatibility)"
