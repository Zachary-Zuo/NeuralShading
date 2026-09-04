param(
    [string]$OutputRoot = "artifacts\viewer\metal-budgeted-pair",
    [Parameter(Mandatory = $true)]
    [string]$HybridCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$DirectCheckpoint,
    [switch]$AcceptNvidiaOmniverseTerms
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

$vMaterialsRoot = Join-Path $projectRoot `
    "assets\source-materials\mdl-vmaterials2\2.4.0\Materials\vMaterials_2"
if (-not (Test-Path -LiteralPath $vMaterialsRoot -PathType Container)) {
    if (-not $AcceptNvidiaOmniverseTerms) {
        throw "vMaterials 2.4.0 is missing. Re-run with -AcceptNvidiaOmniverseTerms to provision it."
    }
    & (Join-Path $PSScriptRoot "fetch_mdl_assets.ps1") `
        -VMaterials2 -AcceptNvidiaOmniverseTerms
    if (-not $?) { throw "Failed to provision vMaterials 2.4.0" }
}
& (Join-Path $PSScriptRoot "fetch_mdl_sdk.ps1")
if (-not $?) { throw "Failed to provision the pinned MDL SDK" }
& (Join-Path $PSScriptRoot "fetch_stb.ps1")
if (-not $?) { throw "Failed to provision pinned stb" }
& (Join-Path $PSScriptRoot "build_mdl_reference.ps1")
if (-not $?) { throw "Failed to build the formal MDL bridge" }

Push-Location $projectRoot
try {
    & (Join-Path $PSScriptRoot "run_falcor_python.ps1") `
        tools/viewer/prepare_metal_catalog.py `
        --output-root $OutputRoot `
        --hybrid-checkpoint $HybridCheckpoint `
        --direct-checkpoint $DirectCheckpoint
    if (-not $?) { throw "Failed to prepare the Metal budgeted viewer handoff" }
}
finally {
    Pop-Location
}

$handoff = Join-Path (Join-Path $projectRoot $OutputRoot) "handoff.json"
if (-not (Test-Path -LiteralPath $handoff -PathType Leaf)) {
    throw "Metal budgeted viewer handoff was not produced: $handoff"
}
Write-Output $handoff
