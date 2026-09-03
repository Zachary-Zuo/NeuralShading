param(
    [string]$OutputRoot = "artifacts\viewer\metal-step00120000",
    [string]$Checkpoint = "artifacts\metal-linux-training\long\checkpoint.step00120000.pt",
    [string]$Registry = "references\mdl-vmaterials2-v1\metal-opaque-v1.json",
    [switch]$DiagnosticPreview,
    [uint32]$DiagnosticLimit = 0,
    [switch]$AcceptNvidiaOmniverseTerms
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

$vMaterialsRoot = Join-Path $projectRoot `
    "assets\source-materials\mdl-vmaterials2\2.4.0\Materials\vMaterials_2"
if (-not (Test-Path -LiteralPath $vMaterialsRoot -PathType Container)) {
    if (-not $AcceptNvidiaOmniverseTerms) {
        throw "vMaterials 2.4.0 is missing. Re-run with -AcceptNvidiaOmniverseTerms to provision it."
    }
    & (Join-Path $PSScriptRoot "fetch_mdl_assets.ps1") `
        -VMaterials2 -AcceptNvidiaOmniverseTerms
    if ($LASTEXITCODE -ne 0) { throw "Failed to provision vMaterials 2.4.0" }
}
& (Join-Path $PSScriptRoot "fetch_mdl_sdk.ps1")
if ($LASTEXITCODE -ne 0) { throw "Failed to provision the pinned MDL SDK" }
& (Join-Path $PSScriptRoot "fetch_stb.ps1")
if ($LASTEXITCODE -ne 0) { throw "Failed to provision pinned stb" }
& (Join-Path $PSScriptRoot "build_mdl_reference.ps1")
if ($LASTEXITCODE -ne 0) { throw "Failed to build the formal MDL bridge" }

Push-Location $projectRoot
try {
    $arguments = @(
        "tools/viewer/prepare_metal_catalog.py",
        "--output-root", $OutputRoot,
        "--checkpoint", $Checkpoint,
        "--registry", $Registry
    )
    if ($DiagnosticPreview) { $arguments += "--diagnostic-preview" }
    if ($DiagnosticLimit -gt 0) {
        if (-not $DiagnosticPreview) {
            throw "DiagnosticLimit requires DiagnosticPreview"
        }
        $arguments += @("--diagnostic-limit", $DiagnosticLimit.ToString())
    }
    & conda run --no-capture-output -n neural-shading python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the linked Metal viewer catalog" }
}
finally {
    Pop-Location
}

$catalog = Join-Path (Join-Path $projectRoot $OutputRoot) "catalog.json"
if (-not (Test-Path -LiteralPath $catalog -PathType Leaf)) {
    throw "Linked Metal viewer catalog was not produced: $catalog"
}
Write-Output $catalog
