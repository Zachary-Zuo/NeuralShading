param(
    [string]$Output = "build\mdl-reference\viewer\catalog.json",
    [ValidateSet(
        "carpaint-shifting-flakes",
        "copper-antique-brushed-patinated",
        "aluminum-scratched",
        "ceramic-tiles-glazed-versailles",
        "velvet",
        "wood-tiles-pine-mosaic"
    )]
    [string]$DefaultAsset = "carpaint-shifting-flakes"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

& (Join-Path $PSScriptRoot "fetch_mdl_assets.ps1") -VMaterials2 -AcceptNvidiaOmniverseTerms
if ($LASTEXITCODE -ne 0) { throw "Failed to provision vMaterials 2.4.0" }
& (Join-Path $PSScriptRoot "fetch_mdl_sdk.ps1")
if ($LASTEXITCODE -ne 0) { throw "Failed to provision the pinned MDL SDK" }
& (Join-Path $PSScriptRoot "fetch_stb.ps1")
if ($LASTEXITCODE -ne 0) { throw "Failed to provision pinned stb" }
& (Join-Path $PSScriptRoot "build_mdl_reference.ps1")
if ($LASTEXITCODE -ne 0) { throw "Failed to build the formal MDL bridge" }

Push-Location $projectRoot
try {
    & conda run -n neural-shading python tools/reference/prepare_mdl_viewer.py `
        --output $Output --default-asset $DefaultAsset
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the MDL viewer catalog" }
}
finally {
    Pop-Location
}

$catalog = Join-Path $projectRoot $Output
if (-not (Test-Path -LiteralPath $catalog -PathType Leaf)) {
    throw "MDL viewer catalog was not produced: $catalog"
}
Write-Output $catalog
