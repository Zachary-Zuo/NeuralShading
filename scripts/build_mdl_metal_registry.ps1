param(
    [switch]$Refresh,
    [string]$InspectionRoot = "artifacts/mdl-metal-inspection-v1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$arguments = @(
    "tools/reference/generate_mdl_metal_registry.py",
    "--module-root",
    "assets/source-materials/mdl-vmaterials2/2.4.0/Materials",
    "--output",
    "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
)
Push-Location $projectRoot
try {
    if ($Refresh) {
        & conda run -n neural-shading python tools/reference/inspect_mdl_metal.py `
            --module-root assets/source-materials/mdl-vmaterials2/2.4.0/Materials `
            --output $InspectionRoot
        if ($LASTEXITCODE -ne 0) {
            throw "MDL Metal inspection failed with exit code $LASTEXITCODE"
        }
        $arguments += @(
            "--inspection-summary", (Join-Path $InspectionRoot "summary.json"),
            "--artifact-root", (Join-Path $InspectionRoot "artifacts")
        )
    }
    else {
        $arguments += "--check"
    }
    & conda run -n neural-shading python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "MDL Metal registry generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
