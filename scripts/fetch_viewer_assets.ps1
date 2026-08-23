param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$presetPath = Join-Path $projectRoot "configs\viewer-studio-v1.json"
if (-not (Test-Path -LiteralPath $presetPath -PathType Leaf)) {
    throw "Viewer studio preset does not exist: $presetPath"
}
$preset = Get-Content -LiteralPath $presetPath -Encoding UTF8 -Raw | ConvertFrom-Json

function Assert-Sha256([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch: expected $Expected, got $actual ($Path)"
    }
}

$sceneSource = Join-Path $projectRoot ($preset.reference_geometry_source -replace '/', '\')
$sceneTargetDirectory = Join-Path $projectRoot "assets\viewer\scenes\studio-v1"
$sceneTarget = Join-Path $sceneTargetDirectory $preset.reference_geometry
Assert-Sha256 $sceneSource $preset.reference_geometry_sha256 "Pinned MaterialX shaderball"
New-Item -ItemType Directory -Force -Path $sceneTargetDirectory | Out-Null
Copy-Item -LiteralPath $sceneSource -Destination $sceneTarget -Force
Assert-Sha256 $sceneTarget $preset.reference_geometry_sha256 "Provisioned viewer shaderball"

$environmentTarget = Join-Path $projectRoot "assets\viewer\environments\polyhaven-1k\$($preset.environment)"
$environmentValid = Test-Path -LiteralPath $environmentTarget -PathType Leaf
if ($environmentValid) {
    $environmentValid = (Get-FileHash -Algorithm SHA256 -LiteralPath $environmentTarget).Hash.ToLowerInvariant() `
        -eq $preset.environment_sha256.ToLowerInvariant()
}
if (-not $environmentValid) {
    & conda run -n neural-shading python (Join-Path $projectRoot "scripts\download_polyhaven_hdris.py")
    if ($LASTEXITCODE -ne 0) { throw "Failed to acquire the frozen Poly Haven HDRI set" }
}
Assert-Sha256 $environmentTarget $preset.environment_sha256 "Provisioned viewer HDRI"
Write-Output "Viewer studio-v1 assets verified."
