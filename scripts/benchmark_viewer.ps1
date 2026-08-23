param(
    [Parameter(Mandatory = $true)]
    [string]$BundleRoot,
    [string]$MethodId = "",
    [string]$Material = "",
    [string]$Environment = "",
    [string]$Preset = "configs\viewer-benchmark-v1.json",
    [string]$OutputDirectory = "artifacts\benchmarks\viewer",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

$bundleRootPath = Resolve-ProjectPath $BundleRoot
$presetPath = Resolve-ProjectPath $Preset
$outputPath = Resolve-ProjectPath $OutputDirectory
$viewer = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\$Configuration\NclsViewer.exe"

if ($Build) {
    & (Join-Path $PSScriptRoot "build_viewer.ps1") -Configuration $Configuration
    if ($LASTEXITCODE -ne 0) { throw "NclsViewer build failed" }
}
else {
    & (Join-Path $PSScriptRoot "fetch_viewer_assets.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Failed to provision the fixed viewer scene" }
}
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    throw "NclsViewer was not found: $viewer; run scripts\build_viewer.ps1 first."
}
if (-not (Test-Path -LiteralPath $bundleRootPath -PathType Container)) {
    throw "MethodBundle root does not exist: $bundleRootPath"
}
if (-not (Test-Path -LiteralPath $presetPath -PathType Leaf)) {
    throw "Benchmark preset does not exist: $presetPath"
}

$presetDocument = Get-Content -LiteralPath $presetPath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($presetDocument.format_name -ne "ncls.viewer-benchmark" -or $presetDocument.format_version -ne 2) {
    throw "Unsupported viewer benchmark preset"
}
if ($presetDocument.camera_path.Count -lt 1) { throw "camera_path must not be empty" }

if ([string]::IsNullOrWhiteSpace($MethodId)) {
    $available = @()
    Get-ChildItem -LiteralPath $bundleRootPath -Filter manifest.json -File -Recurse | ForEach-Object {
        $manifest = Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Raw | ConvertFrom-Json
        if ($manifest.format_name -eq "ncls.method-bundle" -and $manifest.runtime_class -eq "realtime") {
            $available += $manifest.method_id
        }
    }
    $available = @($available | Sort-Object -Unique)
    if ($available.Count -ne 1) {
        throw "-MethodId is required unless BundleRoot contains exactly one realtime MethodBundle."
    }
    $MethodId = $available[0]
}

$materialPath = if ([string]::IsNullOrWhiteSpace($Material)) {
    Resolve-ProjectPath ([string]$presetDocument.source_material)
} else { Resolve-ProjectPath $Material }
$environmentPath = if ([string]::IsNullOrWhiteSpace($Environment)) {
    Resolve-ProjectPath ([string]$presetDocument.environment)
} else { Resolve-ProjectPath $Environment }
$referenceGeometryPath = Resolve-ProjectPath ([string]$presetDocument.reference_geometry)
if ($materialPath -and -not (Test-Path -LiteralPath $materialPath -PathType Leaf)) { throw "MaterialProgram does not exist: $materialPath" }
if ($environmentPath -and -not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) { throw "HDRI does not exist: $environmentPath" }
if (-not (Test-Path -LiteralPath $referenceGeometryPath -PathType Leaf)) { throw "Fixed reference scene does not exist: $referenceGeometryPath" }
$referenceGeometrySha256 = (Get-FileHash -LiteralPath $referenceGeometryPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($referenceGeometrySha256 -ne ([string]$presetDocument.reference_geometry_sha256).ToLowerInvariant()) {
    throw "Fixed reference scene SHA-256 does not match the benchmark preset"
}
$environmentSha256 = (Get-FileHash -LiteralPath $environmentPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($Environment) -and $environmentSha256 -ne ([string]$presetDocument.environment_sha256).ToLowerInvariant()) {
    throw "Fixed HDRI SHA-256 does not match the benchmark preset"
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$inputPath = Join-Path $outputPath "replay-inputs"
New-Item -ItemType Directory -Path $inputPath -Force | Out-Null
$totalFrames = [uint32]$presetDocument.warmup_frames + [uint32]$presetDocument.measurement_frames
$results = @()

foreach ($camera in $presetDocument.camera_path) {
    if ([string]::IsNullOrWhiteSpace($camera.id)) { throw "Each camera_path item needs a non-empty id" }
    $replay = [ordered]@{
        format_name = "ncls.viewer-capture"
        format_version = 3
        method_id = $MethodId
        bundle_root = $bundleRootPath
        source_material = $materialPath
        environment = $environmentPath
        environment_sha256 = $environmentSha256
        reference_geometry = $referenceGeometryPath
        reference_geometry_sha256 = $referenceGeometrySha256
        resolution = @([uint32]$presetDocument.resolution[0], [uint32]$presetDocument.resolution[1])
        object_mode = [uint32]$presetDocument.object_mode
        reference_spp = $totalFrames * [uint32]$presetDocument.reference_samples_per_frame
        reference_samples_per_frame = [uint32]$presetDocument.reference_samples_per_frame
        reference_integrator = "ncls.scene-path-tracer@1"
        reference_scene_max_bounces = [uint32]$presetDocument.reference_scene_max_bounces
        reference_layer_walk_max_depth = [uint32]$presetDocument.reference_layer_walk_max_depth
        camera = $camera
        display = $presetDocument.display
        lighting = $presetDocument.lighting
    }
    $replayPath = Join-Path $inputPath ($camera.id + ".json")
    $replay | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $replayPath -Encoding UTF8
    $captureDirectory = Join-Path $outputPath ("captures\" + $camera.id)
    $capturePath = Join-Path $captureDirectory "capture.json"
    $viewerOutput = & $viewer --replay $replayPath --headless --frames $totalFrames --capture $capturePath 2>&1
    $viewerExitCode = $LASTEXITCODE
    $viewerOutput | Set-Content -LiteralPath (Join-Path $outputPath ($camera.id + ".log")) -Encoding UTF8
    $viewerOutput | Write-Output
    if ($viewerExitCode -ne 0) { throw "NclsViewer failed at camera '$($camera.id)' with exit code $viewerExitCode" }
    $capture = Get-Content -LiteralPath $capturePath -Encoding UTF8 -Raw | ConvertFrom-Json
    $results += [PSCustomObject][ordered]@{
        camera_id = $camera.id
        method_id = $capture.method_id
        width = [uint32]$capture.resolution[0]
        height = [uint32]$capture.resolution[1]
        reference_spp = [uint32]$capture.reference_spp
        estimated_mean_relative_standard_error = [double]$capture.estimated_mean_relative_standard_error
        visibility_ms = [double]$capture.gpu_ms.visibility
        reference_ms = [double]$capture.gpu_ms.reference
        prepare_ms = [double]$capture.gpu_ms.prepare
        lighting_ms = [double]$capture.gpu_ms.lighting
        composite_ms = [double]$capture.gpu_ms.composite
        capture_manifest = $capturePath
    }
}

$results | Export-Csv -LiteralPath (Join-Path $outputPath "metrics.csv") -NoTypeInformation -Encoding UTF8
$summary = [ordered]@{
    format_name = "ncls.viewer-benchmark-result"
    format_version = 2
    created_at = [DateTimeOffset]::UtcNow.ToString("o")
    method_id = $MethodId
    bundle_root = $bundleRootPath
    preset = $presetPath
    preset_sha256 = (Get-FileHash -LiteralPath $presetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    viewer_sha256 = (Get-FileHash -LiteralPath $viewer -Algorithm SHA256).Hash.ToLowerInvariant()
    source_git_commit = (& git -C $projectRoot rev-parse HEAD).Trim()
    falcor_commit = (& git -C (Join-Path $projectRoot "external\Falcor") rev-parse HEAD).Trim()
    reference_geometry = $referenceGeometryPath
    reference_geometry_sha256 = $referenceGeometrySha256
    source_material = $materialPath
    source_material_sha256 = (Get-FileHash -LiteralPath $materialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    environment = $environmentPath
    environment_sha256 = $environmentSha256
    warmup_frames = [uint32]$presetDocument.warmup_frames
    measurement_frames = [uint32]$presetDocument.measurement_frames
    timing_semantics = "Each fixed camera runs warmup+measurement frames. Timings are the last GPU timestamp before capture; visibility/prepare are one-shot costs for that camera."
    results = $results
}
$summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath "summary.json") -Encoding UTF8
Write-Output (Join-Path $outputPath "summary.json")
