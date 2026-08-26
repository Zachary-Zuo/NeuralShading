param(
    [Parameter(Mandatory = $true)][string]$PackageRoot,
    [Parameter(Mandatory = $true)][string]$Slot0PackageId,
    [Parameter(Mandatory = $true)][string]$Slot1PackageId,
    [string]$Preset = "configs\viewer-benchmark-v2.json",
    [string]$OutputDirectory = "artifacts\benchmarks\viewer",
    [ValidateSet("Release", "Debug")][string]$Configuration = "Release",
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

$packageRootPath = Resolve-ProjectPath $PackageRoot
$presetPath = Resolve-ProjectPath $Preset
$outputPath = Resolve-ProjectPath $OutputDirectory
$viewer = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\$Configuration\NclsViewer.exe"
if ($Build) { & (Join-Path $PSScriptRoot "build_viewer.ps1") -Configuration $Configuration }
if ($LASTEXITCODE -ne 0) { throw "NclsViewer build failed" }
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) { throw "NclsViewer was not found: $viewer" }
if (-not (Test-Path -LiteralPath $packageRootPath -PathType Container)) { throw "ScatteringPackage root does not exist: $packageRootPath" }
$presetDocument = Get-Content -LiteralPath $presetPath -Encoding UTF8 -Raw | ConvertFrom-Json
if ($presetDocument.format_name -ne "ncls.viewer-benchmark" -or $presetDocument.format_version -ne 2) { throw "Unsupported viewer benchmark preset" }

$packages = @{}
Get-ChildItem -LiteralPath $packageRootPath -Filter manifest.json -File -Recurse | ForEach-Object {
    $manifest = Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Raw | ConvertFrom-Json
    if ($manifest.format_name -eq "ncls.scattering-package" -and $manifest.format_version -eq 1) {
        $packages[[string]$manifest.package_id] = $_.Directory.FullName
    }
}
foreach ($id in @($Slot0PackageId, $Slot1PackageId)) {
    if (-not $packages.ContainsKey($id)) { throw "ScatteringPackage was not found: $id" }
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$results = @()
foreach ($combination in $presetDocument.combinations) {
    foreach ($camera in $presetDocument.camera_path) {
        $caseId = "$($camera.id)-$($combination[0])-$($combination[1])"
        $capturePath = Join-Path $outputPath "$caseId.json"
        $viewerOutput = & $viewer --headless --frames ([uint32]$presetDocument.warmup_frames + [uint32]$presetDocument.measurement_frames) `
            --slot0-package $packages[$Slot0PackageId] --slot0-mode $combination[0] `
            --slot1-package $packages[$Slot1PackageId] --slot1-mode $combination[1] --capture $capturePath 2>&1
        $viewerOutput | Set-Content -LiteralPath (Join-Path $outputPath "$caseId.log") -Encoding UTF8
        if ($LASTEXITCODE -ne 0) { throw "NclsViewer failed for $caseId" }
        $capture = Get-Content -LiteralPath $capturePath -Encoding UTF8 -Raw | ConvertFrom-Json
        if ($capture.format_name -ne "ncls.viewer-capture" -or $capture.format_version -ne 4 -or $capture.slots.Count -ne 2) { throw "Viewer capture contract mismatch" }
        $results += $capture
    }
}
$summary = [ordered]@{
    format_name = "ncls.viewer-benchmark-result"; format_version = 3
    created_at = [DateTimeOffset]::UtcNow.ToString("o"); preset = $presetPath
    slot_package_ids = @($Slot0PackageId, $Slot1PackageId); results = $results
}
$summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath "summary.json") -Encoding UTF8
Write-Output (Join-Path $outputPath "summary.json")
