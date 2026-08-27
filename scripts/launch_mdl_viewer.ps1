param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [uint32]$Width = 1600,
    [uint32]$Height = 900,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$catalog = Join-Path $projectRoot "build\mdl-reference\viewer\catalog.json"

& (Join-Path $PSScriptRoot "prepare_mdl_viewer.ps1") -Output "build\mdl-reference\viewer\catalog.json"
if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the formal MDL viewer catalog" }

if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot "build_viewer.ps1") -Configuration $Configuration
    if ($LASTEXITCODE -ne 0) { throw "Failed to build NclsViewer" }
}

$viewer = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\$Configuration\NclsViewer.exe"
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    throw "NclsViewer executable is missing: $viewer"
}
if (-not (Test-Path -LiteralPath $catalog -PathType Leaf)) {
    throw "MDL viewer catalog is missing: $catalog"
}

$arguments = @(
    "--material", $catalog,
    "--width", $Width.ToString(),
    "--height", $Height.ToString()
)
$process = Start-Process -FilePath $viewer -ArgumentList $arguments -WorkingDirectory $projectRoot -PassThru
Write-Output "NclsViewer started: PID=$($process.Id)"
Write-Output "Catalog: $catalog"
