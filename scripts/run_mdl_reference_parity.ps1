param(
    [ValidateSet("calibration", "formal")]
    [string]$Mode = "formal",
    [string[]]$AssetId = @("carpaint-shifting-flakes", "copper-antique-brushed-patinated"),
    [string]$OutputDir = "artifacts/reference-parity/mdl/formal-v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $projectRoot "tools\reference\mdl_parity.py"
$falcor2Extension = Get-ChildItem -LiteralPath (Join-Path $projectRoot "external\falcor2\falcor2") `
    -Filter "falcor2_ext*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $falcor2Extension) {
    throw "falcor2 oracle is not built; run scripts/build_falcor2_oracle.ps1"
}

$arguments = @($runner, "--mode", $Mode, "--output-dir", $OutputDir)
foreach ($id in $AssetId) {
    $arguments += @("--asset-id", $id)
}
& (Join-Path $projectRoot "scripts\run_falcor_python.ps1") @arguments
exit $LASTEXITCODE
