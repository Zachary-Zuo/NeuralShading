param(
    [string]$OutputDir = "artifacts/reference-parity/mdl/native-fixtures-v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $projectRoot "tools\reference\mdl_native_parity.py"
& conda run --no-capture-output -n neural-shading python -m ncls.runtime -- $runner --output-dir $OutputDir
exit $LASTEXITCODE
