param(
    [Parameter(Mandatory = $true)]
    [string]$Request,
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$falcor2Root = Join-Path $projectRoot "external\falcor2"
$runner = Join-Path $projectRoot "tools\reference\run_falcor2_mdl_oracle.py"
$extension = Get-ChildItem -LiteralPath (Join-Path $falcor2Root "falcor2") -Filter "falcor2_ext*.pyd" | Select-Object -First 1
if ($null -eq $extension) {
    throw "falcor2 oracle is not built; run scripts/build_falcor2_oracle.ps1"
}

$env:PYTHONPATH = "$falcor2Root;$falcor2Root\external\slangpy"
conda run -n neural-shading python $runner --request $Request --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "falcor2 MDL oracle failed"
}
