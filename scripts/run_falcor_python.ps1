param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$projectSrc = Join-Path $projectRoot "src"
$falcorBin = Join-Path $projectRoot "external\Falcor\build\windows-vs2022\bin\Release"
$falcorModule = Join-Path $falcorBin "python"

if (-not (Test-Path -LiteralPath (Join-Path $falcorModule "falcor\falcor_ext.cp310-win_amd64.pyd"))) {
    throw "FalcorPython Release build was not found. Build target FalcorPython first."
}

$env:PATH = $falcorBin + ";" + $env:PATH
$env:PYTHONPATH = $projectSrc + ";" + $falcorModule

& conda run -n neural-shading python @PythonArgs
exit $LASTEXITCODE
