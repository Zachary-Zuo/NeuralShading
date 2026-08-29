param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$projectSrc = Join-Path $projectRoot "src"
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "PATH 中没有 conda"
}

$layoutJson = & conda run -n neural-shading python `
    (Join-Path $projectRoot "tools/reference/reference_backend_deploy.py") layout `
    --platform-id windows-x86_64@1 --project-root $projectRoot
if ($LASTEXITCODE -ne 0) { throw "无法读取 reference backend manifest" }
$layout = $layoutJson | Where-Object { $_.Trim() } | Select-Object -First 1 | ConvertFrom-Json
$falcorBin = [string]$layout.falcor_runtime_library_root
$falcorModule = [string]$layout.falcor_python_module_root
$falcorExtension = [string]$layout.falcor_extension

if (-not (Test-Path -LiteralPath $falcorExtension)) {
    throw "FalcorPython Release build was not found. Build target FalcorPython first."
}

$env:PATH = $falcorBin + ";" + $env:PATH
$env:PYTHONPATH = $projectSrc + ";" + $falcorModule

& conda run -n neural-shading python @PythonArgs
exit $LASTEXITCODE
