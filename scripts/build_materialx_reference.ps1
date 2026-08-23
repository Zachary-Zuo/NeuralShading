param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$sourceRoot = Join-Path $projectRoot "external\MaterialX"
$buildRoot = Join-Path $projectRoot "build\materialx-reference"
$installRoot = Join-Path $projectRoot "build\materialx-reference-install"
$probeSourceRoot = Join-Path $projectRoot "tools\reference\materialx_probe"
$probeBuildRoot = Join-Path $projectRoot "build\materialx-probe"
$expectedRevision = "270b5cf2ae2be24a3b6ef4b0569f1c93038dda1d"

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    throw "缺少锁定的 external/MaterialX；先运行 scripts/fetch_reference_sources.ps1 -MaterialX"
}
$actualRevision = git -C $sourceRoot rev-parse HEAD
if ($actualRevision -ne $expectedRevision) {
    throw "MaterialX revision mismatch: expected=$expectedRevision actual=$actualRevision"
}
if (git -C $sourceRoot status --short) {
    throw "external/MaterialX 必须是干净工作树"
}

$environmentPrefix = (& conda run -n neural-shading python -c 'import sys; print(sys.prefix)' | Select-Object -First 1)
if (-not $environmentPrefix -or -not (Test-Path -LiteralPath $environmentPrefix)) {
    throw "无法解析 neural-shading Conda 环境路径"
}
$environmentPrefix = $environmentPrefix.Trim()
$cmakePrefix = Join-Path $environmentPrefix "Library"

cmake -S $sourceRoot -B $buildRoot `
    -G "Visual Studio 17 2022" -A x64 `
    -DMATERIALX_BUILD_VIEWER=ON `
    -DMATERIALX_BUILD_OIIO=ON `
    -DMATERIALX_BUILD_TESTS=OFF `
    -DMATERIALX_BUILD_PYTHON=OFF `
    -DMATERIALX_BUILD_DOCS=OFF `
    -DMATERIALX_BUILD_GRAPH_EDITOR=OFF `
    -DMATERIALX_INSTALL_RESOURCES=ON `
    -DCMAKE_INSTALL_PREFIX="$installRoot" `
    -DCMAKE_PREFIX_PATH="$cmakePrefix"
if ($LASTEXITCODE -ne 0) { throw "MaterialX reference configure 失败" }

cmake --build $buildRoot --config $Configuration --target MaterialXView --parallel 12
if ($LASTEXITCODE -ne 0) { throw "MaterialXView build 失败" }

# 独立 parity probe 链接 MaterialX 的已安装 CMake targets。install target 会先补齐
# shader generator/render library，再把标准库资源放到固定的 build 位置。
cmake --build $buildRoot --config $Configuration --target install --parallel 12
if ($LASTEXITCODE -ne 0) { throw "MaterialX reference install 失败" }

cmake -S $probeSourceRoot -B $probeBuildRoot `
    -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_PREFIX_PATH="$installRoot;$cmakePrefix"
if ($LASTEXITCODE -ne 0) { throw "MaterialX parity probe configure 失败" }

cmake --build $probeBuildRoot --config $Configuration --parallel 12
if ($LASTEXITCODE -ne 0) { throw "MaterialX parity probe build 失败" }

if (git -C $sourceRoot status --short) {
    throw "MaterialX build 后上游工作树不干净"
}
Write-Host "MaterialXView ready: $buildRoot\bin\$Configuration\MaterialXView.exe"
Write-Host "MaterialX parity probe ready: $probeBuildRoot\$Configuration\ncls_materialx_probe.exe"
