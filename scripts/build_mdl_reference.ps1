param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$layoutJson = & conda run -n neural-shading python `
    tools/reference/reference_backend_deploy.py layout `
    --platform-id windows-x86_64@1 --project-root $projectRoot
if ($LASTEXITCODE -ne 0) { throw "无法读取 reference backend manifest" }
$layout = $layoutJson | Where-Object { $_.Trim() } | Select-Object -First 1 | ConvertFrom-Json
$sdkRoot = [string]$layout.sdk_root
$stbRoot = [string]$layout.stb_root
$sourceRoot = Join-Path $projectRoot "tools\reference\mdl_sdk_bridge"
$buildRoot = Join-Path $projectRoot "build\mdl-sdk-bridge"

if (-not (Test-Path -LiteralPath (Join-Path $sdkRoot "bin\libmdl_sdk.dll"))) {
    throw "缺少锁定的 MDL SDK；先运行 scripts/fetch_mdl_sdk.ps1"
}
if (-not (Test-Path -LiteralPath (Join-Path $stbRoot "stb_image.h"))) {
    throw "缺少锁定的 stb；先运行 scripts/fetch_stb.ps1"
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "CMakeLists.txt"))) {
    throw "缺少项目 MDL SDK bridge source：$sourceRoot"
}

cmake -S $sourceRoot -B $buildRoot `
    -G "Visual Studio 17 2022" -A x64 `
    -DMDL_SDK_ROOT="$sdkRoot" `
    -DSTB_ROOT="$stbRoot"
if ($LASTEXITCODE -ne 0) { throw "MDL SDK bridge configure 失败" }

cmake --build $buildRoot --config $Configuration --parallel 12
if ($LASTEXITCODE -ne 0) { throw "MDL SDK bridge build 失败" }

$executable = [string]$layout.mdl_bridge_executable
if (-not (Test-Path -LiteralPath $executable)) {
    throw "MDL SDK bridge executable missing：$executable"
}
Write-Host "MDL SDK bridge ready: $executable"
