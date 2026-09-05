param(
    [ValidateSet("Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ($env:OS -ne "Windows_NT") {
    throw "Windows 构建请在原生 Windows PowerShell 运行；Linux 使用 scripts/deploy_reference_linux.sh"
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "PATH 中没有 conda；本脚本不会安装 Conda"
}

$layoutJson = & conda run -n neural-shading python `
    tools/reference/reference_backend_deploy.py layout `
    --platform-id windows-x86_64@1 --project-root $projectRoot
if ($LASTEXITCODE -ne 0) { throw "无法读取 reference backend manifest" }
$layout = $layoutJson | Where-Object { $_.Trim() } | Select-Object -First 1 | ConvertFrom-Json
$falcorRoot = [string]$layout.falcor_root
$falcorExtension = [string]$layout.falcor_extension

& conda run -n neural-shading python tools/reference/reference_backend_deploy.py plan `
    --platform-id windows-x86_64@1 --project-root $projectRoot
if ($LASTEXITCODE -ne 0) { throw "锁定的 Windows reference 依赖不完整或发生漂移" }

if (-not (Test-Path -LiteralPath $falcorExtension)) {
    $setup = Join-Path $falcorRoot "setup.bat"
    $cmake = Join-Path $falcorRoot "tools\.packman\cmake\bin\cmake.exe"
    if (-not (Test-Path -LiteralPath $cmake)) {
        if (-not (Test-Path -LiteralPath $setup)) { throw "Falcor setup.bat 缺失" }
        & $setup
        if ($LASTEXITCODE -ne 0) { throw "Falcor setup 失败" }
    }
    Push-Location $falcorRoot
    try {
        & $cmake --preset windows-vs2022
        if ($LASTEXITCODE -ne 0) { throw "Falcor configure 失败" }
        & $cmake --build build/windows-vs2022 --config $Configuration `
            --target FalcorPython --parallel
        if ($LASTEXITCODE -ne 0) { throw "FalcorPython build 失败" }
    }
    finally {
        Pop-Location
    }
}

& (Join-Path $PSScriptRoot "build_mdl_reference.ps1") -Configuration $Configuration
if ($LASTEXITCODE -ne 0) { throw "MDL program provider build 失败" }

& conda run --no-capture-output -n neural-shading python -m ncls reference probe
if ($LASTEXITCODE -ne 0) { throw "reference backend asset-free probe 失败" }

Write-Host "Reference backend ready: Windows/D3D12; assets not managed"
