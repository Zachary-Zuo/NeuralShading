param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$externalRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "external")).Path
$packageName = "MDL-SDK-2025.0.0-387700.1252-nt-x86-64"
$url = "https://github.com/NVIDIA/MDL-SDK/releases/download/2025/$packageName.zip"
$expectedSize = 221112672
$expectedSha256 = "407464bb19371ad3dc92fb64db52af6ece2177a48d6811dc0f461de3f392b546"
$archive = Join-Path $externalRoot "$packageName.zip"
$target = Join-Path $externalRoot $packageName

if ([System.IO.Path]::GetFullPath((Split-Path -Parent $target)) -ne $externalRoot) {
    throw "拒绝在 external/ 之外安装 MDL SDK：$target"
}

function Assert-Archive {
    param([Parameter(Mandatory = $true)][string]$Path)
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $expectedSize) {
        throw "MDL SDK archive size mismatch: expected=$expectedSize actual=$($item.Length)"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedSha256) {
        throw "MDL SDK archive SHA-256 mismatch: expected=$expectedSha256 actual=$actual"
    }
}

if (Test-Path -LiteralPath $target) {
    if (-not (Test-Path -LiteralPath (Join-Path $target "bin\libmdl_sdk.dll"))) {
        throw "MDL SDK target exists but is incomplete：$target"
    }
    if (Test-Path -LiteralPath $archive) { Assert-Archive -Path $archive }
    Write-Host "MDL SDK already available: $target"
    exit 0
}

if (-not (Test-Path -LiteralPath $archive)) {
    $partial = "$archive.partial"
    if (Test-Path -LiteralPath $partial) {
        throw "发现未确认的 partial download，请检查后移除：$partial"
    }
    curl.exe -L --fail --progress-bar --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw "MDL SDK download failed: $url" }
    Assert-Archive -Path $partial
    Move-Item -LiteralPath $partial -Destination $archive
}
Assert-Archive -Path $archive

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
try {
    $unsafe = $zip.Entries | Where-Object {
        [System.IO.Path]::IsPathRooted($_.FullName) -or $_.FullName -match '(^|[\\/])\.\.([\\/]|$)'
    }
    if ($unsafe) { throw "MDL SDK archive contains unsafe paths" }
    $roots = @($zip.Entries | ForEach-Object { ($_.FullName -split '/')[0] } | Sort-Object -Unique)
    if ($roots.Count -ne 1 -or $roots[0] -ne $packageName) {
        throw "MDL SDK archive root mismatch: $($roots -join ', ')"
    }
}
finally {
    $zip.Dispose()
}

Expand-Archive -LiteralPath $archive -DestinationPath $externalRoot
if (-not (Test-Path -LiteralPath (Join-Path $target "bin\libmdl_sdk.dll"))) {
    throw "MDL SDK extraction incomplete: $target"
}
Write-Host "MDL SDK ready: $target"
