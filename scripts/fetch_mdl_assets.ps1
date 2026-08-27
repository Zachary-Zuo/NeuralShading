param(
    [switch]$VMaterials2,
    [switch]$AcceptNvidiaOmniverseTerms
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $VMaterials2) {
    throw "Only -VMaterials2 is currently supported."
}
if (-not $AcceptNvidiaOmniverseTerms) {
    throw "Pass -AcceptNvidiaOmniverseTerms explicitly. No network or file write has occurred."
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$assetsRoot = Join-Path $projectRoot "assets"
$packageRoot = Join-Path $assetsRoot "source-materials\mdl-vmaterials2"
$target = Join-Path $packageRoot "2.4.0"
$archiveName = "vMaterials_2_4_0_NVD@020240.zip"
$archive = Join-Path $assetsRoot $archiveName
$partialArchive = "$archive.partial"
$partialTarget = Join-Path $packageRoot ".2.4.0.partial"
$url = "https://d4i3qtqj3r0z5.cloudfront.net/vMaterials_2_4_0_NVD%40020240.zip"
$expectedSize = 2220534625
$expectedSha256 = "ab8116e1944c03ae622b2637939510eca9c522a07fd701cf91948fa54e194204"
$expectedEtag = '"719019a683a90c3081489351984b0735-265"'

function Assert-ContainedPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $candidate = [System.IO.Path]::GetFullPath($Path)
    if (-not $candidate.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escaped the configured root: candidate=$candidate root=$rootPath"
    }
}

function Assert-Archive {
    param([Parameter(Mandatory = $true)][string]$Path)
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $expectedSize) {
        throw "vMaterials archive size mismatch: expected=$expectedSize actual=$($item.Length)"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedSha256) {
        throw "vMaterials archive SHA-256 mismatch: expected=$expectedSha256 actual=$actual"
    }
}

function Assert-Package {
    param([Parameter(Mandatory = $true)][string]$Path)
    $packageInfo = Join-Path $Path "PACKAGE-INFO.yaml"
    $materials = Join-Path $Path "Materials\vMaterials_2"
    $licenses = Join-Path $Path "PACKAGE-LICENSES"
    if (-not (Test-Path -LiteralPath $packageInfo -PathType Leaf) -or
        -not (Test-Path -LiteralPath $materials -PathType Container) -or
        -not (Test-Path -LiteralPath $licenses -PathType Container)) {
        throw "vMaterials 2.4.0 target exists but is incomplete: $Path"
    }
    $info = Get-Content -LiteralPath $packageInfo -Encoding UTF8 -Raw
    if ($info -notmatch '(?m)^Package\s*:\s*vMaterials_2_4_0_NVD\s*$' -or
        $info -notmatch '(?m)^Version\s*:\s*20240\s*$') {
        throw "PACKAGE-INFO.yaml does not identify vMaterials 2.4.0: $packageInfo"
    }
}

Assert-ContainedPath -Root $projectRoot -Path $assetsRoot
Assert-ContainedPath -Root $assetsRoot -Path $target
Assert-ContainedPath -Root $assetsRoot -Path $archive

if (Test-Path -LiteralPath $target) {
    Assert-Package -Path $target
    if (Test-Path -LiteralPath $archive) {
        Assert-Archive -Path $archive
    }
    Write-Host "vMaterials 2.4.0 already available: $target"
    Write-Host "Frozen archive identity: size=$expectedSize sha256=$expectedSha256 etag=$expectedEtag"
    exit 0
}

if (Test-Path -LiteralPath $partialTarget) {
    throw "Unverified partial extraction exists: $partialTarget"
}
New-Item -ItemType Directory -Path $assetsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $archive)) {
    if (Test-Path -LiteralPath $partialArchive) {
        throw "Unverified partial download exists: $partialArchive"
    }
    curl.exe -L --fail --progress-bar --output $partialArchive $url
    if ($LASTEXITCODE -ne 0) {
        throw "vMaterials 2.4.0 download failed: $url"
    }
    Assert-Archive -Path $partialArchive
    Move-Item -LiteralPath $partialArchive -Destination $archive
}
Assert-Archive -Path $archive

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
try {
    $entryNames = New-Object System.Collections.Generic.HashSet[string]
    foreach ($entry in $zip.Entries) {
        $name = $entry.FullName.Replace('\', '/')
        [void]$entryNames.Add($name)
        $segments = $name.Split('/')
        $unixType = (($entry.ExternalAttributes -shr 16) -band 0xF000)
        if ([System.IO.Path]::IsPathRooted($name) -or $name.StartsWith('/') -or
            $segments -contains '..' -or $unixType -eq 0xA000) {
            throw "vMaterials archive contains an unsafe entry: $($entry.FullName)"
        }
    }
    foreach ($required in @("PACKAGE-INFO.yaml", "Materials/", "PACKAGE-LICENSES/")) {
        if (-not $entryNames.Contains($required)) {
            throw "vMaterials archive is missing required entry: $required"
        }
    }
}
finally {
    $zip.Dispose()
}

[System.IO.Compression.ZipFile]::ExtractToDirectory($archive, $partialTarget)
Assert-Package -Path $partialTarget
Move-Item -LiteralPath $partialTarget -Destination $target
Write-Host "vMaterials 2.4.0 ready: $target"
Write-Host "Frozen archive identity: size=$expectedSize sha256=$expectedSha256 etag=$expectedEtag"
