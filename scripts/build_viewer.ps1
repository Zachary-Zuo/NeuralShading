param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [switch]$Run,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ViewerArgs
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$falcorRoot = Join-Path $projectRoot "external\Falcor"
$overlayPatch = Join-Path $projectRoot "patches\falcor-viewer-overlay.patch"
$cmake = Join-Path $falcorRoot "tools\.packman\cmake\bin\cmake.exe"
$expectedCommit = "9dc819c162b2070335c65060436041690b7937f8"

if (-not (Test-Path -LiteralPath $cmake -PathType Leaf)) {
    throw "Falcor packman CMake was not found: $cmake"
}
if ((& git -C $falcorRoot rev-parse HEAD).Trim() -ne $expectedCommit) {
    throw "external/Falcor is not at the locked commit $expectedCommit"
}
if ((& git -C $falcorRoot status --porcelain).Count -ne 0) {
    throw "external/Falcor must be clean before applying the viewer overlay"
}

& git -C $falcorRoot apply --check $overlayPatch
if ($LASTEXITCODE -ne 0) { throw "Falcor viewer overlay no longer applies cleanly" }
& git -C $falcorRoot apply $overlayPatch
if ($LASTEXITCODE -ne 0) { throw "Failed to apply Falcor viewer overlay" }

try {
    Push-Location $falcorRoot
    try {
        & $cmake --preset windows-vs2022
        if ($LASTEXITCODE -ne 0) { throw "Falcor CMake configure failed" }
        & $cmake --build "build\windows-vs2022" --config $Configuration --target NclsViewer --parallel
        if ($LASTEXITCODE -ne 0) { throw "NclsViewer build failed" }
    }
    finally {
        Pop-Location
    }
}
finally {
    & git -C $falcorRoot apply --reverse $overlayPatch
    if ($LASTEXITCODE -ne 0) {
        throw "The build finished, but the temporary Falcor overlay could not be reversed"
    }
}

if ((& git -C $falcorRoot status --porcelain).Count -ne 0) {
    throw "external/Falcor is dirty after the overlay was reversed"
}

# Falcor's generated version header observes the temporary overlay while NclsViewer is built.
# Refresh only Falcor.dll once more from the now-clean source tree so runtime provenance is truthful.
Push-Location $falcorRoot
try {
    & $cmake --build "build\windows-vs2022" --config $Configuration --target Falcor --parallel
    if ($LASTEXITCODE -ne 0) { throw "Failed to refresh clean Falcor runtime metadata" }
}
finally {
    Pop-Location
}

$viewer = Join-Path $falcorRoot "build\windows-vs2022\bin\$Configuration\NclsViewer.exe"
if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
    throw "NclsViewer executable was not produced: $viewer"
}
Write-Output $viewer

if ($Run) {
    Push-Location $projectRoot
    try {
        & $viewer @ViewerArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
