param(
    [switch]$OpenPBR,
    [switch]$MaterialX,
    [switch]$All
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$externalRoot = Join-Path $projectRoot "external"
New-Item -ItemType Directory -Force -Path $externalRoot | Out-Null
$externalRoot = (Resolve-Path -LiteralPath $externalRoot).Path

function Sync-PinnedRepository {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Revision
    )

    $target = Join-Path $externalRoot $Name
    $parent = [System.IO.Path]::GetFullPath((Split-Path -Parent $target))
    if ($parent -ne $externalRoot) {
        throw "拒绝在 external/ 之外创建上游目录：$target"
    }
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
            throw "目标已存在但不是 Git clone：$target"
        }
        $dirty = git -C $target status --short
        if ($LASTEXITCODE -ne 0 -or $dirty) {
            throw "上游工作树不是干净状态：$target"
        }
    }
    else {
        git clone --filter=blob:none --no-checkout $Url $target
        if ($LASTEXITCODE -ne 0) { throw "clone 失败：$Url" }
    }

    git -C $target fetch --depth 1 origin $Revision
    if ($LASTEXITCODE -ne 0) { throw "fetch 失败：$Name@$Revision" }
    git -C $target checkout --detach $Revision
    if ($LASTEXITCODE -ne 0) { throw "checkout 失败：$Name@$Revision" }
    $actual = git -C $target rev-parse HEAD
    if ($actual -ne $Revision) {
        throw "上游提交不匹配：$Name expected=$Revision actual=$actual"
    }
    if (git -C $target status --short) {
        throw "checkout 后上游工作树不干净：$target"
    }
    Write-Host "$Name locked at $actual"
}

if (-not ($OpenPBR -or $MaterialX -or $All)) {
    throw "至少指定 -OpenPBR、-MaterialX 或 -All"
}

if ($OpenPBR -or $All) {
    Sync-PinnedRepository -Name "OpenPBR" `
        -Url "git@github.com:AcademySoftwareFoundation/OpenPBR.git" `
        -Revision "f8d6d947dfae4c9b599965a86c22826ea7a8dbfb"
    Sync-PinnedRepository -Name "openpbr-bsdf" `
        -Url "git@github.com:adobe/openpbr-bsdf.git" `
        -Revision "9edf806740d2140846d9bef76e4342fc458e2ef5"
    Sync-PinnedRepository -Name "glm" `
        -Url "git@github.com:g-truc/glm.git" `
        -Revision "0af55ccecd98d4e5a8d1fad7de25ba429d60e863"
}

if ($MaterialX -or $All) {
    $materialXRevision = "270b5cf2ae2be24a3b6ef4b0569f1c93038dda1d"
    Sync-PinnedRepository -Name "MaterialX" `
        -Url "git@github.com:AcademySoftwareFoundation/MaterialX.git" `
        -Revision $materialXRevision

    $materialXRoot = Join-Path $externalRoot "MaterialX"
    $nanoGuiPath = "source/MaterialXView/NanoGUI"
    $nanoGuiRevision = "6452dd6944d2ba5c0c9bc0042a1894f703ce1ace"
    git -C $materialXRoot submodule init -- $nanoGuiPath
    if ($LASTEXITCODE -ne 0) { throw "NanoGUI submodule init 失败" }
    git -C $materialXRoot config "submodule.$nanoGuiPath.url" "git@github.com:mitsuba-renderer/nanogui.git"
    git -C $materialXRoot submodule update --depth 1 -- $nanoGuiPath
    if ($LASTEXITCODE -ne 0) { throw "NanoGUI submodule update 失败" }
    $actualNanoGuiRevision = git -C (Join-Path $materialXRoot $nanoGuiPath) rev-parse HEAD
    if ($actualNanoGuiRevision -ne $nanoGuiRevision) {
        throw "NanoGUI 提交不匹配：expected=$nanoGuiRevision actual=$actualNanoGuiRevision"
    }

    $nanoGuiRoot = Join-Path $materialXRoot $nanoGuiPath
    git -C $nanoGuiRoot submodule init
    git -C $nanoGuiRoot config "submodule.ext/glfw.url" "git@github.com:wjakob/glfw.git"
    git -C $nanoGuiRoot config "submodule.ext/nanobind.url" "git@github.com:wjakob/nanobind.git"
    git -C $nanoGuiRoot config "submodule.ext/nanovg.url" "git@github.com:wjakob/nanovg.git"
    git -C $nanoGuiRoot config "submodule.ext/nanovg_metal.url" "git@github.com:wjakob/nanovg_metal.git"
    git -C $nanoGuiRoot submodule update --init --depth 1
    if ($LASTEXITCODE -ne 0) { throw "NanoGUI dependency submodule update 失败" }

    $nanoBindRoot = Join-Path $nanoGuiRoot "ext/nanobind"
    git -C $nanoBindRoot submodule init
    git -C $nanoBindRoot config "submodule.ext/robin_map.url" "git@github.com:Tessil/robin-map.git"
    git -C $nanoBindRoot submodule update --init --depth 1
    if ($LASTEXITCODE -ne 0) { throw "NanoBind dependency submodule update 失败" }

    $incompleteSubmodules = git -C $nanoGuiRoot submodule status --recursive | Where-Object { $_ -match '^[+-U]' }
    if ($incompleteSubmodules) {
        throw "MaterialX viewer 子模块未锁定：$($incompleteSubmodules -join ', ')"
    }
    if (git -C $materialXRoot status --short) {
        throw "初始化 viewer 子模块后 MaterialX 工作树不干净"
    }
    Write-Host "NanoGUI locked at $actualNanoGuiRevision"
}
