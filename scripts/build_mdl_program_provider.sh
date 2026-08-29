#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This helper builds the Linux MDL program provider only." >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH." >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake was not found on PATH." >&2
    exit 1
fi

mapfile -t layout < <(
    PYTHONPATH="${project_root}/src" conda run --no-capture-output -n neural-shading python -c '
from ncls.paths import PROJECT_ROOT
from ncls.references.backend_manifest import load_reference_backend_manifest
m = load_reference_backend_manifest()
p = m.for_platform("linux-x86_64@1")
print(PROJECT_ROOT / p.mdl_sdk.archive.root)
print(PROJECT_ROOT / "external/stb")
print(PROJECT_ROOT / "tools/reference/mdl_sdk_bridge")
print(PROJECT_ROOT / "build/mdl-sdk-bridge")
print(PROJECT_ROOT / p.mdl_bridge.executable)
'
)

sdk_root="${layout[0]}"
stb_root="${layout[1]}"
source_root="${layout[2]}"
build_root="${layout[3]}"
executable="${layout[4]}"

if [[ ! -f "${sdk_root}/include/mi/mdl_sdk.h" ]]; then
    echo "Pinned MDL SDK headers are missing: ${sdk_root}" >&2
    exit 1
fi
if [[ ! -f "${stb_root}/stb_image.h" ]]; then
    echo "Pinned stb source is missing: ${stb_root}" >&2
    exit 1
fi

parallel="${NCLS_BUILD_JOBS:-$(nproc)}"
cmake -S "${source_root}" -B "${build_root}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${build_root}/Release" \
    -DMDL_SDK_ROOT="${sdk_root}" \
    -DSTB_ROOT="${stb_root}"
cmake --build "${build_root}" --config Release --parallel "${parallel}"

if [[ ! -x "${executable}" ]]; then
    echo "MDL program provider build did not produce ${executable}." >&2
    exit 1
fi
echo "MDL program provider is available at ${executable}."
