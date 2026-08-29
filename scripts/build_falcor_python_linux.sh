#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
falcor_root="${project_root}/external/Falcor"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This launcher only builds the Linux FalcorPython target." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH." >&2
    exit 1
fi

mapfile -t layout < <(
    PYTHONPATH="${project_root}/src" conda run --no-capture-output -n neural-shading python -c '
from ncls.paths import PROJECT_ROOT
from ncls.references.backend_manifest import load_reference_backend_manifest
m = load_reference_backend_manifest()
p = m.for_platform("linux-x86_64@1")
source = next(value for value in m.source_providers if value.provider_id == m.execution_provider)
print(m.falcor_revision)
print(PROJECT_ROOT / source.path)
print(PROJECT_ROOT / p.falcor.build_root)
print(PROJECT_ROOT / p.falcor.python_module_root / p.falcor.python_extension)
'
)
expected_commit="${layout[0]}"
falcor_root="${layout[1]}"
falcor_build_root="${layout[2]}"
falcor_extension_pattern="${layout[3]}"
cmake_bin="${falcor_root}/tools/.packman/cmake/bin/cmake"

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
        echo "Warning: locked Falcor 8.0 only documents experimental Ubuntu 22.04 support; detected ${PRETTY_NAME:-unknown Linux}." >&2
    fi
fi

actual_commit="$(git -C "${falcor_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "Falcor commit mismatch: expected ${expected_commit}, got ${actual_commit}." >&2
    exit 1
fi
if [[ -n "$(git -C "${falcor_root}" status --porcelain)" ]]; then
    echo "external/Falcor must be clean before configuring the project build." >&2
    exit 1
fi
if [[ ! -x "${cmake_bin}" ]]; then
    echo "Falcor Packman dependencies are missing. Run external/Falcor/setup.sh first." >&2
    exit 1
fi

parallel="${NCLS_BUILD_JOBS:-$(nproc)}"
cd "${falcor_root}"
"${cmake_bin}" --preset linux-gcc
"${cmake_bin}" --build --preset linux-gcc-release --target FalcorPython --parallel "${parallel}"

if ! compgen -G "${falcor_extension_pattern}" >/dev/null; then
    echo "FalcorPython build completed without producing ${falcor_extension_pattern}." >&2
    exit 1
fi
if [[ -n "$(git -C "${falcor_root}" status --porcelain)" ]]; then
    echo "Falcor source tree became dirty during the build." >&2
    exit 1
fi

echo "FalcorPython Linux/Vulkan build is available under ${falcor_build_root}."
