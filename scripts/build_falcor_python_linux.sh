#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
falcor_root="${project_root}/external/Falcor"
expected_commit="9dc819c162b2070335c65060436041690b7937f8"
cmake_bin="${falcor_root}/tools/.packman/cmake/bin/cmake"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This launcher only builds the Linux FalcorPython target." >&2
    exit 1
fi

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

falcor_module_dir="${falcor_root}/build/linux-gcc/bin/Release/python/falcor"
if ! compgen -G "${falcor_module_dir}/falcor_ext*.so" >/dev/null; then
    echo "FalcorPython build completed without producing falcor_ext*.so in ${falcor_module_dir}." >&2
    exit 1
fi
if [[ -n "$(git -C "${falcor_root}" status --porcelain)" ]]; then
    echo "Falcor source tree became dirty during the build." >&2
    exit 1
fi

echo "FalcorPython Linux/Vulkan build is available under external/Falcor/build/linux-gcc/bin/Release."
