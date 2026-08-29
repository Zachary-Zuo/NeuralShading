#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Use scripts/run_falcor_python.ps1 on Windows." >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH." >&2
    exit 1
fi
if [[ -n "${NCLS_FALCOR_GPU_INDEX:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES instead of NCLS_FALCOR_GPU_INDEX." >&2
    exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
        echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index." >&2
        exit 1
    fi
    if [[ -n "${NCLS_FALCOR_GPU_INDEX:-}" && "${NCLS_FALCOR_GPU_INDEX}" != "${CUDA_VISIBLE_DEVICES}" ]]; then
        echo "NCLS_FALCOR_GPU_INDEX must match CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    export NCLS_FALCOR_GPU_INDEX="${CUDA_VISIBLE_DEVICES}"
fi

mapfile -t layout < <(
    PYTHONPATH="${project_root}/src" conda run --no-capture-output -n neural-shading python -c '
import sys

from ncls.paths import PROJECT_ROOT
from ncls.references.backend_manifest import load_reference_backend_manifest
m = load_reference_backend_manifest()
p = m.for_platform("linux-x86_64@1")
print(PROJECT_ROOT / p.falcor.runtime_library_root)
print(PROJECT_ROOT / p.falcor.python_module_root)
print(PROJECT_ROOT / p.falcor.python_module_root / p.falcor.python_extension)
print(sys.prefix)
'
)
falcor_bin="${layout[0]}"
falcor_module="${layout[1]}"
falcor_extension_pattern="${layout[2]}"
conda_prefix="${layout[3]}"
cuda_compat="${conda_prefix}/cuda-compat"

if ! compgen -G "${falcor_extension_pattern}" >/dev/null; then
    echo "FalcorPython Release build was not found. Run scripts/deploy_reference_linux.sh first." >&2
    exit 1
fi

export PATH="${falcor_bin}:${PATH}"
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
driver_major="${driver_version%%.*}"
if [[ -d "${cuda_compat}" && "${driver_major}" =~ ^[0-9]+$ && "${driver_major}" -lt 570 ]]; then
    export LD_LIBRARY_PATH="${falcor_bin}:${cuda_compat}:${conda_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    export LD_LIBRARY_PATH="${falcor_bin}:${conda_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${project_root}/src:${falcor_module}${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n neural-shading python "$@"
