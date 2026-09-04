#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ddp_world=""

# Linux multi-GPU DDP entry; each rank uses one remapped CUDA device.
if [[ "${1:-}" == "--gpus" ]]; then
    if [[ "$#" -lt 2 ]]; then
        echo "Usage: $0 --gpus <gpu0,gpu1,...> -- <python args>" >&2
        exit 2
    fi
    gpu_list="$2"
    shift 2
    if [[ "${1:-}" == "--" ]]; then
        shift
    fi
    if [[ "$#" -eq 0 ]]; then
        echo "Usage: $0 --gpus <gpu0,gpu1,...> -- <python args>" >&2
        exit 2
    fi
    if [[ ! "${gpu_list}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
        echo "--gpus must be a comma-separated list of nonnegative physical GPU indices." >&2
        exit 2
    fi
    IFS=',' read -r -a gpu_indices <<< "${gpu_list}"
    declare -A seen=()
    for gpu in "${gpu_indices[@]}"; do
        if [[ -n "${seen[$gpu]:-}" ]]; then
            echo "--gpus contains duplicate GPU index ${gpu}." >&2
            exit 2
        fi
        seen["$gpu"]=1
    done
    ddp_world="${#gpu_indices[@]}"
    if (( ddp_world < 2 )); then
        echo "--gpus requires at least two GPUs for DDP; use single-GPU launcher otherwise." >&2
        exit 2
    fi
    if [[ -n "${NCLS_DDP_GPU_LIST:-}" && "${NCLS_DDP_GPU_LIST}" != "${gpu_list}" ]]; then
        echo "NCLS_DDP_GPU_LIST must not be set independently of --gpus." >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="${gpu_list}"
    export NCLS_DDP_GPU_LIST="${gpu_list}"
    export NCLS_DDP_WORLD_SIZE="${ddp_world}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29517}"
    if [[ "${NCLS_DDP_DEBUG:-}" == "1" ]]; then
        export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
        export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
        export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
        export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
        export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
    fi
    echo "[ddp] GPUs=${gpu_list} world_size=${ddp_world} backend=NCCL" >&2
fi

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
    if [[ -z "${ddp_world}" && ! "${CUDA_VISIBLE_DEVICES}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index." >&2
        exit 1
    fi
    if [[ -n "${ddp_world}" && ! "${CUDA_VISIBLE_DEVICES}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
        echo "DDP CUDA_VISIBLE_DEVICES must be a comma-separated GPU list." >&2
        exit 1
    fi
    if [[ -z "${ddp_world}" && -n "${NCLS_FALCOR_GPU_INDEX:-}" && "${NCLS_FALCOR_GPU_INDEX}" != "${CUDA_VISIBLE_DEVICES}" ]]; then
        echo "NCLS_FALCOR_GPU_INDEX must match CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    if [[ -z "${ddp_world}" ]]; then
        export NCLS_FALCOR_GPU_INDEX="${CUDA_VISIBLE_DEVICES}"
    fi
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

if [[ -n "${ddp_world}" ]]; then
    exec conda run --no-capture-output -n neural-shading \
        torchrun --standalone --nnodes=1 --nproc_per_node="${ddp_world}" \
        -m ncls.ddp_worker "$@"
fi
exec conda run --no-capture-output -n neural-shading python "$@"
