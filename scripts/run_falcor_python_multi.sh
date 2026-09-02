#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_falcor_python.sh --gpus <gpu0,gpu1,...> -- <python args>

Starts one torchrun/NCCL data-parallel job. CUDA_VISIBLE_DEVICES remains the
physical GPU list; each rank receives one remapped cuda:<LOCAL_RANK> device.
Only rank 0 writes checkpoint, metrics, summary and review files.
EOF
}

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The multi-GPU launcher is Linux-only." >&2
    exit 1
fi

if [[ "$#" -lt 2 || "$1" != "--gpus" ]]; then
    usage
    exit 2
fi

gpu_list="$2"
shift 2
if [[ "${1:-}" == "--" ]]; then
    shift
fi
if [[ "$#" -eq 0 ]]; then
    usage
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

world_size="${#gpu_indices[@]}"
if (( world_size < 2 )); then
    echo "--gpus requires at least two GPUs for DDP; use single-GPU launcher otherwise." >&2
    exit 2
fi
if [[ -n "${NCLS_DDP_GPU_LIST:-}" && "${NCLS_DDP_GPU_LIST}" != "${gpu_list}" ]]; then
    echo "NCLS_DDP_GPU_LIST must not be set independently of --gpus." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export NCLS_DDP_GPU_LIST="${gpu_list}"
export NCLS_DDP_WORLD_SIZE="${world_size}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29517}"
echo "[ddp] GPUs=${gpu_list} world_size=${world_size} backend=NCCL" >&2

exec "${project_root}/scripts/run_falcor_python.sh" --ddp "${world_size}" -- "$@"
