#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_falcor_python.sh --gpus <gpu0,gpu1,...> -- <python args>

Starts one independent single-GPU process per physical GPU. Arguments may use
the literal {gpu} token; it is replaced with that process's physical index.
Use distinct output/checkpoint paths when running training on several GPUs.
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

declare -a pids=()
declare -a logs=()
status=0
for gpu in "${gpu_indices[@]}"; do
    log_path=""
    for arg in "$@"; do
        if [[ "$arg" == *"{gpu}"* ]]; then
            candidate="${arg//\{gpu\}/${gpu}}"
            if [[ "$candidate" == *.log || "$candidate" == */logs/* ]]; then
                log_path="$candidate"
                break
            fi
        fi
    done
    if [[ -z "$log_path" ]]; then
        log_path="${project_root}/artifacts/multi-gpu/gpu${gpu}.log"
    fi
    mkdir -p "$(dirname "$log_path")"

    declare -a child_args=()
    for arg in "$@"; do
        child_args+=("${arg//\{gpu\}/${gpu}}")
    done
    echo "[multi-gpu] GPU${gpu}: ${child_args[*]} (log=${log_path})" >&2
    (
        CUDA_VISIBLE_DEVICES="${gpu}" \
            bash "${project_root}/scripts/run_falcor_python.sh" "${child_args[@]}"
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
    logs+=("${log_path}")
done

for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[multi-gpu] GPU${gpu_indices[$index]} failed; see ${logs[$index]}" >&2
        status=1
    else
        echo "[multi-gpu] GPU${gpu_indices[$index]} completed; log=${logs[$index]}" >&2
    fi
done

exit "$status"
