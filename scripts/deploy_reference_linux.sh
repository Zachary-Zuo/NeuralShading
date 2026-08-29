#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
report_path="${NCLS_DEPLOY_REPORT:-${project_root}/artifacts/deployment/reference-linux/${run_id}/report.json}"
current_step="preflight"
environment_ready=0
deployment_complete=0
environment_status="reused"
falcor_status="fresh"
mdl_status="fresh"
fetch_report="${project_root}/build/reference-backend/fetch-linux.json"

write_report() {
    local status="$1"
    if [[ "${environment_ready}" == "1" ]]; then
        PYTHONPATH="${project_root}/src" conda run --no-capture-output -n neural-shading \
            python "${project_root}/tools/reference/reference_backend_deploy.py" report \
            "${report_path}" \
            --deployment-status "${status}" \
            --steps-file "${fetch_report}" \
            --step "environment=${environment_status}=neural-shading" \
            --step "falcor-build=${falcor_status}=FalcorPython" \
            --step "mdl-provider-build=${mdl_status}=ncls_mdl_sdk_bridge" \
            --step "current-step=${status}=${current_step}" || true
    fi
}

finish_report() {
    if [[ "${deployment_complete}" != "1" ]]; then
        write_report "failed"
        echo "Reference backend deployment failed during ${current_step}; report: ${report_path}" >&2
    fi
}
trap finish_report EXIT

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "deploy_reference_linux.sh requires native Linux." >&2
    exit 1
fi
for command in conda git gcc g++ cmake nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Required command is missing: ${command}" >&2
        exit 1
    fi
done

current_step="conda-environment"
if ! conda env list | awk '{print $1}' | grep -Fxq neural-shading; then
    environment_status="fresh"
    conda env create -f "${project_root}/environment.yml"
else
    environment_ready=1
    conda env update -n neural-shading -f "${project_root}/environment.yml"
fi
environment_ready=1
conda run --no-capture-output -n neural-shading python -m pip install \
    -r "${project_root}/requirements-torch-cu128.txt"
# Runtime-compiled CUDA 12.8 PTX needs its matching user-space compatibility
# layer when the host kernel module exposes an older CUDA generation.
conda install -n neural-shading "defaults::cuda-compat=12.8.1" -y

current_step="fetch-locked-dependencies"
mkdir -p "${project_root}/build/reference-backend"
PYTHONPATH="${project_root}/src" conda run --no-capture-output -n neural-shading \
    python "${project_root}/tools/reference/reference_backend_deploy.py" fetch \
    --platform-id linux-x86_64@1 \
    --project-root "${project_root}" \
    --output "${fetch_report}"

current_step="falcor-setup"
packman_bin="${project_root}/external/Falcor/tools/packman/packman"
if [[ -f "${packman_bin}" && ! -x "${packman_bin}" ]]; then
    # Windows copies can lose the executable bit on this upstream shell entry point.
    chmod +x "${packman_bin}"
fi
if [[ ! -x "${project_root}/external/Falcor/tools/.packman/cmake/bin/cmake" ]]; then
    (cd "${project_root}/external/Falcor" && bash setup.sh)
fi
if compgen -G "${project_root}/external/Falcor/build/linux-gcc/bin/Release/python/falcor/falcor_ext*.so" >/dev/null; then
    falcor_status="reused"
fi
current_step="falcor-build"
bash "${project_root}/scripts/build_falcor_python_linux.sh"

if [[ -x "${project_root}/build/mdl-sdk-bridge/Release/ncls_mdl_sdk_bridge" ]]; then
    mdl_status="reused"
fi
current_step="mdl-provider-build"
bash "${project_root}/scripts/build_mdl_program_provider.sh"

current_step="asset-free-probe"
bash "${project_root}/scripts/run_falcor_python.sh" -m ncls.cli reference probe

current_step="complete"
write_report "ready"
deployment_complete=1
echo "Reference backend deployment is ready; assets were not managed. Report: ${report_path}"
