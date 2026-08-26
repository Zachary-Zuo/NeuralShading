#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
falcor_bin="${project_root}/external/Falcor/build/linux-gcc/bin/Release"
falcor_module="${falcor_bin}/python"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Use scripts/run_falcor_python.ps1 on Windows." >&2
    exit 1
fi
if ! compgen -G "${falcor_module}/falcor/falcor_ext*.so" >/dev/null; then
    echo "FalcorPython Release build was not found. Run scripts/build_falcor_python_linux.sh first." >&2
    exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH." >&2
    exit 1
fi

export PATH="${falcor_bin}:${PATH}"
export LD_LIBRARY_PATH="${falcor_bin}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${project_root}/src:${falcor_module}${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n neural-shading python "$@"
