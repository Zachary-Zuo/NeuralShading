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

mapfile -t layout < <(
    PYTHONPATH="${project_root}/src" conda run --no-capture-output -n neural-shading python -c '
from ncls.paths import PROJECT_ROOT
from ncls.references.backend_manifest import load_reference_backend_manifest
m = load_reference_backend_manifest()
p = m.for_platform("linux-x86_64@1")
print(PROJECT_ROOT / p.falcor.runtime_library_root)
print(PROJECT_ROOT / p.falcor.python_module_root)
print(PROJECT_ROOT / p.falcor.python_module_root / p.falcor.python_extension)
'
)
falcor_bin="${layout[0]}"
falcor_module="${layout[1]}"
falcor_extension_pattern="${layout[2]}"

if ! compgen -G "${falcor_extension_pattern}" >/dev/null; then
    echo "FalcorPython Release build was not found. Run scripts/deploy_reference_linux.sh first." >&2
    exit 1
fi

export PATH="${falcor_bin}:${PATH}"
export LD_LIBRARY_PATH="${falcor_bin}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${project_root}/src:${falcor_module}${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n neural-shading python "$@"
