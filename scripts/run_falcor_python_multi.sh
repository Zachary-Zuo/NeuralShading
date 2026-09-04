#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Compatibility-only thin forwarder. All validation and distributed launch policy lives
# in the one public launcher so it cannot drift across two shell entry points.
exec bash "${project_root}/scripts/run_falcor_python.sh" "$@"
