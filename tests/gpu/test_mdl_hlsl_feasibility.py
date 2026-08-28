from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from ncls.paths import PROJECT_ROOT


MDL_SDK_NAME = "MDL-SDK-2025.0.0-387700.1252-nt-x86-64"


def _sdk_root() -> Path:
    root = PROJECT_ROOT / "external" / MDL_SDK_NAME
    if not (root / "bin" / "libmdl_sdk.dll").is_file():
        pytest.skip("锁定的 MDL SDK binary package 未获取")
    return root


@pytest.mark.falcor
def test_mdl_bridge_rejects_emission_outside_v1_capability(tmp_path: Path) -> None:
    """Bridge 的 source capability 边界独立于统一 query dispatcher。"""

    sdk_root = _sdk_root()
    executable = PROJECT_ROOT / "build/mdl-sdk-bridge/Release/ncls_mdl_sdk_bridge.exe"
    if not executable.is_file():
        pytest.skip("项目 MDL SDK bridge 尚未构建；先运行 scripts/build_mdl_reference.ps1")
    environment = os.environ.copy()
    environment["PATH"] = str(sdk_root / "bin") + os.pathsep + environment.get("PATH", "")
    result = subprocess.run(
        [
            str(executable),
            "compile",
            "--sdk-root",
            str(sdk_root),
            "--module-root",
            str(PROJECT_ROOT / "tests/fixtures/mdl"),
            "--material",
            "::emissive_unsupported::emissive_unsupported",
            "--output-dir",
            str(tmp_path / "emissive"),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does not support surface emission" in result.stderr
