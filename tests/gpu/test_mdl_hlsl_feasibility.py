from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import MdlSdkCompilerBridge
from ncls.references.query import ReferenceQueryDispatcher


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


@pytest.mark.falcor
def test_mdl_punched_atlas_preserves_rgba16_before_cutout_fail_closed(
    tmp_path: Path,
) -> None:
    module_root = (
        PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"
    )
    if not (module_root / "vMaterials_2/Leather/Suede_Leather.mdl").is_file():
        pytest.skip("vMaterials 2.4.0 source pack is not available")
    bridge = MdlSdkCompilerBridge(module_root)
    artifact = bridge._run(  # noqa: SLF001 - bridge capability integration test
        "::vMaterials_2::Leather::Suede_Leather",
        "Suede_Leather_Punched",
        {},
        tmp_path / "punched",
    )
    atlases = [
        item for item in artifact.manifest["textures"] if item["pixel_type"] == "Rgba_16"
    ]
    assert len(atlases) == 1
    atlas = atlases[0]
    assert atlas["data"] is not None
    assert (artifact.root / atlas["data"]).stat().st_size == 1024 * 1024 * 4 * 2
    with pytest.raises(ValueError, match="geometry.cutout_opacity"):
        artifact.require_runtime_supported()


@pytest.mark.falcor
def test_generic_falcor_texture_binder_accepts_rgba16_unorm() -> None:
    falcor = pytest.importorskip("falcor")
    from ncls.references.falcor import create_falcor_device

    dispatcher = object.__new__(ReferenceQueryDispatcher)
    dispatcher._falcor = falcor
    dispatcher._device = create_falcor_device(falcor)
    values = np.asarray(
        [[[0, 1, 257, 65535], [65535, 32768, 1024, 9]]], dtype=np.uint16
    )
    texture = dispatcher._texture_payload(  # noqa: SLF001 - typed binder contract test
        "rgba16",
        values.tobytes(),
        {
            "kind": "texture2d",
            "dtype": "uint16",
            "shape": [1, 2, 4],
            "stride": 8,
            "alignment": 2,
            "format": "rgba16-unorm",
            "color_space": "linear",
            "usage": "gFixture",
        },
    )
    assert texture is not None
    assert texture.format == falcor.ResourceFormat.RGBA16Unorm
