from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from ncls.data.falcor import create_falcor_device, import_falcor, output_buffer, structured_buffer
from ncls.paths import PROJECT_ROOT


MDL_SDK_NAME = "MDL-SDK-2025.0.0-387700.1252-nt-x86-64"


def _sdk_root() -> Path:
    root = PROJECT_ROOT / "external" / MDL_SDK_NAME
    if not (root / "bin" / "libmdl_sdk.dll").is_file():
        pytest.skip("锁定的 MDL SDK binary package 未获取")
    return root


def _compile_diffuse_artifact(sdk_root: Path, output: Path) -> dict[str, object]:
    executable = PROJECT_ROOT / "build" / "mdl-sdk-bridge" / "Release" / "ncls_mdl_sdk_bridge.exe"
    if not executable.is_file():
        pytest.skip("项目 MDL SDK bridge 尚未构建；先运行 scripts/build_mdl_reference.ps1")
    environment = os.environ.copy()
    environment["PATH"] = str(sdk_root / "bin") + os.pathsep + environment.get("PATH", "")
    subprocess.run(
        [
            str(executable),
            "compile",
            "--sdk-root",
            str(sdk_root),
            "--module-root",
            str(PROJECT_ROOT / "tests" / "fixtures" / "mdl"),
            "--material",
            "::constant_diffuse::constant_diffuse",
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.falcor
def test_mdl_sdk_hlsl_compiles_and_matches_lambertian(tmp_path: Path) -> None:
    falcor = import_falcor()
    sdk_root = _sdk_root()
    artifact_dir = tmp_path / "compiled"
    manifest = _compile_diffuse_artifact(sdk_root, artifact_dir)
    assert manifest["schema"] == "ncls.mdl-compiled-artifact@1"
    generated = (artifact_dir / str(manifest["code"])).read_text(encoding="utf-8")
    types = (
        sdk_root
        / "examples"
        / "mdl_sdk"
        / "dxr"
        / "content"
        / "mdl_target_code_types.hlsl"
    ).read_text(encoding="utf-8")
    runtime = (PROJECT_ROOT / "shaders" / "ncls" / "reference_backends" / "mdl_runtime.slangh").read_text(
        encoding="utf-8"
    )
    adapter = (PROJECT_ROOT / "shaders" / "ncls" / "reference_backends" / "mdl_query.slang").read_text(
        encoding="utf-8"
    )
    source = "\n".join(
        (
            "#define MDL_NUM_TEXTURE_RESULTS 16",
            "#define MDL_DF_HANDLE_SLOT_MODE -1",
            "struct NclsMdlRendererState { float3 view_direction; };",
            "#define RENDERER_STATE_TYPE NclsMdlRendererState",
            types,
            runtime,
            generated,
            adapter,
        )
    )

    device = create_falcor_device(falcor)
    desc = falcor.ProgramDesc()
    desc.add_shader_module("ncls_mdl_feasibility").add_string(
        source, tmp_path / "ncls_mdl_feasibility.slang"
    )
    desc.cs_entry("main")
    compute = falcor.ComputePass(device, desc)

    views = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0], [-0.6, 0.0, 0.8, 0.0]],
        dtype=np.float32,
    )
    lights = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.6, 0.8, 0.0], [0.0, 0.8, 0.6, 0.0]],
        dtype=np.float32,
    )
    parameters = manifest["parameters"]
    assert isinstance(parameters, list) and parameters[0]["name"] == "tint"
    tint = np.asarray(parameters[0]["value"], dtype=np.float32)
    argument_block = manifest["argument_block"]
    assert isinstance(argument_block, dict)
    argument_words = np.frombuffer(
        (artifact_dir / str(argument_block["path"])).read_bytes(), dtype=np.uint32
    ).view(np.float32).reshape(-1, 4).copy()
    output = output_buffer(device, falcor, len(views))
    surface_rows = np.zeros((len(views), 4), dtype=np.float32)
    compute.globals.gViews = structured_buffer(device, falcor, views, 16)
    compute.globals.gLights = structured_buffer(device, falcor, lights, 16)
    compute.globals.gPositions = structured_buffer(device, falcor, surface_rows, 16)
    compute.globals.gUv = structured_buffer(device, falcor, surface_rows, 16)
    compute.globals.gMdlArgumentBlock = structured_buffer(device, falcor, argument_words, 16)
    compute.globals.gOutput = output
    compute.globals.gQueryCount = len(views)
    compute.execute(threads_x=len(views))
    actual = output.to_numpy().view(np.float32).reshape(len(views), 4)
    expected = tint[None, :] * np.maximum(lights[:, 2:3], 0.0) / np.pi
    np.testing.assert_allclose(actual[:, :3], expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual[:, 3], np.maximum(lights[:, 2], 0.0) / np.pi, rtol=2e-6, atol=2e-7)
    device.end_frame()


@pytest.mark.falcor
def test_mdl_bridge_rejects_emission_outside_v1_capability(tmp_path: Path) -> None:
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
