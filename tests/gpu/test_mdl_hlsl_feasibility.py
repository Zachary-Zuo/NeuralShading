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


def _common_mdl_source(sdk_root: Path, artifact_dir: Path, manifest: dict[str, object]) -> str:
    generated = (artifact_dir / str(manifest["code"])).read_text(encoding="utf-8")
    types = (
        sdk_root
        / "examples"
        / "mdl_sdk"
        / "dxr"
        / "content"
        / "mdl_target_code_types.hlsl"
    ).read_text(encoding="utf-8")
    runtime = (PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh").read_text(
        encoding="utf-8"
    )
    return "\n".join(
        (
            "#define MDL_NUM_TEXTURE_RESULTS 16",
            "#define MDL_DF_HANDLE_SLOT_MODE -1",
            "struct NclsMdlRendererState { float3 view_direction; };",
            "#define RENDERER_STATE_TYPE NclsMdlRendererState",
            types,
            runtime,
            generated,
        )
    )


def _argument_rows(artifact_dir: Path, manifest: dict[str, object]) -> np.ndarray:
    argument_block = manifest["argument_block"]
    assert isinstance(argument_block, dict)
    payload = (artifact_dir / str(argument_block["path"])).read_bytes()
    padded_size = max(16, ((len(payload) + 15) // 16) * 16)
    return (
        np.frombuffer(payload + bytes(padded_size - len(payload)), dtype=np.uint32)
        .view(np.float32)
        .reshape(-1, 4)
        .copy()
    )


@pytest.mark.falcor
def test_mdl_sdk_hlsl_compiles_and_matches_lambertian(tmp_path: Path) -> None:
    falcor = import_falcor()
    sdk_root = _sdk_root()
    artifact_dir = tmp_path / "compiled"
    manifest = _compile_diffuse_artifact(sdk_root, artifact_dir)
    assert manifest["schema"] == "ncls.mdl-compiled-artifact@1"
    adapter = (PROJECT_ROOT / "shaders" / "ncls" / "reference_backends" / "mdl_query.slang").read_text(
        encoding="utf-8"
    )
    source = _common_mdl_source(sdk_root, artifact_dir, manifest) + "\n" + adapter

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
    argument_words = _argument_rows(artifact_dir, manifest)
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
def test_mdl_viewer_adapter_matches_formal_query_and_sample_on_gpu(tmp_path: Path) -> None:
    falcor = import_falcor()
    sdk_root = _sdk_root()
    artifact_dir = tmp_path / "compiled"
    manifest = _compile_diffuse_artifact(sdk_root, artifact_dir)
    common = _common_mdl_source(sdk_root, artifact_dir, manifest)
    formal_adapter = (
        PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_query.slang"
    ).read_text(encoding="utf-8")
    viewer_adapter = (PROJECT_ROOT / "apps/viewer/shaders/MdlViewerAdapter.slang").read_text(
        encoding="utf-8"
    )
    viewer_kernel = r"""
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gLights;
StructuredBuffer<float4> gPositions;
StructuredBuffer<float4> gUv;
RWStructuredBuffer<float4> gOutput;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gQueryCount)
        return;
    float pdf = 0.0;
    const float3 value = nclsMdlEvaluateSurface(
        gPositions[index].xyz,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        gUv[index].xy,
        gViews[index].xyz,
        gLights[index].xyz,
        pdf);
    gOutput[index] = float4(value, pdf);
}
"""
    sample_kernel = r"""
StructuredBuffer<float4> gViews;
StructuredBuffer<float4> gPositions;
StructuredBuffer<float4> gUv;
StructuredBuffer<float4> gXi;
RWStructuredBuffer<float4> gDirectionPdf;
RWStructuredBuffer<float4> gWeightValid;
uniform uint gQueryCount;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gQueryCount)
        return;
    const NclsMdlSample sample = nclsMdlSampleSurface(
        gPositions[index].xyz,
        float3(0.0, 0.0, 1.0),
        float3(1.0, 0.0, 0.0),
        gUv[index].xy,
        gViews[index].xyz,
        gXi[index]);
    gDirectionPdf[index] = float4(sample.directionWorld, sample.pdf);
    gWeightValid[index] = float4(sample.weight, float(sample.valid));
}
"""

    device = create_falcor_device(falcor)

    def make_pass(module_name: str, source: str):
        desc = falcor.ProgramDesc()
        desc.add_shader_module(module_name).add_string(source, tmp_path / f"{module_name}.slang")
        desc.cs_entry("main")
        return falcor.ComputePass(device, desc)

    formal = make_pass("ncls_mdl_formal_query", common + "\n" + formal_adapter)
    viewer = make_pass("ncls_mdl_viewer_adapter", common + "\n" + viewer_adapter + "\n" + viewer_kernel)
    sampler = make_pass("ncls_mdl_viewer_sampler", common + "\n" + viewer_adapter + "\n" + sample_kernel)
    views = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0], [-0.6, 0.0, 0.8, 0.0]],
        dtype=np.float32,
    )
    lights = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.6, 0.8, 0.0], [0.0, 0.8, 0.6, 0.0]],
        dtype=np.float32,
    )
    positions = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, 0.0], [-0.3, 0.2, 0.1, 0.0]],
        dtype=np.float32,
    )
    uv = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    argument_data = structured_buffer(device, falcor, _argument_rows(artifact_dir, manifest), 16)
    input_buffers = {
        "gViews": structured_buffer(device, falcor, views, 16),
        "gLights": structured_buffer(device, falcor, lights, 16),
        "gPositions": structured_buffer(device, falcor, positions, 16),
        "gUv": structured_buffer(device, falcor, uv, 16),
    }

    def execute(compute):
        output = output_buffer(device, falcor, len(views))
        for name, buffer in input_buffers.items():
            setattr(compute.globals, name, buffer)
        compute.globals.gMdlArgumentBlock = argument_data
        compute.globals.gOutput = output
        compute.globals.gQueryCount = len(views)
        compute.execute(threads_x=len(views))
        return output.to_numpy().view(np.float32).reshape(len(views), 4).copy()

    formal_result = execute(formal)
    viewer_result = execute(viewer)
    np.testing.assert_allclose(viewer_result, formal_result, rtol=2e-6, atol=2e-7)

    xi = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [0.3, 0.7, 0.2, 0.8], [0.8, 0.1, 0.6, 0.4]],
        dtype=np.float32,
    )
    direction_output = output_buffer(device, falcor, len(views))
    weight_output = output_buffer(device, falcor, len(views))
    sampler.globals.gViews = input_buffers["gViews"]
    sampler.globals.gPositions = input_buffers["gPositions"]
    sampler.globals.gUv = input_buffers["gUv"]
    sampler.globals.gXi = structured_buffer(device, falcor, xi, 16)
    sampler.globals.gMdlArgumentBlock = argument_data
    sampler.globals.gDirectionPdf = direction_output
    sampler.globals.gWeightValid = weight_output
    sampler.globals.gQueryCount = len(views)
    sampler.execute(threads_x=len(views))
    direction_pdf = direction_output.to_numpy().view(np.float32).reshape(len(views), 4).copy()
    sampled_weight = weight_output.to_numpy().view(np.float32).reshape(len(views), 4).copy()
    np.testing.assert_array_equal(sampled_weight[:, 3], np.ones(len(views), dtype=np.float32))
    np.testing.assert_allclose(
        np.linalg.norm(direction_pdf[:, :3], axis=1), np.ones(len(views)), rtol=2e-6, atol=2e-7
    )
    assert np.all(direction_pdf[:, 2] > 0.0) and np.all(direction_pdf[:, 3] > 0.0)

    input_buffers["gLights"] = structured_buffer(device, falcor, direction_pdf, 16)
    formal_sampled = execute(formal)
    np.testing.assert_allclose(direction_pdf[:, 3], formal_sampled[:, 3], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        sampled_weight[:, :3],
        formal_sampled[:, :3] / formal_sampled[:, 3:4],
        rtol=3e-6,
        atol=3e-7,
    )
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
