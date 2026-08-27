from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ncls.data.providers.mdl import MdlGpuQueryRuntime
from ncls.paths import PROJECT_ROOT
from ncls.references.mdl import MdlSdkCompilerBridge
from tools.reference.mdl_native_protocol import (
    read_native_result_packet,
    write_native_query_packet,
)


@pytest.mark.falcor
def test_mdl_sdk_native_and_current_falcor_match_on_disjoint_diffuse_queries(
    tmp_path: Path,
) -> None:
    module_root = PROJECT_ROOT / "tests/fixtures/mdl"
    try:
        bridge = MdlSdkCompilerBridge(module_root)
    except FileNotFoundError as error:
        pytest.skip(str(error))

    wo = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.31, -0.21, 0.927),
            (-0.58, 0.17, 0.797),
            (0.11, 0.73, 0.675),
            (-0.42, -0.51, 0.751),
        ),
        dtype=np.float32,
    )
    wi = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (-0.25, 0.41, 0.877),
            (0.61, -0.09, 0.787),
            (-0.12, -0.82, 0.56),
            (0.52, 0.47, 0.713),
        ),
        dtype=np.float32,
    )
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    position = np.asarray(
        tuple((0.13 * index, -0.07 * index, 0.0) for index in range(len(wo))),
        dtype=np.float32,
    )
    uv = np.asarray(
        tuple((0.17 + 0.11 * index, 0.23 + 0.09 * index) for index in range(len(wo))),
        dtype=np.float32,
    )
    tint = np.asarray((0.17, 0.53, 0.81), dtype=np.float32)
    query_path = tmp_path / "native-query.bin"
    result_path = tmp_path / "native-result.bin"
    write_native_query_packet(query_path, wo, wi, position, uv)
    artifact = bridge.native_evaluate(
        "::constant_diffuse",
        "constant_diffuse",
        {"tint": tint.tolist()},
        queries=query_path,
        output=tmp_path / "compiled",
        result=result_path,
    )
    native_response, native_pdf = read_native_result_packet(result_path)

    runtime = MdlGpuQueryRuntime(
        artifact,
        sdk_root=bridge.sdk_root,
        query_capacity=len(wo),
    )
    try:
        gpu_response, gpu_pdf = runtime.evaluate_torch(
            0,
            wo,
            wi,
            uv,
            np.zeros((len(wo), 4), dtype=np.float32),
            position,
        )
        current_falcor_response = gpu_response.detach().cpu().numpy().copy()
        current_falcor_pdf = gpu_pdf.detach().cpu().numpy().copy()
    finally:
        runtime.close()

    expected = tint[None, :] * wi[:, 2:3] / np.pi
    np.testing.assert_allclose(native_response, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(current_falcor_response, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(current_falcor_response, native_response, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(current_falcor_pdf, native_pdf, rtol=3e-6, atol=3e-7)
