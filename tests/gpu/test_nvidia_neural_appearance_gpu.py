"""Falcor 编译 NVIDIA 原规模 core，并用独立公式锁定 response、frame 与 PDF。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL = PROJECT_ROOT / "tests/gpu/kernels/nvidia_neural_appearance.cs.slang"


@pytest.fixture(scope="module")
def device():
    return falcor.Device(type=falcor.DeviceType.D3D12)


def _dispatch(device, entry: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = srv | falcor.ResourceBindFlags.UnorderedAccess
    source = np.asarray(values, dtype=np.float32)
    input_buffer = device.create_structured_buffer(
        struct_size=16, element_count=len(source), bind_flags=srv
    )
    output0 = device.create_structured_buffer(
        struct_size=16, element_count=len(source), bind_flags=uav
    )
    output1 = device.create_structured_buffer(
        struct_size=16, element_count=len(source), bind_flags=uav
    )
    input_buffer.from_numpy(source)
    compute = falcor.ComputePass(device, file=KERNEL, cs_entry=entry)
    compute.globals.gInput = input_buffer
    compute.globals.gOutput0 = output0
    compute.globals.gOutput1 = output1
    compute.globals.gCount = len(source)
    compute.execute(threads_x=len(source))
    first = output0.to_numpy().view(np.float32).reshape(len(source), 4).copy()
    second = output1.to_numpy().view(np.float32).reshape(len(source), 4).copy()
    return first, second


def _frame(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray([raw[0], raw[1], raw[2] + 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent = np.asarray([raw[3] + 1.0, raw[4], raw[5]], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    return tangent, np.cross(normal, tangent), normal


@pytest.mark.falcor
def test_nvidia_response_adapter_and_native_pdf_compile_in_falcor(device) -> None:
    values = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32
    )
    response_pdf, bare_f = _dispatch(
        device, "evaluateNvidiaNeuralAppearance", values
    )
    expected_response = math.exp(-3.0)
    np.testing.assert_allclose(
        response_pdf[:, :3], expected_response, rtol=2e-6, atol=1e-7
    )
    normalized = values[:, :3].astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    np.testing.assert_allclose(
        bare_f[:, :3] * normalized[:, 2:3],
        response_pdf[:, :3],
        rtol=2e-6,
        atol=1e-7,
    )
    assert np.all(response_pdf[:, 3] > 0.0)
    assert np.all(bare_f[:, 3] == 1.0)


@pytest.mark.falcor
def test_nvidia_nonorthogonal_frame_and_direction_order_match_oracle(device) -> None:
    values = np.asarray([[0.6, 0.0, 0.8, 0.0]], dtype=np.float32)
    first, second = _dispatch(device, "inspectNvidiaNeuralInput", values)
    raw = np.asarray(
        [0.2, -0.1, 0.3, -0.4, 0.5, 0.1, -0.2, 0.3, -0.1, 0.2, -0.4, 0.6]
    )
    wo = np.asarray([0.0, 0.0, 1.0])
    wi = values[0, :3].astype(np.float64)
    wi /= np.linalg.norm(wi)
    expected: list[float] = []
    for offset in (0, 6):
        tangent, bitangent, normal = _frame(raw[offset : offset + 6])
        for direction in (wo, wi):
            expected.extend(
                [
                    np.dot(direction, tangent),
                    np.dot(direction, bitangent),
                    np.dot(direction, normal),
                ]
            )
    np.testing.assert_allclose(
        np.concatenate((first[0], second[0])), expected[:8], rtol=2e-6, atol=1e-7
    )


@pytest.mark.falcor
def test_nvidia_private_half_record_and_prepared_state_round_trip(device) -> None:
    first, second = _dispatch(
        device,
        "inspectNvidiaNeuralPacking",
        np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        first[0], [0.0, 0.1, 0.6, 0.7], rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        second[0], [-0.05, -0.3, 0.025, 0.225], rtol=5e-4, atol=5e-4
    )


@pytest.mark.falcor
def test_nvidia_matched_ltc_adaptation_compiles_and_recomputes_pdf(device) -> None:
    first, second = _dispatch(
        device,
        "inspectNvidiaMatchedLtc",
        np.asarray([[0.6, 0.0, 0.8, 0.0]], dtype=np.float32),
    )
    assert first[0, 0] > 0.0
    assert first[0, 3] == 1.0
    np.testing.assert_allclose(first[0, 1], second[0, 3], rtol=2e-5, atol=1e-7)
    assert second[0, 2] > 0.0
