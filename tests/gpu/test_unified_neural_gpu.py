"""Falcor编译03唯一Slang core，并锁定positive residual与两种proposal。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ncls.learning.evaluation.sampler_falcor_worker import (
    _buffers as _audit_buffers,
    _dispatch as _audit_dispatch,
)


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL = PROJECT_ROOT / "tests/gpu/kernels/unified_neural.cs.slang"
AUDIT_KERNEL = (
    PROJECT_ROOT
    / "shaders/ncls/backends/unified_neural/unified_sampler_audit.cs.slang"
)


def _nvidia_raw_zero_oracle(wo: np.ndarray, wi: np.ndarray) -> float:
    """独立NumPy oracle；不调用Slang sample/pdf实现。"""
    alpha = 0.5001
    half_vector = wo + wi
    half_vector = half_vector / np.linalg.norm(half_vector)
    if half_vector[2] < 0.0:
        half_vector = -half_vector
    slope_x = -half_vector[0] / half_vector[2]
    slope_y = -half_vector[1] / half_vector[2]
    radius_squared = (slope_x / alpha) ** 2 + (slope_y / alpha) ** 2
    p22 = 1.0 / (math.pi * (1.0 + radius_squared) ** 2 * alpha * alpha)
    half_pdf = p22 / half_vector[2] ** 3
    ggx_pdf = half_pdf / (4.0 * abs(float(np.dot(wo, half_vector))))
    cosine_pdf = wi[2] / math.pi
    return (1.0 / 32.0) * cosine_pdf + (31.0 / 64.0) * (ggx_pdf + cosine_pdf)


@pytest.fixture(scope="module")
def device():
    return falcor.Device(type=falcor.DeviceType.D3D12)


def _dispatch(device, entry: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = srv | falcor.ResourceBindFlags.UnorderedAccess
    source = np.asarray(values, dtype=np.float32)
    input_buffer = device.create_structured_buffer(struct_size=16, element_count=len(source), bind_flags=srv)
    output0 = device.create_structured_buffer(struct_size=16, element_count=len(source), bind_flags=uav)
    output1 = device.create_structured_buffer(struct_size=16, element_count=len(source), bind_flags=uav)
    input_buffer.from_numpy(source)
    compute = falcor.ComputePass(device, file=KERNEL, cs_entry=entry)
    compute.globals.gInput = input_buffer
    compute.globals.gOutput0 = output0
    compute.globals.gOutput1 = output1
    compute.globals.gCount = len(source)
    compute.globals.gMode = 0
    compute.execute(threads_x=len(source))
    first = output0.to_numpy().view(np.float32).reshape(len(source), 4).copy()
    second = output1.to_numpy().view(np.float32).reshape(len(source), 4).copy()
    return first, second


@pytest.mark.falcor
def test_unified_direct_and_positive_residual_are_finite(device) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32)
    direct, residual = _dispatch(device, "evaluateUnifiedNeural", directions)
    expected_direct = 0.1 * math.log(2.0)
    np.testing.assert_allclose(direct[:, :3], expected_direct, rtol=2e-6, atol=2e-7)
    assert np.all(residual[:, :3] > direct[:, :3])
    assert np.all(direct[:, 3] == 1.0) and np.all(residual[:, 3] == 1.0)


@pytest.mark.falcor
def test_unified_nvidia_and_ltc_pdf_have_cosine_safety_support(device) -> None:
    directions = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32)
    result, _ = _dispatch(device, "queryUnifiedSampler", directions)
    assert np.isfinite(result).all()
    assert np.all(result[:, :2] > 0.0)
    wo = np.asarray([0.25, -0.1, 0.963], dtype=np.float64)
    wo /= np.linalg.norm(wo)
    wi = directions[:, :3].astype(np.float64)
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    nvidia_oracle = np.asarray([_nvidia_raw_zero_oracle(wo, value) for value in wi])
    ltc_oracle = wi[:, 2] / math.pi
    np.testing.assert_allclose(result[:, 0], nvidia_oracle, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(result[:, 1], ltc_oracle, rtol=2e-5, atol=1e-7)


@pytest.mark.falcor
def test_unified_sample_recomputes_the_full_mixture_pdf(device) -> None:
    rng = np.random.default_rng(20260824)
    uniforms = np.column_stack((rng.random((4096, 3)), np.zeros(4096))).astype(np.float32)
    nvidia, ltc = _dispatch(device, "sampleUnifiedSampler", uniforms)
    assert np.isfinite(nvidia).all() and np.isfinite(ltc).all()
    nvidia_continuous = (nvidia[:, 2] == 1.0) & (nvidia[:, 3] == 0.0)
    ltc_continuous = (ltc[:, 2] == 1.0) & (ltc[:, 3] == 0.0)
    np.testing.assert_allclose(
        nvidia[nvidia_continuous, 0], nvidia[nvidia_continuous, 1], rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        ltc[ltc_continuous, 0], ltc[ltc_continuous, 1], rtol=2e-5, atol=1e-7
    )
    assert np.count_nonzero(ltc_continuous) == len(ltc)


@pytest.mark.falcor
def test_sampler_audit_compiles_and_dispatches_both_method_identities(device) -> None:
    prepared = np.zeros((1, 27), dtype=np.float32)
    views = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    directions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    prepared_buffer, view_buffer = _audit_buffers(device, prepared, views)
    query = falcor.ComputePass(
        device, file=AUDIT_KERNEL, cs_entry="queryUnifiedLearnedSampler"
    )
    sample = falcor.ComputePass(
        device, file=AUDIT_KERNEL, cs_entry="sampleUnifiedLearnedSampler"
    )
    for method_index in (0, 1):
        for sampler_index in (0, 1):
            queried, _ = _audit_dispatch(
                device,
                query,
                prepared_buffer,
                view_buffer,
                directions,
                0,
                sampler_index,
                method_index,
            )
            sampled, metadata = _audit_dispatch(
                device,
                sample,
                prepared_buffer,
                view_buffer,
                np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32),
                0,
                sampler_index,
                method_index,
            )
            assert np.isfinite(queried).all() and queried[0, 0] > 0.0
            assert np.isfinite(sampled).all() and np.isfinite(metadata).all()
            assert metadata[0, 1] == 1.0
