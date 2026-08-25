"""锁定公共散射 sample/PDF、null 语义与官方解析公式。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL = PROJECT_ROOT / "tests" / "gpu" / "kernels" / "scattering_math.cs.slang"

COSINE = 0
TILTED = 1
LTC = 2
NONCENTERED_GGX = 3
NVIDIA = 4
GGX_VNDF = 5


@pytest.fixture(scope="module")
def device():
    return falcor.Device(type=falcor.DeviceType.D3D12)


def _dispatch(
    device,
    entry: str,
    inputs: np.ndarray,
    mode: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(inputs, dtype=np.float32)
    assert values.ndim == 2 and values.shape[1] == 4
    srv = falcor.ResourceBindFlags.ShaderResource
    uav = falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess
    input_buffer = device.create_structured_buffer(
        struct_size=16, element_count=len(values), bind_flags=srv
    )
    output0 = device.create_structured_buffer(
        struct_size=16, element_count=len(values), bind_flags=uav
    )
    output1 = device.create_structured_buffer(
        struct_size=16, element_count=len(values), bind_flags=uav
    )
    input_buffer.from_numpy(values)
    compute = falcor.ComputePass(device, file=KERNEL, cs_entry=entry)
    compute.globals.gInput = input_buffer
    compute.globals.gOutput0 = output0
    compute.globals.gOutput1 = output1
    compute.globals.gCount = len(values)
    compute.globals.gMode = mode
    compute.execute(threads_x=len(values))
    first = output0.to_numpy().view(np.float32).reshape(len(values), 4).copy()
    second = output1.to_numpy().view(np.float32).reshape(len(values), 4).copy()
    return first, second


def _legendre_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    # 不调用 LAPACK：Falcor 与 Torch 同进程收集测试时，Windows BLAS runtime 会冲突退出。
    nodes = np.cos(
        math.pi * (np.arange(1, order + 1, dtype=np.float64) - 0.25) / (order + 0.5)
    )
    derivative = np.zeros_like(nodes)
    for _ in range(16):
        previous = np.ones_like(nodes)
        current = nodes.copy()
        for degree in range(2, order + 1):
            following = (
                (2.0 * degree - 1.0) * nodes * current - (degree - 1.0) * previous
            ) / degree
            previous, current = current, following
        derivative = order * (nodes * current - previous) / (nodes * nodes - 1.0)
        update = current / derivative
        nodes -= update
        if float(np.max(np.abs(update))) < 2e-15:
            break
    node_weights = 2.0 / ((1.0 - nodes * nodes) * derivative * derivative)
    order_index = np.argsort(nodes)
    return nodes[order_index], node_weights[order_index]


def _quadrature(order_z: int = 128, order_phi: int = 256) -> tuple[np.ndarray, np.ndarray]:
    nodes, node_weights = _legendre_nodes_weights(order_z)
    z = 0.5 * (nodes + 1.0)
    z_weights = 0.5 * node_weights
    phi = (np.arange(order_phi, dtype=np.float64) + 0.5) * (2.0 * math.pi / order_phi)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    radial = np.sqrt(np.maximum(1.0 - zz * zz, 0.0))
    directions = np.stack(
        (radial * np.cos(pp), radial * np.sin(pp), zz, np.zeros_like(zz)), axis=-1
    ).reshape(-1, 4)
    weights = np.broadcast_to(
        z_weights[:, None] * (2.0 * math.pi / order_phi), (order_z, order_phi)
    ).reshape(-1)
    return directions.astype(np.float32), weights


def _query(device, mode: int, directions: np.ndarray) -> np.ndarray:
    result, _ = _dispatch(device, "queryScatteringMath", directions, mode)
    return result[:, 0].astype(np.float64)


def _samples(device, mode: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    random = np.zeros((count, 4), dtype=np.float32)
    random[:, :3] = np.random.default_rng(seed).random((count, 3), dtype=np.float32)
    return _dispatch(device, "sampleScatteringMath", random, mode)


def _solid_angle_bin_mass(
    directions: np.ndarray,
    masses: np.ndarray,
    bins_z: int = 8,
    bins_phi: int = 16,
) -> np.ndarray:
    z = np.clip(directions[:, 2], 0.0, np.nextafter(1.0, 0.0))
    phi = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi)
    z_index = np.minimum((z * bins_z).astype(np.int64), bins_z - 1)
    phi_index = np.minimum((phi * (bins_phi / (2.0 * math.pi))).astype(np.int64), bins_phi - 1)
    return np.bincount(
        z_index * bins_phi + phi_index,
        weights=masses,
        minlength=bins_z * bins_phi,
    )


@pytest.mark.falcor
@pytest.mark.parametrize("mode", [COSINE, LTC])
def test_cosine_and_ltc_fixed_quadrature_normalization(device, mode: int) -> None:
    directions, weights = _quadrature()
    integral = float(np.sum(_query(device, mode, directions) * weights))
    assert abs(integral - 1.0) <= 2e-4


@pytest.mark.falcor
@pytest.mark.parametrize("mode", [COSINE, TILTED, LTC, NONCENTERED_GGX, NVIDIA])
def test_directional_samples_revaluate_full_pdf(device, mode: int) -> None:
    samples, metadata = _samples(device, mode, 32768, 0x5100 + mode)
    assert np.all(metadata[:, 0] == 1.0)
    continuous = metadata[:, 1] == 0.0
    queried = _query(device, mode, samples[continuous])
    np.testing.assert_allclose(samples[continuous, 3], queried, rtol=2e-5, atol=2e-6)
    if mode == NVIDIA:
        counts = np.bincount(metadata[:, 3].astype(np.int64), minlength=3) / len(metadata)
        raw_logits = np.asarray([0.6, -0.25], dtype=np.float64)
        learned = np.exp(raw_logits - np.max(raw_logits))
        learned /= learned.sum()
        expected = np.asarray([1.0 / 32.0, *(31.0 / 32.0 * learned)])
        np.testing.assert_allclose(counts, expected, atol=0.008)


@pytest.mark.falcor
@pytest.mark.parametrize("mode", [COSINE, TILTED, LTC, NONCENTERED_GGX, NVIDIA])
def test_null_mass_and_histogram_match_pdf(device, mode: int) -> None:
    directions, weights = _quadrature()
    pdf = _query(device, mode, directions)
    expected_bins = _solid_angle_bin_mass(directions, pdf * weights)
    continuous_mass = float(expected_bins.sum())

    samples, metadata = _samples(device, mode, 1 << 17, 0x7100 + mode)
    assert np.all(metadata[:, 0] == 1.0)
    null = metadata[:, 1] != 0.0
    continuous = ~null
    empirical_bins = _solid_angle_bin_mass(
        samples[continuous, :3], np.full(np.count_nonzero(continuous), 1.0 / len(samples))
    )
    null_frequency = float(np.mean(null))
    assert abs(continuous_mass + null_frequency - 1.0) <= 1e-2
    total_variation = 0.5 * (
        np.abs(empirical_bins - expected_bins).sum()
        + abs(null_frequency - (1.0 - continuous_mass))
    )
    assert total_variation <= 0.04


@pytest.mark.falcor
def test_ggx_vndf_normal_histogram_and_reflection_null_are_separate(device) -> None:
    directions, weights = _quadrature(256, 512)
    pdf = _query(device, GGX_VNDF, directions)
    expected_bins = _solid_angle_bin_mass(directions, pdf * weights)
    assert abs(float(expected_bins.sum()) - 1.0) <= 5e-4

    samples, metadata = _samples(device, GGX_VNDF, 1 << 17, 0x9105)
    assert np.all(metadata[:, 0] == 1.0)
    queried = _query(device, GGX_VNDF, samples)
    np.testing.assert_allclose(samples[:, 3], queried, rtol=2e-5, atol=2e-6)
    empirical_bins = _solid_angle_bin_mass(
        samples[:, :3], np.full(len(samples), 1.0 / len(samples))
    )
    assert 0.5 * np.abs(empirical_bins - expected_bins).sum() <= 0.04
    reflection_null_frequency = float(np.mean(metadata[:, 1] != 0.0))
    assert 0.0 < reflection_null_frequency < 1.0


@pytest.mark.falcor
def test_boundary_random_inputs_are_finite_and_nvidia_has_full_support(device) -> None:
    edge = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [np.nextafter(np.float32(1.0), np.float32(0.0)), 0.5, 0.75, 0.0],
            [1e-7, np.nextafter(np.float32(1.0), np.float32(0.0)), 0.25, 0.0],
        ],
        dtype=np.float32,
    )
    for mode in (COSINE, TILTED, LTC, NONCENTERED_GGX, NVIDIA, GGX_VNDF):
        samples, metadata = _dispatch(device, "sampleScatteringMath", edge, mode)
        assert np.isfinite(samples).all()
        assert np.isfinite(metadata).all()
        assert np.all(metadata[:, 0] == 1.0)

    directions, _ = _quadrature(32, 64)
    nvidia_pdf = _query(device, NVIDIA, directions)
    assert np.isfinite(nvidia_pdf).all()
    assert np.all(nvidia_pdf > 0.0)

    grazing_z = 1.1e-6
    grazing = np.asarray(
        [[math.sqrt(1.0 - grazing_z * grazing_z), 0.0, grazing_z, 0.0]],
        dtype=np.float32,
    )
    assert float(_query(device, NVIDIA, grazing)[0]) > 0.0

    boundary, boundary_metadata = _dispatch(
        device, "checkScatteringBoundaries", np.zeros((6, 4), dtype=np.float32), 0
    )
    assert np.isfinite(boundary).all()
    assert np.isfinite(boundary_metadata).all()
    assert np.all(boundary_metadata[:, 0] == 1.0)
    assert np.all(boundary_metadata[:, 3] == 1.0)


@pytest.mark.falcor
def test_mixture_selection_remaps_each_cdf_interval(device) -> None:
    inputs = np.zeros((3, 4), dtype=np.float32)
    inputs[:, 0] = np.asarray([0.1, 0.35, 0.75], dtype=np.float32)
    result, _ = _dispatch(device, "checkMixture3Selection", inputs, 0)
    np.testing.assert_array_equal(result[:, 0], np.asarray([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(result[:, 1], 0.5, rtol=0.0, atol=2e-7)
    np.testing.assert_array_equal(result[:, 2], 1.0)


@pytest.mark.falcor
def test_nvidia_listing3_decode_order_and_range_warps(device) -> None:
    result, _ = _dispatch(
        device, "checkNvidiaProposalDecode", np.zeros((3, 4), dtype=np.float32), 0
    )
    # 固定值独立按 NVIDIA supplemental Listing 3 计算，避免以生产 decoder 自证。
    np.testing.assert_allclose(
        result[0],
        np.asarray([0.2949176613, 0.6437739428, 0.5734623444, 1.0]),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result[1],
        np.asarray([0.2039607805, -0.1516781131, 0.6276991716, -0.4308131846]),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result[2],
        np.asarray([0.7005671425, 0.2994328575, 0.6786744193, 0.2900755807]),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.falcor
def test_fixed_ltc_and_nvidia_listing4_pdf_oracles(device) -> None:
    direction = np.asarray([[0.2, 0.3, 0.93, 0.0]], dtype=np.float32)
    # 固定值独立按 LTC Jacobian 与 Listing 4 的 slope density / reflection Jacobian 计算。
    expected = {
        LTC: 0.2368317526,
        NONCENTERED_GGX: 0.00681699554,
        NVIDIA: 0.09029001565,
    }
    for mode, oracle in expected.items():
        value = float(_query(device, mode, direction)[0])
        assert value == pytest.approx(oracle, rel=2e-5, abs=2e-7)


@pytest.mark.falcor
def test_directional_sample_states_distinguish_null_and_invalid(device) -> None:
    inputs = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, -0.5, 0.0],
            [np.nan, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    samples, metadata = _dispatch(device, "checkDirectionalSampleStates", inputs, 0)
    assert np.isfinite(samples).all()
    np.testing.assert_array_equal(metadata[:, :2], [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(samples[1, 3], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(samples[2], [0.0, 0.0, 1.0, 0.0], rtol=0.0, atol=0.0)


@pytest.mark.falcor
def test_ltc_k2_analytic_control_core_compiles_and_is_finite(device) -> None:
    directions = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32
    )
    result, _ = _dispatch(device, "evaluateLtcK2AnalyticControl", directions, 0)
    assert np.isfinite(result).all()
    assert np.all(result[:, :3] > 0.0)
    assert np.all(result[:, 3] == 1.0)


@pytest.mark.falcor
def test_ltc_k2_analytic_control_identity(device) -> None:
    directions = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.6, 0.0, 0.8, 0.0]], dtype=np.float32
    )
    diffuse, identity_ltc = _dispatch(
        device, "checkLtcK2AnalyticControlIdentity", directions, 0
    )
    expected_diffuse = np.asarray([0.3, 0.2, 0.1]) / math.pi
    expected_ltc = np.asarray([0.24, 0.12, 0.06]) / math.pi
    np.testing.assert_allclose(
        diffuse[:, :3], np.broadcast_to(expected_diffuse, (2, 3)), rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        identity_ltc[:, :3],
        np.broadcast_to(expected_ltc, (2, 3)),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(diffuse[:, 3], 1.0)
    np.testing.assert_array_equal(identity_ltc[:, 3], 1.0)
