from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


HEMISPHERE_PARAMETERIZATION_ID = "equal-area-fibonacci-hemisphere@2"
SPHERE_PARAMETERIZATION_ID = "equal-area-fibonacci-sphere@1"
VIEW_PARAMETERIZATION_ID = "grazing-weighted-fibonacci-hemisphere@2"
MIXTURE_QUERY_PROFILE_ID = "ncls.e0-peak-grazing-mixture@1"
_PEAK_SCALES = ((0.0025, 384.0), (0.0125, 96.0), (0.06, 24.0))
_STANDARD_NORMAL = NormalDist()


def equal_area_hemisphere(bin_count: int = 128, *, azimuth_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """返回确定性的等立体角上半球方向与 quadrature 权重。"""

    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    index = np.arange(bin_count, dtype=np.float64)
    z = (index + 0.5) / bin_count
    phi = index * (np.pi * (3.0 - np.sqrt(5.0))) + azimuth_offset
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1).astype(np.float32)
    weights = np.full(bin_count, 2.0 * np.pi / bin_count, dtype=np.float32)
    return directions, weights


def equal_area_sphere(bin_count: int = 128, *, azimuth_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """返回确定性的等立体角整球方向；用于含透射的 BSDF query profile。"""

    if bin_count < 2:
        raise ValueError("full-sphere bin_count must be at least two")
    index = np.arange(bin_count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / bin_count
    phi = index * (np.pi * (3.0 - np.sqrt(5.0))) + azimuth_offset
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1).astype(np.float32)
    weights = np.full(bin_count, 4.0 * np.pi / bin_count, dtype=np.float32)
    return directions, weights


def stratified_view_directions(
    view_count: int = 16,
    max_theta_degrees: float = 82.0,
    *,
    azimuth_offset: float = 0.0,
) -> np.ndarray:
    """生成确定性的出射方向，并提高 grazing-angle 覆盖。"""

    if view_count < 1:
        raise ValueError("view_count must be positive")
    if not 0.0 < max_theta_degrees < 90.0:
        raise ValueError("max_theta_degrees must lie in (0, 90)")
    u = (np.arange(view_count, dtype=np.float64) + 0.5) / view_count
    theta = np.deg2rad(max_theta_degrees) * np.power(u, 0.65)
    phi = np.arange(view_count, dtype=np.float64) * (np.pi * (3.0 - np.sqrt(5.0))) + azimuth_offset
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=1
    ).astype(np.float32)


def stratified_uv(sample_count: int, seed: int) -> np.ndarray:
    """生成可复现的 jittered UV 样本；数量不要求是完全平方数。"""

    if sample_count < 1 or seed < 0:
        raise ValueError("UV sample count must be positive and seed nonnegative")
    side = int(np.ceil(np.sqrt(sample_count)))
    rng = np.random.default_rng(seed)
    cells = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), axis=-1).reshape(-1, 2)
    jitter = rng.random((len(cells), 2))
    return ((cells[:sample_count] + jitter[:sample_count]) / side).astype(np.float32)


def _truncated_normal_samples(
    center: float,
    sigma: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lower = _STANDARD_NORMAL.cdf((0.0 - center) / sigma)
    upper = _STANDARD_NORMAL.cdf((1.0 - center) / sigma)
    quantiles = lower + (upper - lower) * ((np.arange(count) + rng.random(count)) / count)
    return np.asarray(
        [center + sigma * _STANDARD_NORMAL.inv_cdf(float(np.clip(value, 1e-12, 1.0 - 1e-12))) for value in quantiles],
        dtype=np.float64,
    )


def _truncated_normal_pdf(values: np.ndarray, center: float, sigma: float) -> np.ndarray:
    lower = _STANDARD_NORMAL.cdf((0.0 - center) / sigma)
    upper = _STANDARD_NORMAL.cdf((1.0 - center) / sigma)
    normalization = max(upper - lower, np.finfo(np.float64).tiny)
    standardized = (values - center) / sigma
    return np.exp(-0.5 * standardized * standardized) / (
        sigma * math.sqrt(2.0 * math.pi) * normalization
    )


def _peak_component_pdf(directions: np.ndarray, view: np.ndarray, sign: float) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    absolute_z = np.abs(values[..., 2])
    correct_side = values[..., 2] * sign > 0.0
    center_z = float(np.clip(view[2], 0.0, 1.0))
    center_phi = math.atan2(-float(view[1]), -float(view[0]))
    phi = np.arctan2(values[..., 1], values[..., 0])
    radial = max(0.0, 1.0 - center_z * center_z)
    result = np.zeros_like(absolute_z)
    for sigma, base_kappa in _PEAK_SCALES:
        kappa = base_kappa * radial
        z_pdf = _truncated_normal_pdf(absolute_z, center_z, sigma)
        phi_pdf = np.exp(kappa * np.cos(phi - center_phi)) / (2.0 * math.pi * np.i0(kappa))
        result += z_pdf * phi_pdf / len(_PEAK_SCALES)
    return np.where(correct_side, result, 0.0)


def _sample_peak_component(
    view: np.ndarray,
    count: int,
    sign: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 1:
        return np.empty((0, 3), dtype=np.float64)
    center_z = float(np.clip(view[2], 0.0, 1.0))
    center_phi = math.atan2(-float(view[1]), -float(view[0]))
    scale_indices = np.arange(count) % len(_PEAK_SCALES)
    rng.shuffle(scale_indices)
    result = np.empty((count, 3), dtype=np.float64)
    radial_factor = max(0.0, 1.0 - center_z * center_z)
    for scale_index, (sigma, base_kappa) in enumerate(_PEAK_SCALES):
        selected = np.flatnonzero(scale_indices == scale_index)
        if not len(selected):
            continue
        z = _truncated_normal_samples(center_z, sigma, len(selected), rng)
        kappa = base_kappa * radial_factor
        phi = rng.vonmises(center_phi, kappa, size=len(selected)) if kappa > 1e-8 else rng.uniform(-math.pi, math.pi, len(selected))
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        result[selected] = np.stack((radius * np.cos(phi), radius * np.sin(phi), sign * z), axis=1)
    return result


def _sample_uniform(count: int, full_sphere: bool, rng: np.random.Generator) -> np.ndarray:
    u = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(u)
    z = 1.0 - 2.0 * u if full_sphere else u
    phi = 2.0 * math.pi * ((np.arange(count, dtype=np.float64) + rng.random(count)) / count)
    rng.shuffle(phi)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1)


def _sample_grazing(count: int, full_sphere: bool, rng: np.random.Generator, exponent: float = 7.0) -> np.ndarray:
    u = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(u)
    absolute_z = 1.0 - np.power(1.0 - u, 1.0 / (exponent + 1.0))
    if full_sphere:
        signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        z = signs * absolute_z
    else:
        z = absolute_z
    phi = rng.uniform(-math.pi, math.pi, count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1)


def peak_grazing_mixture_pdf(
    directions: np.ndarray,
    view: np.ndarray,
    *,
    full_sphere: bool,
    component_weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    exponent = 7.0
    if full_sphere:
        weights = component_weights or (0.4, 0.25, 0.2, 0.15)
        uniform = np.full(values.shape[:-1], 1.0 / (4.0 * math.pi))
        grazing = (exponent + 1.0) * np.power(1.0 - np.abs(values[..., 2]), exponent) / (4.0 * math.pi)
        return (
            weights[0] * uniform
            + weights[1] * _peak_component_pdf(values, view, 1.0)
            + weights[2] * _peak_component_pdf(values, view, -1.0)
            + weights[3] * grazing
        )
    weights = component_weights or (0.5, 0.35, 0.15)
    uniform = np.full(values.shape[:-1], 1.0 / (2.0 * math.pi))
    grazing = (exponent + 1.0) * np.power(1.0 - values[..., 2], exponent) / (2.0 * math.pi)
    return weights[0] * uniform + weights[1] * _peak_component_pdf(values, view, 1.0) + weights[2] * grazing


def peak_grazing_mixture_query(
    views: np.ndarray,
    direction_count: int,
    *,
    full_sphere: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成按 wo 对齐、PDF 可计算的 uniform + peak + grazing 固定 query mixture。"""

    view_values = np.asarray(views, dtype=np.float64)
    if direction_count < 16 or seed < 0:
        raise ValueError("mixture query requires at least 16 directions and a nonnegative seed")
    base_weights = (0.4, 0.25, 0.2, 0.15) if full_sphere else (0.5, 0.35, 0.15)
    counts = [max(1, int(round(direction_count * weight))) for weight in base_weights]
    counts[-1] += direction_count - sum(counts)
    if counts[-1] < 1:
        deficit = 1 - counts[-1]
        counts[-1] = 1
        counts[0] -= deficit
    actual_weights = tuple(count / direction_count for count in counts)
    all_directions = np.empty((len(view_values), direction_count, 3), dtype=np.float32)
    all_pdf = np.empty((len(view_values), direction_count), dtype=np.float32)
    all_weights = np.empty((len(view_values), direction_count), dtype=np.float32)
    for view_index, view in enumerate(view_values):
        rng = np.random.default_rng(seed ^ ((view_index + 1) * 0x9E3779B1))
        if full_sphere:
            parts = (
                _sample_uniform(counts[0], True, rng),
                _sample_peak_component(view, counts[1], 1.0, rng),
                _sample_peak_component(view, counts[2], -1.0, rng),
                _sample_grazing(counts[3], True, rng),
            )
        else:
            parts = (
                _sample_uniform(counts[0], False, rng),
                _sample_peak_component(view, counts[1], 1.0, rng),
                _sample_grazing(counts[2], False, rng),
            )
        directions = np.concatenate(parts)
        permutation = rng.permutation(direction_count)
        directions = directions[permutation]
        pdf = peak_grazing_mixture_pdf(
            directions,
            view,
            full_sphere=full_sphere,
            component_weights=actual_weights,
        )
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise RuntimeError("mixture query produced an invalid PDF")
        all_directions[view_index] = directions.astype(np.float32)
        all_pdf[view_index] = pdf.astype(np.float32)
        all_weights[view_index] = (1.0 / (direction_count * pdf)).astype(np.float32)
    return all_directions, all_weights, all_pdf
