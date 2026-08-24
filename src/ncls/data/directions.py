from __future__ import annotations

import math

import numpy as np


HEMISPHERE_PARAMETERIZATION_ID = "equal-area-fibonacci-hemisphere@2"
SPHERE_PARAMETERIZATION_ID = "equal-area-fibonacci-sphere@1"
VIEW_PARAMETERIZATION_ID = "grazing-weighted-fibonacci-hemisphere@2"
MIXTURE_QUERY_PROFILE_ID = "ncls.e0-peak-grazing-mixture@2"
E1_MIXTURE_QUERY_PROFILE_ID = "ncls.e1-independent-peak-grazing-mixture@1"
E2_LAYER_STACK_MIXTURE_QUERY_PROFILE_V1_ID = "ncls.e2-layer-stack-independent-peak-grazing-mixture@1"
E2_LAYER_STACK_MIXTURE_QUERY_PROFILE_ID = "ncls.e2-layer-stack-independent-peak-grazing-mixture@2"
_PEAK_ANGULAR_SCALES = (0.0025, 0.0125, 0.06)


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


def grazing_anchored_view_directions(
    view_count: int,
    *,
    max_theta_degrees: float = 89.0,
    azimuth_offset: float = 0.0,
) -> np.ndarray:
    """生成分层 `wo`，并让每个非空 role 明确包含一个掠射 probe。"""

    result = stratified_view_directions(
        view_count,
        max_theta_degrees=max_theta_degrees,
        azimuth_offset=azimuth_offset,
    )
    phi = (view_count - 1) * (np.pi * (3.0 - np.sqrt(5.0))) + azimuth_offset
    theta = np.deg2rad(max_theta_degrees)
    result[-1] = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    return result


def stratified_uv(sample_count: int, seed: int) -> np.ndarray:
    """生成可复现的 jittered UV 样本；数量不要求是完全平方数。"""

    if sample_count < 1 or seed < 0:
        raise ValueError("UV sample count must be positive and seed nonnegative")
    side = int(np.ceil(np.sqrt(sample_count)))
    rng = np.random.default_rng(seed)
    cells = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), axis=-1).reshape(-1, 2)
    jitter = rng.random((len(cells), 2))
    return ((cells[:sample_count] + jitter[:sample_count]) / side).astype(np.float32)


def _peak_center(view: np.ndarray, sign: float) -> np.ndarray:
    center = np.asarray(
        (-float(view[0]), -float(view[1]), sign * abs(float(view[2]))),
        dtype=np.float64,
    )
    return center / np.linalg.norm(center)


def _vmf_pdf(directions: np.ndarray, center: np.ndarray, kappa: float) -> np.ndarray:
    """返回数值稳定的三维 von Mises–Fisher 球面 PDF。"""

    dots = np.clip(
        np.sum(np.asarray(directions, dtype=np.float64) * center, axis=-1),
        -1.0,
        1.0,
    )
    log_normalization = (
        math.log(kappa)
        - math.log(2.0 * math.pi)
        - math.log1p(-math.exp(-2.0 * kappa))
    )
    return np.exp(log_normalization + kappa * (dots - 1.0))


def _folded_vmf_pdf(
    directions: np.ndarray,
    center: np.ndarray,
    kappa: float,
    sign: float,
) -> np.ndarray:
    """把完整球 vMF 折到指定半球；原方向与镜像 PDF 之和仍严格归一。"""

    values = np.asarray(directions, dtype=np.float64)
    reflected = values.copy()
    reflected[..., 2] *= -1.0
    result = _vmf_pdf(values, center, kappa) + _vmf_pdf(reflected, center, kappa)
    return np.where(values[..., 2] * sign > 0.0, result, 0.0)


def _peak_component_pdf(
    directions: np.ndarray,
    view: np.ndarray,
    sign: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    center = _peak_center(view, sign) if center is None else np.asarray(center, dtype=np.float64)
    result = np.zeros(np.asarray(directions).shape[:-1], dtype=np.float64)
    for sigma in _PEAK_ANGULAR_SCALES:
        result += _folded_vmf_pdf(directions, center, 1.0 / (sigma * sigma), sign)
    return result / len(_PEAK_ANGULAR_SCALES)


def _sample_vmf(
    center: np.ndarray,
    kappa: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    u = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(u)
    tail = math.exp(-2.0 * kappa)
    cosine = 1.0 + np.log(
        np.maximum(u + (1.0 - u) * tail, np.finfo(np.float64).tiny)
    ) / kappa
    cosine = np.clip(cosine, -1.0, 1.0)
    phi = 2.0 * math.pi * ((np.arange(count, dtype=np.float64) + rng.random(count)) / count)
    rng.shuffle(phi)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    helper = np.asarray(
        (0.0, 0.0, 1.0) if abs(center[2]) < 0.9 else (1.0, 0.0, 0.0)
    )
    tangent = np.cross(helper, center)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(center, tangent)
    return (
        sine[:, None] * np.cos(phi)[:, None] * tangent
        + sine[:, None] * np.sin(phi)[:, None] * bitangent
        + cosine[:, None] * center
    )


def _sample_peak_component(
    view: np.ndarray,
    count: int,
    sign: float,
    rng: np.random.Generator,
    center: np.ndarray | None = None,
) -> np.ndarray:
    if count < 1:
        return np.empty((0, 3), dtype=np.float64)
    center = _peak_center(view, sign) if center is None else np.asarray(center, dtype=np.float64)
    scale_indices = np.arange(count) % len(_PEAK_ANGULAR_SCALES)
    rng.shuffle(scale_indices)
    result = np.empty((count, 3), dtype=np.float64)
    for scale_index, sigma in enumerate(_PEAK_ANGULAR_SCALES):
        selected = np.flatnonzero(scale_indices == scale_index)
        if not len(selected):
            continue
        result[selected] = _sample_vmf(center, 1.0 / (sigma * sigma), len(selected), rng)
    result[:, 2] = sign * np.abs(result[:, 2])
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
    reflection_center: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    exponent = 7.0
    if full_sphere:
        weights = component_weights or (0.4, 0.25, 0.2, 0.15)
        uniform = np.full(values.shape[:-1], 1.0 / (4.0 * math.pi))
        grazing = (exponent + 1.0) * np.power(1.0 - np.abs(values[..., 2]), exponent) / (4.0 * math.pi)
        return (
            weights[0] * uniform
            + weights[1] * _peak_component_pdf(values, view, 1.0, reflection_center)
            + weights[2] * _peak_component_pdf(values, view, -1.0)
            + weights[3] * grazing
        )
    weights = component_weights or (0.5, 0.35, 0.15)
    uniform = np.full(values.shape[:-1], 1.0 / (2.0 * math.pi))
    grazing = (exponent + 1.0) * np.power(1.0 - values[..., 2], exponent) / (2.0 * math.pi)
    return (
        weights[0] * uniform
        + weights[1] * _peak_component_pdf(values, view, 1.0, reflection_center)
        + weights[2] * grazing
    )


def peak_grazing_mixture_query(
    views: np.ndarray,
    direction_count: int,
    *,
    full_sphere: bool,
    seed: int,
    reflection_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成按 wo 对齐、PDF 可计算的 uniform + peak + grazing 固定 query mixture。"""

    view_values = np.asarray(views, dtype=np.float64)
    if direction_count < 16 or seed < 0:
        raise ValueError("mixture query requires at least 16 directions and a nonnegative seed")
    if reflection_centers is None:
        center_values: np.ndarray | None = None
    else:
        center_values = np.asarray(reflection_centers, dtype=np.float64)
        if center_values.shape != view_values.shape or not np.all(np.isfinite(center_values)):
            raise ValueError("reflection_centers must match views and contain finite values")
        lengths = np.linalg.norm(center_values, axis=1)
        if np.any(np.abs(lengths - 1.0) > 2e-4):
            raise ValueError("reflection_centers must be normalized")
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
        reflection_center = None if center_values is None else center_values[view_index]
        if full_sphere:
            parts = (
                _sample_uniform(counts[0], True, rng),
                _sample_peak_component(view, counts[1], 1.0, rng, reflection_center),
                _sample_peak_component(view, counts[2], -1.0, rng),
                _sample_grazing(counts[3], True, rng),
            )
        else:
            parts = (
                _sample_uniform(counts[0], False, rng),
                _sample_peak_component(view, counts[1], 1.0, rng, reflection_center),
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
            reflection_center=reflection_center,
        )
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise RuntimeError("mixture query produced an invalid PDF")
        all_directions[view_index] = directions.astype(np.float32)
        all_pdf[view_index] = pdf.astype(np.float32)
        all_weights[view_index] = (1.0 / (direction_count * pdf)).astype(np.float32)
    return all_directions, all_weights, all_pdf
