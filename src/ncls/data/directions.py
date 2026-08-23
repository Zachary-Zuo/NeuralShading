from __future__ import annotations

import numpy as np


HEMISPHERE_PARAMETERIZATION_ID = "equal-area-fibonacci-hemisphere@2"
SPHERE_PARAMETERIZATION_ID = "equal-area-fibonacci-sphere@1"
VIEW_PARAMETERIZATION_ID = "grazing-weighted-fibonacci-hemisphere@2"


def equal_area_hemisphere(bin_count: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """返回确定性的等立体角上半球方向与 quadrature 权重。"""

    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    index = np.arange(bin_count, dtype=np.float64)
    z = (index + 0.5) / bin_count
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1).astype(np.float32)
    weights = np.full(bin_count, 2.0 * np.pi / bin_count, dtype=np.float32)
    return directions, weights


def equal_area_sphere(bin_count: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """返回确定性的等立体角整球方向；用于含透射的 BSDF query profile。"""

    if bin_count < 2:
        raise ValueError("full-sphere bin_count must be at least two")
    index = np.arange(bin_count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / bin_count
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1).astype(np.float32)
    weights = np.full(bin_count, 4.0 * np.pi / bin_count, dtype=np.float32)
    return directions, weights


def stratified_view_directions(view_count: int = 16, max_theta_degrees: float = 82.0) -> np.ndarray:
    """生成确定性的出射方向，并提高 grazing-angle 覆盖。"""

    if view_count < 1:
        raise ValueError("view_count must be positive")
    if not 0.0 < max_theta_degrees < 90.0:
        raise ValueError("max_theta_degrees must lie in (0, 90)")
    u = (np.arange(view_count, dtype=np.float64) + 0.5) / view_count
    theta = np.deg2rad(max_theta_degrees) * np.power(u, 0.65)
    phi = np.arange(view_count, dtype=np.float64) * (np.pi * (3.0 - np.sqrt(5.0)))
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
