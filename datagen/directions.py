from __future__ import annotations

import numpy as np


def equal_area_hemisphere(bin_count: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic equal-solid-angle hemisphere nodes and weights."""
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    index = np.arange(bin_count, dtype=np.float64)
    z = (index + 0.5) / bin_count
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.zeros((bin_count, 4), dtype=np.float32)
    directions[:, 0] = (radius * np.cos(phi)).astype(np.float32)
    directions[:, 1] = (radius * np.sin(phi)).astype(np.float32)
    directions[:, 2] = z.astype(np.float32)
    weights = np.full(bin_count, 2.0 * np.pi / bin_count, dtype=np.float32)
    return directions, weights


def stratified_view_directions(view_count: int = 16, max_theta_degrees: float = 82.0) -> np.ndarray:
    """Generate deterministic views with extra density near grazing angles."""
    if view_count < 1:
        raise ValueError("view_count must be positive")
    u = (np.arange(view_count, dtype=np.float64) + 0.5) / view_count
    theta = np.deg2rad(max_theta_degrees) * np.power(u, 0.65)
    phi = np.arange(view_count, dtype=np.float64) * (np.pi * (3.0 - np.sqrt(5.0)))
    directions = np.zeros((view_count, 4), dtype=np.float32)
    directions[:, 0] = (np.sin(theta) * np.cos(phi)).astype(np.float32)
    directions[:, 1] = (np.sin(theta) * np.sin(phi)).astype(np.float32)
    directions[:, 2] = np.cos(theta).astype(np.float32)
    return directions
