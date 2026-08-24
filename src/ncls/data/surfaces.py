from __future__ import annotations

import math

from ncls.data.contract import PositionKind, SurfaceSample
from ncls.data.directions import stratified_uv


CONSTANT_FOOTPRINT_PROFILE = "constant"
FOOTPRINT_SWEEP_PROFILE = "footprint-sweep-v1"
SURFACE_PROFILES = (CONSTANT_FOOTPRINT_PROFILE, FOOTPRINT_SWEEP_PROFILE)
FOOTPRINT_SCALE_MULTIPLIERS = (1.0, 4.0, 16.0, 64.0)
FOOTPRINT_ROTATIONS = (0.0, math.pi / 8.0, math.pi / 4.0, 3.0 * math.pi / 8.0)
FOOTPRINT_SWEEP_MINIMUM_SAMPLE_COUNT = 20


def uv_surface_samples(
    sample_count: int,
    footprint_width: float,
    seed: int,
    surface_profile: str,
) -> tuple[SurfaceSample, ...]:
    if surface_profile not in SURFACE_PROFILES:
        raise ValueError(f"unknown surface profile {surface_profile!r}")
    if sample_count < 1 or footprint_width < 0.0 or seed < 0:
        raise ValueError("UV surface sample arguments are invalid")
    if surface_profile == CONSTANT_FOOTPRINT_PROFILE:
        return tuple(
            SurfaceSample(
                PositionKind.UV,
                uv=(float(uv[0]), float(uv[1])),
                uv_dx=(footprint_width, 0.0),
                uv_dy=(0.0, footprint_width),
            )
            for uv in stratified_uv(sample_count, seed)
        )

    if sample_count < FOOTPRINT_SWEEP_MINIMUM_SAMPLE_COUNT or footprint_width <= 0.0:
        raise ValueError(
            f"{FOOTPRINT_SWEEP_PROFILE} requires at least "
            f"{FOOTPRINT_SWEEP_MINIMUM_SAMPLE_COUNT} samples and a positive footprint width"
        )
    interior_count = sample_count - 4
    interior_uv = stratified_uv(interior_count, seed)
    epsilon = min(max(0.25 * footprint_width, 1e-7), 0.01)
    seam_uv = ((epsilon, 0.37), (1.0 - epsilon, 0.37), (0.61, epsilon), (0.61, 1.0 - epsilon))
    result: list[SurfaceSample] = []
    for index, uv in enumerate(interior_uv):
        scale = footprint_width * FOOTPRINT_SCALE_MULTIPLIERS[index % 4]
        angle = FOOTPRINT_ROTATIONS[(index // 4) % 4]
        cosine, sine = math.cos(angle), math.sin(angle)
        result.append(SurfaceSample(
            PositionKind.UV,
            uv=(float(uv[0]), float(uv[1])),
            uv_dx=(2.0 * scale * cosine, 2.0 * scale * sine),
            uv_dy=(-0.5 * scale * sine, 0.5 * scale * cosine),
        ))
    result.extend(
        SurfaceSample(
            PositionKind.UV,
            uv=(float(uv[0]), float(uv[1])),
            uv_dx=(footprint_width, 0.0),
            uv_dy=(0.0, footprint_width),
        )
        for uv in seam_uv
    )
    return tuple(result)
