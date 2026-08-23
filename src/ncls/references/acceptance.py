from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACCEPTANCE_SCHEMA = "ncls.reference-acceptance"
ACCEPTANCE_VERSION = 1


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class DeterministicDirectionalGate:
    absolute_floor: float
    median_relative_l1_max: float
    p99_relative_l1_max: float
    max_relative_l1_max: float
    max_scaled_absolute_error: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeterministicDirectionalGate:
        gate = cls(*(float(value[name]) for name in (
            "absolute_floor",
            "median_relative_l1_max",
            "p99_relative_l1_max",
            "max_relative_l1_max",
            "max_scaled_absolute_error",
        )))
        if gate.absolute_floor <= 0.0 or min(
            gate.median_relative_l1_max,
            gate.p99_relative_l1_max,
            gate.max_relative_l1_max,
            gate.max_scaled_absolute_error,
        ) < 0.0:
            raise ValueError("deterministic directional gate thresholds must be nonnegative")
        return gate


@dataclass(frozen=True)
class DeterministicDirectionalMetrics:
    query_count: int
    median_relative_l1: float
    p99_relative_l1: float
    max_relative_l1: float
    max_absolute_error: float
    max_scaled_absolute_error: float
    passed: bool


@dataclass(frozen=True)
class MonteCarloGate:
    sigma_multiplier: float
    relative_allowance: float
    absolute_floor: float
    minimum_coverage: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MonteCarloGate:
        gate = cls(*(float(value[name]) for name in (
            "sigma_multiplier", "relative_allowance", "absolute_floor", "minimum_coverage"
        )))
        if gate.sigma_multiplier <= 0.0 or gate.relative_allowance < 0.0 or gate.absolute_floor <= 0.0:
            raise ValueError("Monte Carlo gate tolerances are invalid")
        if not 0.0 <= gate.minimum_coverage <= 1.0:
            raise ValueError("Monte Carlo minimum coverage must lie in [0, 1]")
        return gate


@dataclass(frozen=True)
class ImageGate:
    luminance_floor: float
    p95_relative_l1_max: float
    linear_psnr_min_db: float
    max_absolute_mae: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImageGate:
        gate = cls(*(float(value[name]) for name in (
            "luminance_floor", "p95_relative_l1_max", "linear_psnr_min_db", "max_absolute_mae"
        )))
        if gate.luminance_floor <= 0.0 or gate.p95_relative_l1_max < 0.0 or gate.max_absolute_mae < 0.0:
            raise ValueError("image gate tolerances are invalid")
        return gate


@dataclass(frozen=True)
class ImageMetrics:
    pixel_count: int
    p95_relative_l1: float
    linear_psnr_db: float
    absolute_mae: float
    peak_value: float
    passed: bool


@dataclass(frozen=True)
class ReferenceAcceptance:
    deterministic_directional: DeterministicDirectionalGate
    monte_carlo: MonteCarloGate
    linear_hdr_image: ImageGate
    linear_hdr_textured_image: ImageGate


def load_reference_acceptance(path: str | Path) -> ReferenceAcceptance:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read reference acceptance file {source}: {error}") from error
    root = _mapping("reference acceptance", value)
    if root.get("schema_name") != ACCEPTANCE_SCHEMA or root.get("schema_version") != ACCEPTANCE_VERSION:
        raise ValueError("unsupported reference acceptance schema")
    profiles = _mapping("profiles", root.get("profiles"))
    return ReferenceAcceptance(
        DeterministicDirectionalGate.from_dict(_mapping("deterministic_directional", profiles.get("deterministic_directional"))),
        MonteCarloGate.from_dict(_mapping("monte_carlo", profiles.get("monte_carlo"))),
        ImageGate.from_dict(_mapping("linear_hdr_image", profiles.get("linear_hdr_image"))),
        ImageGate.from_dict(_mapping("linear_hdr_textured_image", profiles.get("linear_hdr_textured_image"))),
    )


def deterministic_directional_metrics(
    native: Sequence[Sequence[float]] | np.ndarray,
    candidate: Sequence[Sequence[float]] | np.ndarray,
    gate: DeterministicDirectionalGate,
) -> DeterministicDirectionalMetrics:
    native_array = np.asarray(native, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if native_array.shape != candidate_array.shape or native_array.ndim != 2 or native_array.shape[1] != 3:
        raise ValueError("directional results must have matching [N, 3] shapes")
    if native_array.shape[0] == 0 or not np.isfinite(native_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("directional results must be nonempty and finite")
    numerator = np.sum(np.abs(native_array - candidate_array), axis=1)
    denominator = np.maximum(
        np.maximum(np.sum(np.abs(native_array), axis=1), np.sum(np.abs(candidate_array), axis=1)),
        gate.absolute_floor,
    )
    relative = numerator / denominator
    max_absolute = float(np.max(np.abs(native_array - candidate_array)))
    scaled_absolute = np.abs(native_array - candidate_array) / np.maximum(
        np.maximum(np.abs(native_array), np.abs(candidate_array)), 1.0
    )
    max_scaled_absolute = float(np.max(scaled_absolute))
    median = float(np.median(relative))
    p99 = float(np.quantile(relative, 0.99))
    maximum = float(np.max(relative))
    passed = (
        median <= gate.median_relative_l1_max
        and p99 <= gate.p99_relative_l1_max
        and maximum <= gate.max_relative_l1_max
        and max_scaled_absolute <= gate.max_scaled_absolute_error
    )
    return DeterministicDirectionalMetrics(
        native_array.shape[0], median, p99, maximum, max_absolute, max_scaled_absolute, passed
    )


def linear_hdr_image_metrics(
    native: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    candidate: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    gate: ImageGate,
    *,
    mask: Sequence[Sequence[bool]] | np.ndarray | None = None,
) -> ImageMetrics:
    native_array = np.asarray(native, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if native_array.shape != candidate_array.shape or native_array.ndim != 3 or native_array.shape[2] != 3:
        raise ValueError("linear HDR images must have matching [H, W, 3] shapes")
    if not np.isfinite(native_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("linear HDR images must be finite")
    if mask is None:
        selected = np.ones(native_array.shape[:2], dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != native_array.shape[:2]:
            raise ValueError("linear HDR image mask must match image height and width")
    if not np.any(selected):
        raise ValueError("linear HDR image mask must select at least one pixel")

    native_pixels = native_array[selected]
    candidate_pixels = candidate_array[selected]
    difference = np.abs(native_pixels - candidate_pixels)
    numerator = np.sum(difference, axis=1)
    denominator = np.maximum(
        np.maximum(np.sum(np.abs(native_pixels), axis=1), np.sum(np.abs(candidate_pixels), axis=1)),
        gate.luminance_floor,
    )
    relative = numerator / denominator
    mean_squared_error = float(np.mean(np.square(native_pixels - candidate_pixels)))
    peak = max(float(np.max(np.abs(native_pixels))), float(np.max(np.abs(candidate_pixels))), 1.0)
    psnr = float("inf") if mean_squared_error == 0.0 else float(
        20.0 * np.log10(peak / np.sqrt(mean_squared_error))
    )
    p95 = float(np.quantile(relative, 0.95))
    mae = float(np.mean(difference))
    passed = (
        p95 <= gate.p95_relative_l1_max
        and psnr >= gate.linear_psnr_min_db
        and mae <= gate.max_absolute_mae
    )
    return ImageMetrics(int(np.count_nonzero(selected)), p95, psnr, mae, peak, passed)
