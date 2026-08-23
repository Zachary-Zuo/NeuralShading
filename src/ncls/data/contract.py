from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


SPLIT_NAMES = ("train", "validation", "test")


class PositionKind(IntEnum):
    CONSTANT = 0
    UV = 1
    SURFACE_POINT = 2


def _unit_vectors(name: str, values: np.ndarray, *, ndim: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != ndim or result.shape[-1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape [..., 3] and contain finite values")
    lengths = np.linalg.norm(result, axis=-1)
    if np.any(np.abs(lengths - 1.0) > 2e-4):
        raise ValueError(f"{name} must contain normalized directions")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class ReferenceDescriptor:
    family_id: str
    reference_id: str
    native_schema_id: str
    query_profile_id: str = "ncls.local-surface-rgb@1"
    incident_domain: str = "upper-hemisphere"
    position_kind: PositionKind = PositionKind.CONSTANT
    deterministic: bool = True
    capabilities: tuple[str, ...] = ("evaluate",)
    implementation_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("family_id", "reference_id", "native_schema_id", "query_profile_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.incident_domain not in {"upper-hemisphere", "full-sphere"}:
            raise ValueError("incident_domain must be upper-hemisphere or full-sphere")
        if self.implementation_sha256 and (
            len(self.implementation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.implementation_sha256)
        ):
            raise ValueError("implementation_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SourceState:
    state_id: str
    family_id: str
    reference_id: str
    asset_id: str
    split_group_id: str
    native_schema_id: str
    native_payload: bytes
    source_uri: str
    source_sha256: str
    split: int
    runtime_state: Any = field(repr=False, compare=False)
    parent_state_id: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.split < len(SPLIT_NAMES):
            raise ValueError("split must be train, validation or test")
        if not self.native_payload:
            raise ValueError("native_payload must preserve the exact sampled state")
        for name in (
            "state_id", "family_id", "reference_id", "asset_id", "split_group_id",
            "native_schema_id", "source_sha256",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        for name in ("state_id", "source_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def make_state_id(
    family_id: str,
    native_schema_id: str,
    native_payload: bytes,
    source_sha256: str,
) -> str:
    digest = hashlib.sha256()
    for value in (
        family_id.encode("utf-8"),
        native_schema_id.encode("utf-8"),
        native_payload,
        source_sha256.encode("ascii"),
    ):
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class SurfaceSample:
    position_kind: PositionKind = PositionKind.CONSTANT
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    uv: tuple[float, float] = (0.0, 0.0)
    uv_dx: tuple[float, float] = (0.0, 0.0)
    uv_dy: tuple[float, float] = (0.0, 0.0)
    geometric_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    geometric_tangent: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        values = (*self.position, *self.uv, *self.uv_dx, *self.uv_dy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("surface sample coordinates must be finite")
        normal = np.asarray(self.geometric_normal, dtype=np.float64)
        tangent = np.asarray(self.geometric_tangent, dtype=np.float64)
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-4):
            raise ValueError("geometric_normal must be normalized")
        if not np.isclose(np.linalg.norm(tangent), 1.0, atol=1e-4):
            raise ValueError("geometric_tangent must be normalized")
        if abs(float(np.dot(normal, tangent))) > 1e-4:
            raise ValueError("geometric normal and tangent must be orthogonal")


@dataclass(frozen=True)
class QueryPlan:
    view_directions: np.ndarray
    light_directions: np.ndarray
    solid_angle_weights: np.ndarray
    proposal_pdf: np.ndarray
    proposal_id: str
    seed: int

    def __post_init__(self) -> None:
        views = _unit_vectors("view_directions", self.view_directions, ndim=2)
        lights = _unit_vectors("light_directions", self.light_directions, ndim=2)
        if len(views) < 1 or len(lights) < 1:
            raise ValueError("QueryPlan requires at least one view and light direction")
        weights = np.asarray(self.solid_angle_weights, dtype=np.float32)
        pdf = np.asarray(self.proposal_pdf, dtype=np.float32)
        if weights.shape != (len(lights),) or pdf.shape != (len(lights),):
            raise ValueError("direction weights and proposal PDF must match light directions")
        if np.any(weights <= 0.0) or np.any(pdf <= 0.0) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(pdf)):
            raise ValueError("direction weights and proposal PDF must be positive and finite")
        if not self.proposal_id or self.seed < 0:
            raise ValueError("query proposal identity must be nonempty and seed nonnegative")
        object.__setattr__(self, "view_directions", views)
        object.__setattr__(self, "light_directions", lights)
        object.__setattr__(self, "solid_angle_weights", np.ascontiguousarray(weights))
        object.__setattr__(self, "proposal_pdf", np.ascontiguousarray(pdf))


@dataclass(frozen=True)
class EvaluatedBlock:
    mean: np.ndarray
    variance: np.ndarray
    replica_mean_a: np.ndarray
    replica_mean_b: np.ndarray
    sample_count: np.ndarray
    valid: np.ndarray
    event_flags: np.ndarray
    reference_pdf: np.ndarray
    rng_seed: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        if mean.ndim != 4 or mean.shape[-1] != 3:
            raise ValueError("reference mean must have shape [surface, view, light, 3]")
        for name in ("variance", "replica_mean_a", "replica_mean_b"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != mean.shape:
                raise ValueError(f"{name} must match reference mean")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, np.ascontiguousarray(value))
        sample_count = np.asarray(self.sample_count, dtype=np.uint32)
        valid = np.asarray(self.valid, dtype=np.uint8)
        event_flags = np.asarray(self.event_flags, dtype=np.uint32)
        reference_pdf = np.asarray(self.reference_pdf, dtype=np.float32)
        rng_seed = np.asarray(self.rng_seed, dtype=np.uint64)
        scalar_shape = mean.shape[:-1]
        for name, value in (
            ("sample_count", sample_count), ("valid", valid),
            ("event_flags", event_flags), ("reference_pdf", reference_pdf),
            ("rng_seed", rng_seed),
        ):
            if value.shape != scalar_shape:
                raise ValueError(f"{name} must have shape {scalar_shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(reference_pdf)):
            raise ValueError("reference output must contain finite values")
        if np.any(np.asarray(self.variance) < 0.0) or np.any(reference_pdf < 0.0):
            raise ValueError("reference variance and PDF must be nonnegative")
        if np.any(sample_count < 1):
            raise ValueError("every reference query must record at least one sample")
        if np.any(valid > 1):
            raise ValueError("valid must contain only zero or one")
        object.__setattr__(self, "mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "sample_count", np.ascontiguousarray(sample_count))
        object.__setattr__(self, "valid", np.ascontiguousarray(valid))
        object.__setattr__(self, "event_flags", np.ascontiguousarray(event_flags))
        object.__setattr__(self, "reference_pdf", np.ascontiguousarray(reference_pdf))
        object.__setattr__(self, "rng_seed", np.ascontiguousarray(rng_seed))

    @classmethod
    def deterministic(
        cls,
        mean: np.ndarray,
        *,
        valid: np.ndarray | None = None,
        event_flags: np.ndarray | None = None,
        reference_pdf: np.ndarray | None = None,
        rng_seed: np.ndarray | None = None,
    ) -> "EvaluatedBlock":
        value = np.asarray(mean, dtype=np.float32)
        scalar_shape = value.shape[:-1]
        return cls(
            value,
            np.zeros_like(value),
            value,
            value,
            np.ones(scalar_shape, dtype=np.uint32),
            np.ones(scalar_shape, dtype=np.uint8) if valid is None else valid,
            np.ones(scalar_shape, dtype=np.uint32) if event_flags is None else event_flags,
            np.zeros(scalar_shape, dtype=np.float32) if reference_pdf is None else reference_pdf,
            np.zeros(scalar_shape, dtype=np.uint64) if rng_seed is None else rng_seed,
        )


class ReferenceProvider(Protocol):
    descriptor: ReferenceDescriptor

    def source_states(self) -> Sequence[SourceState]: ...

    def surface_samples(self, state: SourceState) -> Sequence[SurfaceSample]: ...

    def query_plan(self, state: SourceState) -> QueryPlan: ...

    def evaluate(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample],
        plan: QueryPlan,
    ) -> EvaluatedBlock: ...

    def metadata(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...
