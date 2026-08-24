from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.contract import (
    QUERY_ROLE_NAMES,
    PositionKind,
    QueryPlan,
    ReferenceDescriptor,
    SourceState,
    SurfaceSample,
)
from ncls.data.directions import (
    equal_area_hemisphere,
    equal_area_sphere,
    grazing_anchored_view_directions,
    peak_grazing_mixture_query,
    polar_band_view_directions,
)
from ncls.data.surfaces import uv_surface_samples
from ncls.paths import PROJECT_ROOT


def implementation_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little") + relative)
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()


def assign_group_splits(group_ids: Sequence[str], seed: int) -> Mapping[str, int]:
    unique = tuple(dict.fromkeys(group_ids))
    if not unique:
        raise ValueError("split assignment requires at least one group")
    if len(unique) < 3:
        return {name: 0 for name in unique}
    order = sorted(
        unique,
        key=lambda name: hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest(),
    )
    validation_count = max(1, round(0.1 * len(order)))
    test_count = max(1, round(0.1 * len(order)))
    if validation_count + test_count >= len(order):
        validation_count = test_count = 1
    result = {name: 0 for name in order}
    for name in order[:validation_count]:
        result[name] = 1
    for name in order[validation_count : validation_count + test_count]:
        result[name] = 2
    return result


class BaseProvider:
    descriptor: ReferenceDescriptor

    def __init__(self, config: CollectionConfig) -> None:
        self.config = config

    def query_plan(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample] = (),
    ) -> QueryPlan:
        full_sphere = self.descriptor.incident_domain == "full-sphere"
        domain = "full-sphere" if full_sphere else "upper-hemisphere"
        query_seed = self.config.seed ^ int(state.state_id[:16], 16)
        role = QUERY_ROLE_NAMES.index(self.config.query_role)
        partition_index = role + (
            state.split * len(QUERY_ROLE_NAMES)
            if self.config.split_direction_scramble
            else 0
        )
        role_offset = np.mod(partition_index * 0.2718281828459045, 2.0) * np.pi
        regular_view_count = self.config.view_count - self.config.transmission_view_count
        regular_views = grazing_anchored_view_directions(
            regular_view_count,
            min_theta_degrees=self.config.grazing_min_degrees,
            max_theta_degrees=self.config.grazing_max_degrees,
            grazing_fraction=self.config.grazing_view_fraction,
            azimuth_offset=role_offset,
        )
        if self.config.transmission_view_count:
            if not full_sphere:
                raise ValueError("transmission-relevant views require a full-sphere reference")
            critical_views = polar_band_view_directions(
                self.config.transmission_view_count,
                self.config.critical_view_min_degrees,
                self.config.critical_view_max_degrees,
                azimuth_offset=role_offset + 0.5 * np.pi,
            )
            views = np.concatenate((regular_views, critical_views))
        else:
            views = regular_views
        role_seed = query_seed ^ ((role + 1) * 0x9E3779B1)
        if self.config.proposal != "uniform":
            lights, weights, pdf = peak_grazing_mixture_query(
                views,
                self.config.light_count,
                full_sphere=full_sphere,
                seed=role_seed,
                component_weights=self.config.mixture_weights,
                critical_band_abs_cosine=(
                    self.config.critical_wi_abs_cosine_min,
                    self.config.critical_wi_abs_cosine_max,
                ),
            )
            transmission = "-transmission" if full_sphere else ""
            proposal = f"{self.config.query_role}-peak-aware-{domain}{transmission}"
        else:
            measure = 4.0 * np.pi if full_sphere else 2.0 * np.pi
            direction_rows = []
            for view_index in range(self.config.view_count):
                azimuth = 0.5 * role_offset + view_index * 0.173 * np.pi
                directions, _ = (
                    equal_area_sphere(self.config.light_count, azimuth_offset=azimuth)
                    if full_sphere
                    else equal_area_hemisphere(self.config.light_count, azimuth_offset=azimuth)
                )
                direction_rows.append(directions)
            lights = np.stack(direction_rows)
            weights = np.full(
                (self.config.view_count, self.config.light_count),
                measure / self.config.light_count,
                dtype=np.float32,
            )
            pdf = np.full_like(weights, 1.0 / measure)
            proposal = f"{self.config.query_role}-uniform-{domain}"
        return QueryPlan(
            views,
            lights,
            weights,
            pdf,
            (proposal,) * len(views),
            query_seed,
            np.full(len(views), role, dtype=np.uint8),
        )

    def surface_samples(self, state: SourceState) -> Sequence[SurfaceSample]:
        if self.descriptor.position_kind == PositionKind.CONSTANT:
            return (SurfaceSample(),)
        if self.descriptor.position_kind == PositionKind.UV:
            return uv_surface_samples(
                self.config.spatial_sample_count,
                self.config.footprint_width,
                self.config.seed ^ int(state.state_id[-16:], 16),
                self.config.surface_profile,
            )
        raise NotImplementedError("surface-point providers must define their own surface sampler")

    def metadata(self) -> Mapping[str, Any]:
        return {
            **asdict(self.descriptor),
            "position_kind": int(self.descriptor.position_kind),
        }

    def close(self) -> None:
        pass
