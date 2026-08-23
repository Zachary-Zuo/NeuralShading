from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.contract import PositionKind, QueryPlan, ReferenceDescriptor, SourceState, SurfaceSample
from ncls.data.directions import (
    equal_area_hemisphere,
    equal_area_sphere,
    peak_grazing_mixture_query,
    stratified_uv,
    stratified_view_directions,
)
from ncls.paths import PROJECT_ROOT
from ncls.source_materials.identity import sha256_file



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

    def query_plan(self, state: SourceState) -> QueryPlan:
        split_offset = (
            (0.0, 0.371, 0.743)[state.split] * np.pi
            if self.config.split_direction_scramble
            else 0.0
        )
        views = stratified_view_directions(self.config.view_count, azimuth_offset=split_offset)
        if self.config.query_profile_id == "ncls.e0-peak-grazing-mixture@1":
            full_sphere = self.descriptor.incident_domain == "full-sphere"
            query_seed = self.config.seed ^ int(state.state_id[:16], 16)
            lights, weights, pdf = peak_grazing_mixture_query(
                views,
                self.config.light_count,
                full_sphere=full_sphere,
                seed=query_seed,
            )
            split_name = ("train", "validation", "test")[state.split]
            domain = "full-sphere-transmission-critical" if full_sphere else "upper-hemisphere"
            return QueryPlan(
                views,
                lights,
                weights,
                pdf,
                f"uniform-peak-grazing-{domain}-{split_name}@1",
                query_seed,
            )
        if self.descriptor.incident_domain == "full-sphere":
            lights, weights = equal_area_sphere(self.config.light_count, azimuth_offset=0.5 * split_offset)
            measure = 4.0 * np.pi
            proposal = "uniform-solid-angle-full-sphere-split-scrambled@2"
        else:
            lights, weights = equal_area_hemisphere(self.config.light_count, azimuth_offset=0.5 * split_offset)
            measure = 2.0 * np.pi
            proposal = "uniform-solid-angle-upper-hemisphere-split-scrambled@3"
        return QueryPlan(
            views,
            lights,
            weights,
            np.full(len(lights), 1.0 / measure, dtype=np.float32),
            proposal,
            self.config.seed ^ int(state.state_id[:16], 16),
        )

    def surface_samples(self, state: SourceState) -> Sequence[SurfaceSample]:
        if self.descriptor.position_kind == PositionKind.CONSTANT:
            return (SurfaceSample(),)
        if self.descriptor.position_kind == PositionKind.UV:
            uv_values = stratified_uv(
                self.config.spatial_sample_count,
                self.config.seed ^ int(state.state_id[-16:], 16),
            )
            width = self.config.footprint_width
            return tuple(
                SurfaceSample(
                    PositionKind.UV,
                    uv=(float(uv[0]), float(uv[1])),
                    uv_dx=(width, 0.0),
                    uv_dy=(0.0, width),
                )
                for uv in uv_values
            )
        raise NotImplementedError("surface-point providers must define their own surface sampler")

    def metadata(self) -> Mapping[str, Any]:
        return {
            **asdict(self.descriptor),
            "position_kind": int(self.descriptor.position_kind),
        }

    def close(self) -> None:
        pass
