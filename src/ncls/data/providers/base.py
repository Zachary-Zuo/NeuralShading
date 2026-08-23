from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.contract import (
    QUERY_ROLE_NAMES,
    PositionKind,
    QueryPlan,
    QueryRole,
    ReferenceDescriptor,
    SourceState,
    SurfaceSample,
)
from ncls.data.directions import (
    MIXTURE_QUERY_PROFILE_ID,
    equal_area_hemisphere,
    equal_area_sphere,
    peak_grazing_mixture_query,
    stratified_view_directions,
)
from ncls.data.surfaces import uv_surface_samples
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

    def query_plan(
        self,
        state: SourceState,
        surfaces: Sequence[SurfaceSample] = (),
    ) -> QueryPlan:
        full_sphere = self.descriptor.incident_domain == "full-sphere"
        domain = "full-sphere" if full_sphere else "upper-hemisphere"
        query_seed = self.config.seed ^ int(state.state_id[:16], 16)
        role_counts = (
            self.config.view_count,
            self.config.validation_view_count,
            self.config.test_view_count,
            self.config.adversarial_view_count,
        )
        view_parts: list[np.ndarray] = []
        light_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        pdf_parts: list[np.ndarray] = []
        proposal_ids: list[str] = []
        query_roles: list[int] = []
        for role, count in enumerate(role_counts):
            if count == 0:
                continue
            role_name = QUERY_ROLE_NAMES[role]
            partition_index = role + (
                state.split * len(QUERY_ROLE_NAMES)
                if self.config.split_direction_scramble
                else 0
            )
            role_offset = np.mod(partition_index * 0.2718281828459045, 2.0) * np.pi
            views = stratified_view_directions(count, azimuth_offset=role_offset)
            role_seed = query_seed ^ ((role + 1) * 0x9E3779B1)
            use_mixture = (
                self.config.query_profile_id == MIXTURE_QUERY_PROFILE_ID
                and role in {int(QueryRole.TRAIN), int(QueryRole.ADVERSARIAL_PROBE)}
            )
            if use_mixture:
                lights, weights, pdf = peak_grazing_mixture_query(
                    views,
                    self.config.light_count,
                    full_sphere=full_sphere,
                    seed=role_seed,
                )
                transmission = "-transmission-critical" if full_sphere else ""
                proposal = f"{role_name}-uniform-peak-grazing-{domain}{transmission}@2"
            else:
                measure = 4.0 * np.pi if full_sphere else 2.0 * np.pi
                direction_rows = []
                for view_index in range(count):
                    azimuth = 0.5 * role_offset + view_index * 0.173 * np.pi
                    directions, _ = (
                        equal_area_sphere(self.config.light_count, azimuth_offset=azimuth)
                        if full_sphere
                        else equal_area_hemisphere(self.config.light_count, azimuth_offset=azimuth)
                    )
                    direction_rows.append(directions)
                lights = np.stack(direction_rows)
                weights = np.full((count, self.config.light_count), measure / self.config.light_count, dtype=np.float32)
                pdf = np.full((count, self.config.light_count), 1.0 / measure, dtype=np.float32)
                proposal = f"{role_name}-uniform-solid-angle-{domain}-independent@1"
            view_parts.append(views)
            light_parts.append(lights)
            weight_parts.append(weights)
            pdf_parts.append(pdf)
            proposal_ids.extend((proposal,) * count)
            query_roles.extend((role,) * count)
        return QueryPlan(
            np.concatenate(view_parts),
            np.concatenate(light_parts),
            np.concatenate(weight_parts),
            np.concatenate(pdf_parts),
            proposal_ids,
            query_seed,
            query_roles,
        )

    def surface_samples(self, state: SourceState) -> Sequence[SurfaceSample]:
        if self.descriptor.position_kind == PositionKind.CONSTANT:
            return (SurfaceSample(),)
        if self.descriptor.position_kind == PositionKind.UV:
            return uv_surface_samples(
                self.config.spatial_sample_count,
                self.config.footprint_width,
                self.config.seed ^ int(state.state_id[-16:], 16),
                self.config.surface_profile_id,
            )
        raise NotImplementedError("surface-point providers must define their own surface sampler")

    def metadata(self) -> Mapping[str, Any]:
        return {
            **asdict(self.descriptor),
            "position_kind": int(self.descriptor.position_kind),
        }

    def close(self) -> None:
        pass
