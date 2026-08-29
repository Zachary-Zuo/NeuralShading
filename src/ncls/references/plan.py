from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ncls.core.identity import require_sha256, sha256_bytes, sha256_json
from ncls.core.scattering import MaterialPayload, ReferenceProgramDefinition
from ncls.core.source import SourceSnapshot


@dataclass(frozen=True)
class ReferenceMaterialRecord:
    """plan 中一个全局 source identity 到 group-local material 的映射。"""

    global_source_index: int
    local_material_index: int
    snapshot: SourceSnapshot
    material: MaterialPayload
    argument_block_offset: int = 0
    read_only_data_offset: int = 0

    def __post_init__(self) -> None:
        if min(
            self.global_source_index,
            self.local_material_index,
            self.argument_block_offset,
            self.read_only_data_offset,
        ) < 0:
            raise ValueError("reference material record indices and offsets must be nonnegative")
        if self.material.source_snapshot_id != self.snapshot.snapshot_id:
            raise ValueError("reference material record payload belongs to another snapshot")

    def to_identity_dict(self) -> dict[str, object]:
        return {
            "global_source_index": self.global_source_index,
            "local_material_index": self.local_material_index,
            "source_snapshot_id": self.snapshot.snapshot_id,
            "material_payload": {
                "blobs": {
                    name: {
                        "sha256": sha256_bytes(payload),
                        "descriptor": dict(self.material.blob_descriptors[name]),
                    }
                    for name, payload in self.material.blobs.items()
                },
                "resources": {
                    name: {
                        "sha256": sha256_bytes(payload),
                        "descriptor": dict(self.material.resource_descriptors[name]),
                    }
                    for name, payload in self.material.resources.items()
                },
                "samplers": {
                    name: dict(value)
                    for name, value in self.material.sampler_descriptors.items()
                },
            },
            "argument_block_offset": self.argument_block_offset,
            "read_only_data_offset": self.read_only_data_offset,
        }


@dataclass(frozen=True)
class ReferenceExecutionGroup:
    group_id: str
    definition: ReferenceProgramDefinition
    records: tuple[ReferenceMaterialRecord, ...]

    def __post_init__(self) -> None:
        require_sha256("reference execution group_id", self.group_id)
        records = tuple(self.records)
        if not records:
            raise ValueError("reference execution group requires material records")
        if tuple(record.local_material_index for record in records) != tuple(range(len(records))):
            raise ValueError("reference execution group local material indices must be dense and ordered")
        if len({record.global_source_index for record in records}) != len(records):
            raise ValueError("reference execution group global source indices must be unique")
        for record in records:
            self.definition.validate_snapshot(record.snapshot)
            if (
                self.definition.execution_group_key(record.snapshot, record.material)
                != self.group_id
            ):
                raise ValueError("reference material record disagrees with execution group identity")
        object.__setattr__(self, "records", records)

    @property
    def snapshots(self) -> tuple[SourceSnapshot, ...]:
        return tuple(record.snapshot for record in self.records)

    @property
    def global_source_indices(self) -> tuple[int, ...]:
        return tuple(record.global_source_index for record in self.records)

    def to_identity_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "reference_program": self.definition.descriptor.to_dict(),
            "records": [record.to_identity_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ReferenceExecutionPlan:
    groups: tuple[ReferenceExecutionGroup, ...]
    query_recipe: Mapping[str, object]
    schema_name: str = "ncls.reference-execution-plan"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.reference-execution-plan" or self.schema_version != 1:
            raise ValueError("unsupported ReferenceExecutionPlan schema")
        groups = tuple(self.groups)
        if not groups or len({group.group_id for group in groups}) != len(groups):
            raise ValueError("reference execution plan requires unique groups")
        records = sorted(
            (record for group in groups for record in group.records),
            key=lambda record: record.global_source_index,
        )
        if tuple(record.global_source_index for record in records) != tuple(range(len(records))):
            raise ValueError("reference execution plan global source indices must be dense and ordered")
        if len({record.snapshot.snapshot_id for record in records}) != len(records):
            raise ValueError("reference execution plan source snapshots must be unique")
        recipe = dict(self.query_recipe)
        if not recipe:
            raise ValueError("reference execution plan requires a query recipe")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "query_recipe", recipe)

    @property
    def records(self) -> tuple[ReferenceMaterialRecord, ...]:
        return tuple(
            sorted(
                (record for group in self.groups for record in group.records),
                key=lambda record: record.global_source_index,
            )
        )

    @property
    def snapshots(self) -> tuple[SourceSnapshot, ...]:
        return tuple(record.snapshot for record in self.records)

    @property
    def source_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(snapshot.snapshot_id for snapshot in self.snapshots)

    @property
    def identity(self) -> str:
        return sha256_json(self.to_identity_dict())

    def to_identity_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "groups": [group.to_identity_dict() for group in self.groups],
            "query_recipe": dict(self.query_recipe),
        }

    def group(self, group_id: str) -> ReferenceExecutionGroup:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(f"unknown reference execution group {group_id!r}")


def compile_reference_execution_plan(
    entries: Iterable[tuple[ReferenceProgramDefinition, SourceSnapshot]],
    *,
    query_recipe: Mapping[str, object],
) -> ReferenceExecutionPlan:
    """把跨 source/program 的有序 snapshot 集合编译为唯一 canonical plan。"""

    values = tuple(entries)
    if not values:
        raise ValueError("reference execution plan requires source entries")
    grouped: dict[
        tuple[str, str],
        list[tuple[int, ReferenceProgramDefinition, SourceSnapshot, MaterialPayload]],
    ] = {}
    order: list[tuple[str, str]] = []
    for global_index, (definition, snapshot) in enumerate(values):
        definition.validate_snapshot(snapshot)
        material = definition.compile_material(snapshot)
        group_id = definition.execution_group_key(snapshot, material)
        key = (definition.descriptor.descriptor_sha256, group_id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((global_index, definition, snapshot, material))
    groups = []
    for key in order:
        rows = grouped[key]
        definition = rows[0][1]
        if any(row[1].descriptor != definition.descriptor for row in rows):
            raise ValueError("one reference execution group cannot mix program descriptors")
        layouts = definition.execution_group_layout(tuple(row[3] for row in rows))
        if len(layouts) != len(rows):
            raise ValueError("reference execution group layout count mismatch")
        records = tuple(
            ReferenceMaterialRecord(
                global_index,
                local_index,
                snapshot,
                material,
                int(layouts[local_index][0]),
                int(layouts[local_index][1]),
            )
            for local_index, (global_index, _, snapshot, material) in enumerate(rows)
        )
        groups.append(ReferenceExecutionGroup(key[1], definition, records))
    return ReferenceExecutionPlan(tuple(groups), query_recipe)


def compile_single_program_plan(
    definition: ReferenceProgramDefinition,
    snapshots: Sequence[SourceSnapshot],
    *,
    query_recipe: Mapping[str, object],
) -> ReferenceExecutionPlan:
    return compile_reference_execution_plan(
        ((definition, snapshot) for snapshot in snapshots),
        query_recipe=query_recipe,
    )


__all__ = [
    "ReferenceExecutionGroup",
    "ReferenceExecutionPlan",
    "ReferenceMaterialRecord",
    "compile_reference_execution_plan",
    "compile_single_program_plan",
]
