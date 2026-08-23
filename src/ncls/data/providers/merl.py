from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.contract import EvaluatedBlock, PositionKind, QueryPlan, ReferenceDescriptor, SourceState, SurfaceSample, make_state_id
from ncls.source_materials import MerlBrdfReference, MerlMaterial

from .base import BaseProvider, PROJECT_ROOT, assign_group_splits, implementation_hash
from .falcor import execute_direction_kernel, import_falcor, structured_buffer


@dataclass(frozen=True)
class MerlProviderConfig:
    material_ids: tuple[str, ...] = ()
    asset_root: Path = PROJECT_ROOT / "data/source-materials/merl-brdf/v1"
    material_index: Path = PROJECT_ROOT / "references/merl-brdf-v1/materials.json"


class MerlProvider(BaseProvider):
    def __init__(self, collection: CollectionConfig, config: MerlProviderConfig = MerlProviderConfig()) -> None:
        super().__init__(collection)
        self.provider_config = config
        shader = PROJECT_ROOT / "shaders/ncls/data/reference_merl.cs.slang"
        self.descriptor = ReferenceDescriptor(
            "merl.measured-brdf@1",
            "ncls.merl-brdf@1",
            "ncls.merl-material@1",
            incident_domain="upper-hemisphere",
            position_kind=PositionKind.CONSTANT,
            deterministic=True,
            capabilities=("evaluate", "measured-table"),
            implementation_sha256=implementation_hash((
                Path(__file__),
                PROJECT_ROOT / "src/ncls/source_materials/merl.py",
                shader,
                PROJECT_ROOT / "shaders/ncls/reference/merl_reference.slang",
            )),
        )
        document = json.loads(config.material_index.read_text(encoding="utf-8"))
        records = {str(record["material_id"]): record for record in document["materials"]}
        selected = config.material_ids or tuple(records)
        missing = sorted(set(selected) - set(records))
        if missing:
            raise ValueError(f"unknown MERL material IDs: {missing}")
        splits = assign_group_splits(selected, collection.seed)
        states = []
        for material_id in selected:
            record = records[material_id]
            table = config.asset_root / str(record["table_uri"])
            if not table.is_file() or table.stat().st_size != int(record["size"]):
                raise FileNotFoundError(f"MERL source table is missing or truncated: {table}")
            material = MerlMaterial(material_id, str(record["table_uri"]), document["source_record"], document["license"])
            payload = material.to_json().encode("utf-8")
            source_hash = str(record["sha256"])
            states.append(SourceState(
                make_state_id(self.descriptor.family_id, self.descriptor.native_schema_id, payload, source_hash),
                self.descriptor.family_id,
                self.descriptor.reference_id,
                material_id,
                material_id,
                self.descriptor.native_schema_id,
                payload,
                str(record["table_uri"]),
                source_hash,
                splits[material_id],
                material,
            ))
        self._states = tuple(states)
        self._falcor = None
        self._device = None
        self._compute = None

    def source_states(self) -> Sequence[SourceState]:
        return self._states

    def _runtime(self):
        if self._compute is None:
            self._falcor = import_falcor()
            self._device = self._falcor.Device(type=self._falcor.DeviceType.D3D12)
            self._compute = self._falcor.ComputePass(
                self._device,
                file=PROJECT_ROOT / "shaders/ncls/data/reference_merl.cs.slang",
                cs_entry="evaluateReference",
            )
        return self._falcor, self._device, self._compute

    def evaluate(self, state: SourceState, surfaces: Sequence[SurfaceSample], plan: QueryPlan) -> EvaluatedBlock:
        falcor, device, compute = self._runtime()
        reference = MerlBrdfReference(state.runtime_state, self.provider_config.asset_root)
        compute.globals.gBrdfTable = structured_buffer(device, falcor, reference.gpu_table(), 12)
        response = execute_direction_kernel(
            compute, device, falcor, plan.view_directions, plan.light_directions, len(surfaces)
        )
        return EvaluatedBlock.deterministic(response)

    def metadata(self):
        return {
            **super().metadata(),
            "material_index": self.provider_config.material_index.relative_to(PROJECT_ROOT).as_posix(),
            "material_count": len(self._states),
        }

    def close(self) -> None:
        self._compute = None
        self._device = None
        self._falcor = None
