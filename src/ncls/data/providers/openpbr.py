from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.contract import EvaluatedBlock, PositionKind, QueryPlan, ReferenceDescriptor, SourceState, SurfaceSample, make_state_id
from ncls.source_materials import OpenPBRMaterial, load_openpbr_luts, resolve_openpbr_inputs
from ncls.source_materials.openpbr import ACESCG_TO_LINEAR_SRGB

from .base import BaseProvider, PROJECT_ROOT, assign_group_splits, implementation_hash
from .falcor import direction_rows, import_falcor, output_buffer, structured_buffer


@dataclass(frozen=True)
class OpenPBRProviderConfig:
    material_ids: tuple[str, ...] = ()
    source_root: Path = PROJECT_ROOT / "external/OpenPBR"
    material_index: Path = PROJECT_ROOT / "references/openpbr-1.1.1-v1/materials.json"


class OpenPBRProvider(BaseProvider):
    def __init__(self, collection: CollectionConfig, config: OpenPBRProviderConfig = OpenPBRProviderConfig()) -> None:
        super().__init__(collection)
        self.provider_config = config
        shader = PROJECT_ROOT / "shaders/ncls/data/reference_openpbr.cs.slang"
        self.descriptor = ReferenceDescriptor(
            "openpbr.surface@1.1.1",
            "ncls.openpbr@1.1.1",
            "ncls.openpbr-material@1",
            incident_domain="full-sphere",
            position_kind=PositionKind.CONSTANT,
            deterministic=True,
            capabilities=("evaluate", "sample", "pdf", "transmission"),
            implementation_sha256=implementation_hash((
                Path(__file__),
                PROJECT_ROOT / "src/ncls/source_materials/openpbr.py",
                shader,
                PROJECT_ROOT / "shaders/ncls/reference/openpbr_reference.slang",
            )),
        )
        document = json.loads(config.material_index.read_text(encoding="utf-8"))
        records = {str(record["material_id"]): record for record in document["materials"]}
        selected = config.material_ids or tuple(records)
        missing = sorted(set(selected) - set(records))
        if missing:
            raise ValueError(f"unknown OpenPBR material IDs: {missing}")
        splits = assign_group_splits(selected, collection.seed)
        states = []
        for material_id in selected:
            record = records[material_id]
            path = config.source_root / str(record["document"])
            if not path.is_file() or path.stat().st_size != int(record["size"]):
                raise FileNotFoundError(f"OpenPBR source document is missing or truncated: {path}")
            material = OpenPBRMaterial.from_materialx(path)
            material = replace(material, source_document=str(record["document"]).replace("\\", "/"))
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
                str(record["document"]),
                source_hash,
                splits[material_id],
                material,
            ))
        self._states = tuple(states)
        self._falcor = None
        self._device = None
        self._compute = None
        self._resources = []

    def source_states(self) -> Sequence[SourceState]:
        return self._states

    def _runtime(self):
        if self._compute is not None:
            return self._falcor, self._device, self._compute
        falcor = import_falcor()
        device = falcor.Device(type=falcor.DeviceType.D3D12)
        compute = falcor.ComputePass(
            device,
            file=PROJECT_ROOT / "shaders/ncls/data/reference_openpbr.cs.slang",
            cs_entry="evaluateReference",
        )
        lut_data = load_openpbr_luts(PROJECT_ROOT / "external/openpbr-bsdf")

        def texture_2d(data: np.ndarray):
            texture = device.create_texture(
                width=data.shape[1], height=data.shape[0], format=falcor.ResourceFormat.RGBA32Float,
                mip_levels=1, bind_flags=falcor.ResourceBindFlags.ShaderResource,
            )
            texture.from_numpy(np.ascontiguousarray(data))
            self._resources.append(texture)
            return texture

        def texture_3d(data: np.ndarray):
            texture = device.create_texture(
                width=data.shape[2], height=data.shape[1], depth=data.shape[0],
                format=falcor.ResourceFormat.RGBA32Float, mip_levels=1,
                bind_flags=falcor.ResourceBindFlags.ShaderResource,
            )
            texture.from_numpy(np.ascontiguousarray(data))
            self._resources.append(texture)
            return texture

        compute.globals.gOpenPbrIdealDielectricEnergy = texture_3d(lut_data.ideal_dielectric_energy)
        compute.globals.gOpenPbrIdealDielectricAverage = texture_2d(lut_data.ideal_dielectric_average)
        compute.globals.gOpenPbrIdealDielectricRatio = texture_2d(lut_data.ideal_dielectric_ratio)
        compute.globals.gOpenPbrOpaqueDielectricEnergy = texture_3d(lut_data.opaque_dielectric_energy)
        compute.globals.gOpenPbrOpaqueDielectricAverage = texture_2d(lut_data.opaque_dielectric_average)
        compute.globals.gOpenPbrIdealMetalEnergy = texture_2d(lut_data.ideal_metal_energy)
        compute.globals.gOpenPbrIdealMetalAverage = texture_2d(lut_data.ideal_metal_average)
        compute.globals.gOpenPbrLtc = texture_2d(lut_data.ltc)
        compute.globals.gOpenPbrLutSampler = device.create_sampler(
            address_mode_u=falcor.TextureAddressingMode.Clamp,
            address_mode_v=falcor.TextureAddressingMode.Clamp,
            address_mode_w=falcor.TextureAddressingMode.Clamp,
        )
        self._falcor, self._device, self._compute = falcor, device, compute
        return falcor, device, compute

    def evaluate(self, state: SourceState, surfaces: Sequence[SurfaceSample], plan: QueryPlan) -> EvaluatedBlock:
        falcor, device, compute = self._runtime()
        material: OpenPBRMaterial = state.runtime_state
        flat = np.ascontiguousarray(resolve_openpbr_inputs(material, asset_root=self.provider_config.source_root))
        compute.globals.gResolvedInputs = structured_buffer(device, falcor, flat, 4)
        view_rows, light_rows = direction_rows(plan.view_directions, plan.light_directions, len(surfaces))
        query_count = len(view_rows)
        compute.globals.gViews = structured_buffer(device, falcor, view_rows, 16)
        compute.globals.gLights = structured_buffer(device, falcor, light_rows, 16)
        output = output_buffer(device, falcor, query_count)
        compute.globals.gOutput = output
        compute.globals.gQueryCount = query_count
        compute.execute(threads_x=query_count)
        rows = output.to_numpy().view(np.float32).reshape(query_count, 4).copy()
        response = rows[:, :3]
        if material.color_space == "acescg":
            response = response @ ACESCG_TO_LINEAR_SRGB.T
        shape = (len(surfaces), len(plan.view_directions), len(plan.light_directions))
        response = response.reshape(*shape, 3).astype(np.float32)
        pdf = rows[:, 3].reshape(shape).astype(np.float32)
        lights = np.broadcast_to(plan.light_directions, (*shape, 3))
        event_flags = np.where(lights[..., 2] >= 0.0, 1, 2).astype(np.uint32)
        return EvaluatedBlock.deterministic(response, event_flags=event_flags, reference_pdf=pdf)

    def metadata(self):
        return {
            **super().metadata(),
            "material_index": self.provider_config.material_index.relative_to(PROJECT_ROOT).as_posix(),
            "material_count": len(self._states),
            "output_color_conversion": "ACEScg source responses are converted to linear-sRGB",
        }

    def close(self) -> None:
        self._resources.clear()
        self._compute = None
        self._device = None
        self._falcor = None
