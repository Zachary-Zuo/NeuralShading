from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ncls.core.identity import sha256_file, sha256_json
from ncls.core.material import LayerStackIR, MaterialProgram, canonicalize_layer_stack
from ncls.core.source import SourceSnapshot
from ncls.data import DataExecutionPlan, PipelineTrace
from ncls.learning.source_adaptation import (
    DenseNativeAssetCollection,
    MaterialXNativeAssetCollection,
    NativeAssetCollection,
    NativeAssetRole,
    encode_layer_stack_native_features,
    encode_mdl_fixed_native_features,
    layer_stack_native_feature_layout,
    materialx_native_feature_layout,
    mdl_fixed_native_feature_layout,
)
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.mdl_metal_assets import MdlMetalNativeAssetCollection
from ncls.source_materials.mdl import MdlMaterialSource
from ncls.source_materials.mdl_metal import (
    MdlMetalRegistry,
    PARAMETER_RESPONSIBILITIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MDL_METAL_REGISTRY_PATH = (
    PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
)


class MethodSourceAdapter(ABC):
    method_key: str
    family_id: str
    source_contract_version: int
    adapter_id: str
    implementation_sha256: str

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        values = tuple(snapshots)
        if not values:
            raise ValueError("method source adapter requires source snapshots")
        if any(
            snapshot.family_id != self.family_id
            or snapshot.source_contract_version != self.source_contract_version
            for snapshot in values
        ):
            raise ValueError("method source adapter received an incompatible snapshot")
        self.snapshots = values
        self.device = device

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "adapter_id": self.adapter_id,
                "implementation_sha256": self.implementation_sha256,
                "source_snapshot_ids": [value.snapshot_id for value in self.snapshots],
            }
        )

    @abstractmethod
    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        raise NotImplementedError

    def configure_data_execution(
        self, plan: DataExecutionPlan, trace: PipelineTrace
    ) -> None:
        del plan, trace

    def prefetch_host(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> None:
        del candidates, request

    def execution_source_indices(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> tuple[int, ...]:
        del request
        values = tuple(int(value) for value in candidates)
        if not values:
            raise ValueError("source execution cohort cannot be empty")
        return values

    def close(self) -> None:
        pass

    @abstractmethod
    def native_assets(self) -> NativeAssetCollection:
        raise NotImplementedError


class NvidiaLayerStackSourceAdapter(MethodSourceAdapter):
    method_key = "nvidia-neural-appearance"
    family_id = "ncls.layer-stack@1"
    source_contract_version = 1
    adapter_id = "nvidia.layer-stack-native@2"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        stacks = []
        for snapshot in self.snapshots:
            if isinstance(snapshot.native_object, LayerStackIR):
                stacks.append(snapshot.native_object)
            else:
                stacks.append(
                    canonicalize_layer_stack(
                        MaterialProgram.from_json(snapshot.native_payload.decode("utf-8"))
                    )
                )
        values = np.stack([encode_layer_stack_native_features(stack) for stack in stacks])
        self._feature_table = torch.as_tensor(
            values, dtype=torch.float32, device=device
        )
        self._layout_id = layer_stack_native_feature_layout().layout_id
        self._native_assets = DenseNativeAssetCollection(
            tuple(
                (torch.from_numpy(values[index : index + 1, None, :]).clone(),)
                for index in range(len(self.snapshots))
            ),
            tuple(snapshot.snapshot_id for snapshot in self.snapshots),
            self._layout_id,
            "constant",
            "constant",
            "clamp",
            (
                NativeAssetRole(
                    "encoder-input",
                    "layer-stack-native-records",
                    0,
                    self._feature_table.shape[1],
                    "linear",
                    "constant",
                ),
            ),
        )

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del generator, options, execution_source_indices
        count = int(source_index.shape[0])
        return (
            {
                "uv": torch.zeros((count, 2), dtype=torch.float32, device=self.device),
                "uv_dx": torch.zeros((count, 2), dtype=torch.float32, device=self.device),
                "uv_dy": torch.zeros((count, 2), dtype=torch.float32, device=self.device),
                "mip_level": torch.zeros(count, dtype=torch.float32, device=self.device),
                "native_features": self._feature_table[source_index],
            },
            {"native_feature_layout_id": self._layout_id},
        )

    def native_assets(self) -> NativeAssetCollection:
        return self._native_assets


class NvidiaMaterialXSourceAdapter(MethodSourceAdapter):
    method_key = "nvidia-neural-appearance"
    family_id = "materialx.document@1.39.4"
    source_contract_version = 1
    adapter_id = "nvidia.materialx-standard-surface-spatial@2"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        if len(self.snapshots) != 1:
            raise RuntimeError("NVIDIA materialization trains one source snapshot per run")
        snapshot = self.snapshots[0]
        inputs = snapshot.editor_metadata.get("resolved_inputs")
        paths = snapshot.editor_metadata.get("resource_paths")
        if not isinstance(inputs, bytes) or not isinstance(paths, Mapping):
            raise ValueError("MaterialX snapshot is missing canonical runtime bindings")
        constants = np.frombuffer(inputs, dtype=np.float32).copy()
        self._assets = MaterialXNativeAssetCollection.from_textures(
            constants,
            base_color=_path(paths.get("base-color")),
            roughness=_path(paths.get("roughness")),
            metalness=_path(paths.get("metalness")),
            normal=_path(paths.get("normal")),
            asset_id=snapshot.snapshot_id,
        )
        self._layout_id = materialx_native_feature_layout().layout_id

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del options, execution_source_indices
        count = int(source_index.shape[0])
        uv = torch.rand((count, 2), generator=generator, device=self.device)
        level_shapes = self._assets.descriptors[0].domain("surface-uv").level_shapes
        maximum_mip = len(level_shapes) - 1
        exponential = -torch.log(
            torch.clamp(
                1.0 - torch.rand(count, generator=generator, device=self.device),
                min=1e-7,
            )
        )
        mip_level = torch.clamp(exponential, max=float(maximum_mip))
        texel_extent = max(level_shapes[0])
        footprint = torch.pow(2.0, mip_level) / float(texel_extent)
        uv_dx = torch.stack((footprint, torch.zeros_like(footprint)), dim=1)
        uv_dy = torch.stack((torch.zeros_like(footprint), footprint), dim=1)
        return (
            {
                "uv": uv,
                "uv_dx": uv_dx,
                "uv_dy": uv_dy,
                "mip_level": mip_level,
                "native_features": self._assets.sample_torch(uv, mip_level),
            },
            {"native_feature_layout_id": self._layout_id},
        )

    def native_assets(self) -> NativeAssetCollection:
        return self._assets


class NvidiaMdlFixedSourceAdapter(MethodSourceAdapter):
    method_key = "nvidia-neural-appearance"
    family_id = "mdl.program@1"
    source_contract_version = 1
    adapter_id = "nvidia.mdl-fixed-uniform@1"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        if len(self.snapshots) != 1:
            raise RuntimeError("NVIDIA MDL fixed-uniform training requires one snapshot")
        from ncls.source_materials.mdl import MdlMaterialSource

        source = MdlMaterialSource.from_snapshot(self.snapshots[0])
        values, schema_identity = encode_mdl_fixed_native_features(source.arguments)
        self._feature_table = torch.as_tensor(
            values[None, :], dtype=torch.float32, device=device
        )
        self._layout_id = mdl_fixed_native_feature_layout().layout_id
        self._schema_identity = schema_identity
        self._native_assets = DenseNativeAssetCollection(
            ((torch.from_numpy(values[None, None, :]).clone(),),),
            (self.snapshots[0].snapshot_id,),
            self._layout_id,
            "constant",
            "constant",
            "clamp",
            (
                NativeAssetRole(
                    "encoder-input",
                    "mdl-typed-parameters",
                    0,
                    self._feature_table.shape[1],
                    "signed-bounded",
                    "constant",
                ),
            ),
        )

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        del generator, options, execution_source_indices
        if bool((source_index != 0).any()):
            raise ValueError("MDL fixed-uniform adapter accepts only source index zero")
        count = int(source_index.shape[0])
        zeros = torch.zeros((count, 2), dtype=torch.float32, device=self.device)
        return (
            {
                "uv": zeros,
                "uv_dx": zeros.clone(),
                "uv_dy": zeros.clone(),
                "mip_level": torch.zeros(
                    count, dtype=torch.float32, device=self.device
                ),
                "native_features": self._feature_table.expand(count, -1),
            },
            {
                "native_feature_layout_id": self._layout_id,
                "mdl_parameter_schema_identity": self._schema_identity,
            },
        )

    def native_assets(self) -> NativeAssetCollection:
        return self._native_assets


_METAL_TYPE_IDS = {
    "float": 0,
    "double": 0,
    "color": 1,
    "float2": 2,
    "float3": 3,
    "float4": 4,
    "bool": 5,
    "enum": 6,
    "int": 7,
}


def _components(value: object) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    if isinstance(value, bool):
        return (1.0 if value else 0.0,)
    if isinstance(value, (int, float)):
        return (float(value),)
    return (0.0,)


def _normalized_components(parameter: Mapping[str, Any], value: object) -> np.ndarray:
    result = np.zeros(4, dtype=np.float32)
    values = _components(value)
    defaults = _components(parameter.get("value", 0.0))
    minimum = parameter.get("minimum", parameter.get("soft_minimum"))
    maximum = parameter.get("maximum", parameter.get("soft_maximum"))
    minimums = _components(minimum) if minimum is not None else ()
    maximums = _components(maximum) if maximum is not None else ()
    for index, item in enumerate(values[:4]):
        if minimums and maximums:
            lower = minimums[min(index, len(minimums) - 1)]
            upper = maximums[min(index, len(maximums) - 1)]
        else:
            lower = upper = None
        if lower is not None and upper is not None and upper > lower:
            normalized = 2.0 * (item - lower) / (
                upper - lower
            ) - 1.0
        else:
            default = defaults[min(index, len(defaults) - 1)]
            scale = max(1.0, abs(default))
            normalized = np.tanh((item - default) / scale)
        result[index] = np.float32(np.clip(normalized, -4.0, 4.0))
    return result


class MetalFusedMdlSourceAdapter(MethodSourceAdapter):
    method_key = "metal-fused-neural-material"
    family_id = "mdl.program@1"
    source_contract_version = 1
    adapter_id = "metal-fused.mdl-vmaterials2-metal@1"
    implementation_sha256 = sha256_file(Path(__file__))

    def __init__(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> None:
        super().__init__(snapshots, device)
        self.registry = MdlMetalRegistry.load(MDL_METAL_REGISTRY_PATH)
        sources = tuple(MdlMaterialSource.from_snapshot(snapshot) for snapshot in self.snapshots)
        roots = {source.module_root for source in sources}
        if len(roots) != 1:
            raise ValueError("one Metal run requires one canonical vMaterials module root")
        self._assets = MdlMetalNativeAssetCollection(
            self.registry, next(iter(roots)), working_set_capacity=32
        )
        self._asset_indices = {
            descriptor.asset_id: index
            for index, descriptor in enumerate(self._assets.descriptors)
        }
        graph_indices = {value: index for index, value in enumerate(sorted(self.registry.graphs))}
        schema_indices = {
            value: index for index, value in enumerate(sorted(self.registry.parameter_schemas))
        }
        recipe_indices = {value: index for index, value in enumerate(sorted(self.registry.recipes))}
        metals = sorted({record.metal for record in self.registry.exports})
        finishes = sorted({record.finish for record in self.registry.exports})
        metal_indices = {value: index for index, value in enumerate(metals)}
        finish_indices = {value: index for index, value in enumerate(finishes)}
        semantic_keys = sorted(
            {
                (str(parameter["name"]), str(parameter["type"]))
                for record in self.registry.exports
                for parameter in record.parameters
            }
        )
        semantic_indices = {value: index for index, value in enumerate(semantic_keys)}
        if len(semantic_indices) != 154:
            raise ValueError("Metal typed semantic table drifted from the opaque audit")
        tables: dict[str, list[np.ndarray | int]] = {
            "graph": [],
            "schema": [],
            "recipe": [],
            "metal": [],
            "finish": [],
            "asset": [],
            "semantic": [],
            "type": [],
            "responsibility": [],
            "discrete": [],
            "continuous": [],
            "presence": [],
            "optical": [],
            "access": [],
            "frame": [],
            "distribution": [],
        }
        maximum_mips = []
        maximum_extents = []
        maximum_extents_xy = []
        for source in sources:
            record = self.registry.resolve_exact_locator(source.module, source.export)
            tables["graph"].append(graph_indices[record.graph_id])
            tables["schema"].append(schema_indices[record.parameter_schema_id])
            tables["recipe"].append(recipe_indices[record.recipe_id])
            tables["metal"].append(metal_indices[record.metal])
            tables["finish"].append(finish_indices[record.finish])
            asset_index = self._asset_indices[record.texture_set_id]
            tables["asset"].append(asset_index)
            encoded = self._encode_parameters(record.parameters, source.arguments, semantic_indices)
            for name in ("semantic", "type", "responsibility", "discrete", "continuous", "presence"):
                tables[name].append(encoded[name])
            tables["optical"].append(self._canonical_optical(source.arguments))
            tables["access"].append(self._access_state(source.arguments))
            tables["frame"].append(self._frame_state(source.arguments))
            texture_set = self.registry.texture_sets[record.texture_set_id]
            tables["distribution"].append(
                int(
                    any(
                        "beckmann" in str(slot.get("source_path", "")).lower()
                        for slot in texture_set["slots"]
                    )
                )
            )
            descriptor = self._assets.descriptors[asset_index]
            maximum_mips.append(max(len(domain.level_shapes) for domain in descriptor.domains) - 1)
            maximum_extents.append(max(max(domain.level_shapes[0]) for domain in descriptor.domains))
            maximum_extents_xy.append(
                (
                    max(domain.level_shapes[0][1] for domain in descriptor.domains),
                    max(domain.level_shapes[0][0] for domain in descriptor.domains),
                )
            )
        integer_names = {
            "graph",
            "schema",
            "recipe",
            "metal",
            "finish",
            "asset",
            "semantic",
            "type",
            "responsibility",
            "discrete",
            "presence",
            "distribution",
        }
        self._tables = {
            name: torch.as_tensor(
                np.asarray(values),
                dtype=torch.int64 if name in integer_names else torch.float32,
                device=device,
            )
            for name, values in tables.items()
        }
        self._maximum_mips = torch.tensor(
            maximum_mips, dtype=torch.float32, device=device
        )
        self._maximum_extents = torch.tensor(
            maximum_extents, dtype=torch.float32, device=device
        )
        self._maximum_extents_xy = torch.tensor(
            maximum_extents_xy, dtype=torch.float32, device=device
        )
        self._source_asset_indices = tuple(int(value) for value in tables["asset"])

    def configure_data_execution(
        self, plan: DataExecutionPlan, trace: PipelineTrace
    ) -> None:
        self._assets.enable_gpu_sampling(
            self.device,
            budget_bytes=plan.residency_budget_bytes,
            trace=trace,
            num_workers=plan.num_workers,
            host_prefetch=plan.host_prefetch,
        )

    def prefetch_host(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> None:
        execution_sources = self.execution_source_indices(candidates, request)
        self._assets.prefetch_gpu_sampling(
            tuple(
                dict.fromkeys(
                    self._source_asset_indices[source_index]
                    for source_index in execution_sources
                )
            )
        )

    def close(self) -> None:
        self._assets.close()

    def execution_source_indices(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> tuple[int, ...]:
        cohorts: dict[int, list[int]] = {}
        for value in candidates:
            source_index = int(value)
            try:
                asset_index = self._source_asset_indices[source_index]
            except IndexError as error:
                raise ValueError("Metal execution group source index is out of range") from error
            cohorts.setdefault(asset_index, []).append(source_index)
        if not cohorts:
            raise ValueError("Metal source execution cohort cannot be empty")
        asset_indices = tuple(sorted(cohorts))
        selector = int(
            sha256_json(
                {
                    "schema": "ncls.metal-resource-cohort@1",
                    "global_step": request.global_step,
                    "candidate_sources": [int(value) for value in candidates],
                }
            )[:16],
            16,
        )
        return tuple(cohorts[asset_indices[selector % len(asset_indices)]])

    @staticmethod
    def _encode_parameters(
        parameters: Sequence[Mapping[str, Any]],
        arguments: Mapping[str, Mapping[str, Any]],
        semantic_indices: Mapping[tuple[str, str], int],
    ) -> Mapping[str, np.ndarray]:
        maximum = 32
        semantic = np.zeros(maximum, dtype=np.int64)
        type_id = np.zeros(maximum, dtype=np.int64)
        responsibility = np.zeros(maximum, dtype=np.int64)
        discrete = np.zeros(maximum, dtype=np.int64)
        continuous = np.zeros((maximum, 4), dtype=np.float32)
        presence = np.zeros(maximum, dtype=np.int64)
        if len(parameters) > maximum:
            raise ValueError("Metal export exceeds the frozen typed-token bound")
        for index, parameter in enumerate(parameters):
            name = str(parameter["name"])
            kind = str(parameter["type"])
            current = arguments.get(name, {}).get("value", parameter.get("value"))
            semantic[index] = semantic_indices[(name, kind)]
            type_id[index] = _METAL_TYPE_IDS[kind]
            responsibility[index] = PARAMETER_RESPONSIBILITIES.index(
                str(parameter["responsibility"])
            )
            continuous[index] = _normalized_components(parameter, current)
            if kind == "bool":
                discrete[index] = int(bool(current))
            elif kind == "enum":
                choices = tuple(arguments.get(name, {}).get("choices", ()))
                choice_names = [str(item.get("name")) for item in choices]
                discrete[index] = (
                    choice_names.index(str(current)) if str(current) in choice_names else 0
                )
            elif kind == "int":
                discrete[index] = max(0, int(current))
            presence[index] = 1
        return {
            "semantic": semantic,
            "type": type_id,
            "responsibility": responsibility,
            "discrete": discrete,
            "continuous": continuous,
            "presence": presence,
        }

    @staticmethod
    def _argument(arguments: Mapping[str, Mapping[str, Any]], names: Sequence[str], default: object) -> object:
        for name in names:
            if name in arguments:
                return arguments[name].get("value", default)
        return default

    @classmethod
    def _canonical_optical(cls, arguments: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
        result = np.zeros(16, dtype=np.float32)
        color = _components(
            cls._argument(
                arguments,
                ("metal_color", "metal_tint", "normal_reflectivity", "color_1"),
                (0.7, 0.7, 0.7),
            )
        )
        result[: min(3, len(color))] = color[:3]
        grazing = _components(cls._argument(arguments, ("grazing_reflectivity",), color))
        result[3 : 3 + min(3, len(grazing))] = grazing[:3]
        scalar_names = (
            "roughness",
            "metal_roughness",
            "reflection_roughness",
            "steel_anisotropy",
            "brushing_anisotropy",
            "reflection_brightness",
            "metalness",
            "paint_roughness",
            "oxide_roughness",
            "polish_film_strength",
        )
        for offset, name in enumerate(scalar_names, start=6):
            if offset >= 16:
                break
            result[offset] = float(_components(cls._argument(arguments, (name,), 0.0))[0])
        return result

    @classmethod
    def _access_state(cls, arguments: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
        scale = _components(cls._argument(arguments, ("texture_scale",), (1.0, 1.0)))
        translate = _components(cls._argument(arguments, ("texture_translate",), (0.0, 0.0)))
        angle = math.radians(
            float(_components(cls._argument(arguments, ("texture_rotate",), 0.0))[0])
        )
        result = np.zeros(16, dtype=np.float32)
        result[0:2] = (scale + (1.0, 1.0))[:2]
        result[2:4] = (translate + (0.0, 0.0))[:2]
        result[4:6] = (math.cos(angle), math.sin(angle))
        result[6] = float(bool(cls._argument(arguments, ("infinite_tiling",), True)))
        result[7] = float(bool(cls._argument(arguments, ("no_uv",), False)))
        result[8] = float(_components(cls._argument(arguments, ("uv_space_index",), 0))[0])
        result[9] = float(_components(cls._argument(arguments, ("scale",), 1.0))[0])
        return result

    @classmethod
    def _frame_state(cls, arguments: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
        result = np.zeros(8, dtype=np.float32)
        result[0] = float(bool(cls._argument(arguments, ("enable_round_corners", "roundcorners_enable"), False)))
        result[1] = float(_components(cls._argument(arguments, ("radius", "radius_mm", "roundcorner_radius", "roundcorners_radius_mm"), 0.0))[0])
        result[2] = float(bool(cls._argument(arguments, ("across_materials", "roundcorners_across_materials"), False)))
        result[3] = float(bool(cls._argument(arguments, ("object_scaled_bump",), False)))
        return result

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "adapter_id": self.adapter_id,
                "implementation_sha256": self.implementation_sha256,
                "registry_identity": self.registry.identity,
                "asset_collection_identity": self._assets.collection_id,
                "source_snapshot_ids": [value.snapshot_id for value in self.snapshots],
            }
        )

    def asset_index_for_source(self, source_index: int = 0) -> int:
        if not 0 <= source_index < len(self.snapshots):
            raise ValueError("Metal source index is out of range")
        return int(self._tables["asset"][source_index].item())

    def compiler_tensors_for_source(
        self, source_index: int = 0, *, device: torch.device | None = None
    ) -> Mapping[str, torch.Tensor]:
        """Return one complete typed compiler record without sampling a training batch."""

        if not 0 <= source_index < len(self.snapshots):
            raise ValueError("Metal source index is out of range")
        target = self.device if device is None else device
        names = {
            "metal_graph_index": "graph",
            "metal_schema_index": "schema",
            "metal_recipe_index": "recipe",
            "metal_identity_index": "metal",
            "metal_finish_index": "finish",
            "metal_asset_index": "asset",
            "metal_typed_semantic_id": "semantic",
            "metal_typed_type_id": "type",
            "metal_typed_responsibility_id": "responsibility",
            "metal_typed_discrete": "discrete",
            "metal_typed_continuous": "continuous",
            "metal_typed_presence": "presence",
            "metal_canonical_optical": "optical",
            "metal_access_state": "access",
            "metal_frame_state": "frame",
            "metal_distribution_id": "distribution",
        }
        return {
            output: self._tables[source][source_index : source_index + 1]
            .detach()
            .to(target)
            .contiguous()
            for output, source in names.items()
        }

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        count = int(source_index.shape[0])
        extent = float(options.get("spatial_tile_extent", 1.0))
        coherent = bool(options.get("asset_tile_coherent", True))
        patch_size = int(options.get("source_patch_size", 16))
        if not 0.0 < extent <= 1.0:
            raise ValueError("Metal spatial tile extent must lie in (0,1]")
        if coherent:
            anchor_value = options.get("spatial_anchor")
            if anchor_value is None:
                origin = torch.rand((1, 2), generator=generator, device=self.device)
            else:
                if (
                    not isinstance(anchor_value, (list, tuple))
                    or len(anchor_value) != 2
                    or any(not 0.0 <= float(value) <= 1.0 for value in anchor_value)
                ):
                    raise ValueError("Metal spatial anchor must be a normalized float2")
                origin = torch.tensor(
                    anchor_value, dtype=torch.float32, device=self.device
                )[None]
            uv = torch.frac(
                origin
                + extent
                * (
                    torch.rand((count, 2), generator=generator, device=self.device)
                    - (0.5 if anchor_value is not None else 0.0)
                )
            )
        else:
            uv = torch.rand((count, 2), generator=generator, device=self.device)
        maximum_mip = self._maximum_mips.index_select(0, source_index)
        footprint_recipe = options.get("footprint_recipe")
        if footprint_recipe == "balanced-zero-one-four-texel@1":
            footprint_texels = torch.tensor(
                (0.0, 1.0, 4.0), dtype=torch.float32, device=self.device
            ).repeat((count + 2) // 3)[:count]
            footprint_texels = footprint_texels[
                torch.randperm(count, generator=generator, device=self.device)
            ]
            exponential = torch.where(
                footprint_texels > 0.0,
                torch.log2(torch.clamp(footprint_texels, min=1.0)),
                torch.zeros_like(footprint_texels),
            )
        elif footprint_recipe is None:
            exponential = -torch.log(
                torch.clamp(
                    1.0 - torch.rand(count, generator=generator, device=self.device),
                    min=1e-7,
                )
            )
            footprint_texels = torch.pow(2.0, exponential)
        else:
            raise ValueError("Metal footprint recipe is unsupported")
        access = self._tables["access"].index_select(0, source_index)
        scale = access[:, 0:2]
        # The reference receives the renderer footprint before authored UV
        # transforms, while the asset codec addresses the transformed texture.
        # Select the adjacent source mips from the conservative transformed
        # footprint so texture_scale does not silently desynchronize GT and
        # neural filtering.
        lod_scale = torch.clamp(torch.amax(torch.abs(scale), dim=1), min=1e-6)
        surface_mip = exponential
        mip_level = torch.clamp(
            surface_mip + torch.log2(lod_scale), min=0.0
        )
        mip_level = torch.minimum(mip_level, maximum_mip)
        extent_pixels = self._maximum_extents.index_select(0, source_index)
        footprint = footprint_texels / extent_pixels
        uv_dx = torch.stack((footprint, torch.zeros_like(footprint)), dim=1)
        uv_dy = torch.stack((torch.zeros_like(footprint), footprint), dim=1)
        cosine, sine = access[:, 4:5], access[:, 5:6]
        scaled = uv * scale
        access_uv = torch.stack(
            (
                cosine[:, 0] * scaled[:, 0] - sine[:, 0] * scaled[:, 1],
                sine[:, 0] * scaled[:, 0] + cosine[:, 0] * scaled[:, 1],
            ),
            dim=1,
        ) + access[:, 2:4]
        access_uv = torch.where(
            (access[:, 6:7] > 0.5), torch.frac(access_uv), access_uv
        )
        asset_index = self._tables["asset"].index_select(0, source_index)
        if execution_source_indices is None:
            active_asset_indices = None
        else:
            try:
                active_asset_indices = tuple(
                    sorted(
                        {
                            self._source_asset_indices[int(index)]
                            for index in execution_source_indices
                        }
                    )
                )
            except IndexError as error:
                raise ValueError("Metal execution group source index is out of range") from error
        patches, slot_mask, role_class = self._assets.sample_local_patches(
            asset_index,
            access_uv,
            mip_level,
            patch_size=patch_size,
            active_asset_indices=active_asset_indices,
        )
        tensors = {
            "uv": uv,
            "uv_dx": uv_dx,
            "uv_dy": uv_dy,
            "mip_level": mip_level,
            "metal_mip_fraction": torch.frac(mip_level),
            "metal_texture_patches": patches,
            "metal_texture_slot_mask": slot_mask,
            "metal_texture_role_class": role_class,
            "metal_graph_index": self._tables["graph"].index_select(0, source_index),
            "metal_schema_index": self._tables["schema"].index_select(0, source_index),
            "metal_recipe_index": self._tables["recipe"].index_select(0, source_index),
            "metal_identity_index": self._tables["metal"].index_select(0, source_index),
            "metal_finish_index": self._tables["finish"].index_select(0, source_index),
            "metal_asset_index": asset_index,
            "metal_typed_semantic_id": self._tables["semantic"].index_select(0, source_index),
            "metal_typed_type_id": self._tables["type"].index_select(0, source_index),
            "metal_typed_responsibility_id": self._tables["responsibility"].index_select(0, source_index),
            "metal_typed_discrete": self._tables["discrete"].index_select(0, source_index),
            "metal_typed_continuous": self._tables["continuous"].index_select(0, source_index),
            "metal_typed_presence": self._tables["presence"].index_select(0, source_index),
            "metal_canonical_optical": self._tables["optical"].index_select(0, source_index),
            "metal_access_state": access,
            "metal_frame_state": self._tables["frame"].index_select(0, source_index),
            "metal_distribution_id": self._tables["distribution"].index_select(0, source_index),
        }
        return tensors, {
            "metal_registry_identity": self.registry.identity,
            "native_asset_collection_identity": self._assets.collection_id,
            "metal_typed_layout_identity": sha256_json(
                {
                    "schema": "ncls.metal-fused-typed-tensors@1",
                    "semantic_count": 154,
                    "maximum_tokens": 32,
                }
            ),
        }

    def native_assets(self) -> NativeAssetCollection:
        return self._assets


def _balanced_one_native_texel_offsets(
    extent_pixels_xy: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    if (
        extent_pixels_xy.ndim != 2
        or extent_pixels_xy.shape[1] != 2
        or extent_pixels_xy.shape[0] % 2
        or not bool(torch.isfinite(extent_pixels_xy).all())
        or not bool((extent_pixels_xy >= 1.0).all())
    ):
        raise ValueError(
            "balanced native-texel offsets require finite even [batch,2] extents"
        )
    count = int(extent_pixels_xy.shape[0])
    axis = torch.arange(count, device=extent_pixels_xy.device) % 2
    axis = axis[
        torch.randperm(count, generator=generator, device=extent_pixels_xy.device)
    ]
    one_texel_xy = torch.reciprocal(extent_pixels_xy)
    return torch.stack(
        (
            torch.where(
                axis == 0,
                one_texel_xy[:, 0],
                torch.zeros_like(one_texel_xy[:, 0]),
            ),
            torch.where(
                axis == 1,
                one_texel_xy[:, 1],
                torch.zeros_like(one_texel_xy[:, 1]),
            ),
        ),
        dim=1,
    )


class MetalBudgetedMdlSourceAdapter(MetalFusedMdlSourceAdapter):
    """Metal budgeted 复用同一 source audit，并增加 paired-UV 在线查询字段。"""

    method_key = "metal-budgeted-neural-material"
    adapter_id = "metal-budgeted.mdl-vmaterials2-metal@1"

    def sample_tensors(
        self,
        source_index: torch.Tensor,
        generator: torch.Generator,
        options: Mapping[str, Any],
        *,
        execution_source_indices: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
        tensors, provenance = super().sample_tensors(
            source_index,
            generator,
            options,
            execution_source_indices=execution_source_indices,
        )
        if not bool(options.get("paired_uv", False)):
            return tensors, provenance
        count = int(source_index.shape[0])
        extent_pixels_xy = self._maximum_extents_xy.index_select(0, source_index)
        paired_recipe = options.get("paired_uv_recipe")
        if paired_recipe != "one-native-texel-axis-balanced@1":
            raise ValueError("Metal paired UV recipe is unsupported")
        offset = _balanced_one_native_texel_offsets(
            extent_pixels_xy, generator
        )
        paired_uv = tensors["uv"] + offset
        access = tensors["metal_access_state"]
        scale = access[:, 0:2]
        cosine, sine = access[:, 4:5], access[:, 5:6]
        scaled = paired_uv * scale
        paired_access_uv = torch.stack(
            (
                cosine[:, 0] * scaled[:, 0] - sine[:, 0] * scaled[:, 1],
                sine[:, 0] * scaled[:, 0] + cosine[:, 0] * scaled[:, 1],
            ),
            dim=1,
        ) + access[:, 2:4]
        paired_access_uv = torch.where(
            access[:, 6:7] > 0.5, torch.frac(paired_access_uv), paired_access_uv
        )
        if execution_source_indices is None:
            active_asset_indices = None
        else:
            active_asset_indices = tuple(
                sorted(
                    {
                        self._source_asset_indices[int(index)]
                        for index in execution_source_indices
                    }
                )
            )
        paired_patches, paired_mask, paired_roles = self._assets.sample_local_patches(
            tensors["metal_asset_index"],
            paired_access_uv,
            tensors["mip_level"],
            patch_size=int(options.get("source_patch_size", 16)),
            active_asset_indices=active_asset_indices,
        )
        if not torch.equal(paired_mask, tensors["metal_texture_slot_mask"]):
            raise RuntimeError("paired Metal asset slot mask drifted")
        if not torch.equal(paired_roles, tensors["metal_texture_role_class"]):
            raise RuntimeError("paired Metal asset role table drifted")
        return {
            **tensors,
            "paired_uv": paired_uv,
            "paired_uv_dx": tensors["uv_dx"],
            "paired_uv_dy": tensors["uv_dy"],
            "metal_paired_texture_patches": paired_patches,
        }, {
            **provenance,
            "paired_uv_recipe": "one-native-texel-axis-balanced@1",
        }


def _path(value: object) -> Path | None:
    return None if value is None else Path(str(value)).resolve()


__all__ = [
    "MetalBudgetedMdlSourceAdapter",
    "MetalFusedMdlSourceAdapter",
    "MethodSourceAdapter",
    "NvidiaLayerStackSourceAdapter",
    "NvidiaMaterialXSourceAdapter",
    "NvidiaMdlFixedSourceAdapter",
]
