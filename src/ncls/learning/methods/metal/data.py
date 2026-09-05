from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from ncls.core.identity import sha256_file, sha256_json
from ncls.core.source import SourceSnapshot
from ncls.data import DataExecutionPlan, PipelineTrace
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.learning.batches import TrainingRouteRequest
from ncls.learning.methods.metal.native_assets import MdlMetalNativeAssetCollection
from ncls.source_materials.mdl import MdlMaterialSource
from ncls.source_materials.mdl_metal import MdlMetalRegistry, PARAMETER_RESPONSIBILITIES
from ncls.paths import PROJECT_ROOT

from ncls.learning.source_adapters import MethodSourceAdapter
from ncls.learning.conditioning_resources import AdaptedConditioning
from ncls.learning.conditioning_resources import ConditioningResources
from ncls.learning.methods.metal.native_uv import group_compatible_uv, native_slot_mappings
from ncls.learning.methods.metal.spatial_bundle import build_spatial_bundle


MDL_METAL_REGISTRY_PATH = PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"

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

class MetalBudgetedMdlSourceAdapter(MethodSourceAdapter):
    method_key = "metal-budgeted-neural-material"
    family_id = "mdl.program@1"
    source_contract_version = 1
    adapter_id = "metal-spatial.mdl-native-uv-groups@1"
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
        spatial_contracts = []
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
            raw_slots = self._assets.raw_slots(asset_index)
            parameters = {str(parameter["name"]): parameter.get("value") for parameter in record.parameters}
            parameters.update({name: value.get("value") for name, value in source.arguments.items()})
            module_file = source.module_root.joinpath(*source.module.strip(":").split("::")).with_suffix(".mdl")
            paths = {position: str(slot["source_path"]) for position, slot in enumerate(texture_set["slots"])
                     if raw_slots[position].spatial}
            groups = group_compatible_uv(native_slot_mappings(module_file, parameters, paths))
            spatial_contracts.append((asset_index, raw_slots, groups))
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
        self._spatial_contracts = tuple(spatial_contracts)
        self._spatial_cohort_keys = tuple(sha256_json({"asset": asset,
            "groups": [group.mapping.identity for group in groups]}) for asset, _, groups in spatial_contracts)
        self._spatial_tile_schedules: dict[str, Any] = {}
        self._spatial_splits: dict[str, Any] = {}

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
        # raw RF 资源按 CPU cohort/request 计划在 consume 时交给同一 host pipeline。
        # 不预取旧的整张 coarse-mip pyramid，也不在 lookahead 中创建 GPU 资源。
        return None

    def close(self) -> None:
        self._assets.close()

    def execution_source_indices(
        self,
        candidates: Sequence[int],
        request: TrainingRouteRequest,
    ) -> tuple[int, ...]:
        cohorts: dict[str, list[int]] = {}
        for value in candidates:
            source_index = int(value)
            try:
                asset_index = self._spatial_cohort_keys[source_index]
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

    def native_assets(self) -> NativeAssetCollection:
        return self._assets

    def spatial_contract_for_source(self, source_index: int = 0):
        if not 0 <= source_index < len(self.snapshots):
            raise ValueError("Metal source index is out of range")
        return self._spatial_contracts[source_index]

    def sample_tensors(
        self, source_index: torch.Tensor, generator: torch.Generator,
        options: Mapping[str, Any], *, execution_source_indices: Sequence[int] | None = None,
    ) -> AdaptedConditioning:
        from ncls.learning.methods.metal.spatial_schedule import spatial_cohort

        candidates = tuple(range(len(self.snapshots))) if execution_source_indices is None else tuple(execution_source_indices)
        if not candidates or any(index < 0 or index >= len(self.snapshots) for index in candidates):
            raise ValueError("Metal execution cohort has invalid source indices")
        if len({self._spatial_cohort_keys[index] for index in candidates}) != 1:
            raise ValueError("raw Metal sampling requires one CPU-declared UV/resource cohort")
        asset_index, slots, groups = self._spatial_contracts[candidates[0]]
        cohort = spatial_cohort(slots, groups, options)
        frozen_plan = None
        split_identity = None
        if options.get("spatial_split_recipe") == "heldout-rf-tiles@1":
            from ncls.learning.methods.metal.spatial_schedule import freeze_spatial_split
            split_options = {name: options.get(name) for name in (
                "spatial_core_texels", "spatial_train_tiles", "spatial_validation_tiles", "spatial_split_seed",
                "footprint_recipe", "paired_uv")}
            split_identity = sha256_json({"cohort": self._spatial_cohort_keys[candidates[0]], "options": split_options})
            if split_identity not in self._spatial_splits:
                maximum = 0. if options.get("footprint_recipe") == "point@1" else 4.
                self._spatial_splits[split_identity] = freeze_spatial_split(slots, groups, options, maximum, bool(options.get("paired_uv", False)))
            schedule = self._spatial_splits[split_identity][int(bool(options.get("validation", False)))]
            index = int(options.get("logical_request_index", 0)) % len(schedule)
            cohort, frozen_plan = schedule[index]
        elif options.get("spatial_split_recipe") is not None:
            raise ValueError("unsupported spatial split recipe")
        count = int(source_index.shape[0])
        x0, y0, x1, y1 = cohort.bounds
        uv = torch.rand((count, 2), generator=generator, device=self.device)
        uv = uv * uv.new_tensor((x1 - x0, y1 - y0)) + uv.new_tensor((x0, y0))
        recipe = options.get("footprint_recipe", "balanced-zero-one-four-texel@1")
        if recipe == "balanced-zero-one-four-texel@1":
            footprints = uv.new_tensor((0., 1., 4.)).repeat((count + 2) // 3)[:count]
            footprints = footprints[torch.randperm(count, generator=generator, device=self.device)]
            maximum_footprint = 4.0
        elif recipe == "point@1":
            footprints, maximum_footprint = uv.new_zeros(count), 0.0
        else:
            raise ValueError("Metal spatial footprint recipe is unsupported")
        footprint = footprints * cohort.footprint_step
        dx = torch.stack((footprint, torch.zeros_like(footprint)), dim=1)
        dy = torch.stack((torch.zeros_like(footprint), footprint), dim=1)
        paired = bool(options.get("paired_uv", False))
        pair_x, pair_y = cohort.pair_step if paired else (0., 0.)
        if paired and options.get("paired_uv_recipe") != "one-native-texel-axis-balanced@1":
            raise ValueError("Metal paired UV recipe is unsupported")
        # 主/pair 的所有 bilinear 邻点和 learned RF 由同一 bundle 覆盖。
        # bounds 不做 frac；seam 交由每个原生 UV 组自己的 address mode 处理。
        bounds = (x0, y0, x1 + pair_x, y1 + pair_y)
        maximum_dx = (maximum_footprint * cohort.footprint_step, 0.)
        maximum_dy = (0., maximum_footprint * cohort.footprint_step)
        key = sha256_json({"cohort": self._spatial_cohort_keys[candidates[0]],
                           "bounds": bounds, "dx": maximum_dx, "dy": maximum_dy})
        plan = frozen_plan or self._spatial_tile_schedules.get(key)
        if plan is None:
            plan = build_spatial_bundle(slots, groups, bounds, maximum_dx, maximum_dy)
            # 这里只缓存不含 tensor 的 RF plan，容量固定，绝不缓存 learned feature。
            if len(self._spatial_tile_schedules) >= 32:
                self._spatial_tile_schedules.pop(next(iter(self._spatial_tile_schedules)))
            self._spatial_tile_schedules[key] = plan
        compiler_names = {
            "metal_graph_index": "graph", "metal_schema_index": "schema",
            "metal_recipe_index": "recipe", "metal_identity_index": "metal",
            "metal_finish_index": "finish", "metal_asset_index": "asset",
            "metal_typed_semantic_id": "semantic", "metal_typed_type_id": "type",
            "metal_typed_responsibility_id": "responsibility", "metal_typed_discrete": "discrete",
            "metal_typed_continuous": "continuous", "metal_typed_presence": "presence",
            "metal_canonical_optical": "optical", "metal_access_state": "access",
            "metal_frame_state": "frame", "metal_distribution_id": "distribution",
        }
        values = {name: self._tables[table].index_select(0, source_index)
                  for name, table in compiler_names.items()}
        values.update({"uv": uv, "uv_dx": dx, "uv_dy": dy,
                       "filter_random": torch.rand(count, generator=generator, device=self.device)})
        if paired:
            # effective native texel 步长按真实仿射/Jacobian 得出，原始 query 空间保持不变。
            axis = torch.arange(count, device=self.device) % 2
            axis = axis[torch.randperm(count, generator=generator, device=self.device)]
            offset = torch.stack((torch.where(axis == 0, pair_x, 0.),
                                  torch.where(axis == 1, pair_y, 0.)), dim=1)
            values.update({"paired_uv": uv + offset, "paired_uv_dx": dx, "paired_uv_dy": dy})
        resource = self._assets.acquire_spatial_bundle(asset_index, plan)
        resources = ConditioningResources((resource,))
        try:
            return AdaptedConditioning(values, {
                "metal_registry_identity": self.registry.identity,
                "native_asset_collection_identity": self._assets.collection_id,
                "spatial_bundle_identity": key,
                "spatial_split_identity": split_identity,
                "spatial_query_bounds": list(cohort.bounds),
                "spatial_pair_step": list(cohort.pair_step),
                "native_uv_groups": len(groups),
                "native_asset_reads": sum(2 * group.mapping.lookup_count for group in groups),
                "footprint_recipe": recipe,
            }, resources, {"metal_spatial": torch.zeros(count, dtype=torch.int64, device=self.device)})
        except BaseException:
            resources.release()
            raise


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
