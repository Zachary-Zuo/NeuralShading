from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Protocol, TypeAlias

import torch

from ncls.learning.source_adaptation import NativeAssetDescriptor, NativeAssetTile
from ncls.learning.conditioning_resources import ConditioningResources


TrainingRouteKind = Literal["asset-tile", "reference-evaluator", "method-sampler"]


class BatchLease(Protocol):
    def release(self) -> None: ...


def _validate_tensor_mapping(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    values = dict(tensors)
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("training conditioning values must all be torch tensors")
    if len({value.device for value in values.values()}) != 1:
        raise ValueError("training conditioning tensors must share one device")
    return values


def _validate_finite(values: tuple[torch.Tensor, ...], message: str) -> None:
    checks = tuple(torch.isfinite(value).all() for value in values if value.is_floating_point())
    if not checks:
        return
    valid = torch.stack(checks).all()
    if valid.device.type == "cuda":
        torch._assert_async(valid)
    elif not bool(valid):
        raise ValueError(message)


@dataclass(frozen=True)
class TrainingRouteRequest:
    """一次 typed online query stream 请求。"""

    name: str
    kind: TrainingRouteKind
    batch_size: int
    direction_count: int
    global_step: int
    seed: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.kind not in {"asset-tile", "reference-evaluator", "method-sampler"}:
            raise ValueError("training route identity or kind is invalid")
        if self.batch_size < 1 or self.direction_count < 1:
            raise ValueError("training route sizes must be positive")
        if self.global_step < 0 or self.seed < 0:
            raise ValueError("training route step and seed must be nonnegative")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class TrainingConditioning:
    source_family_id: str
    source_snapshot_ids: tuple[str, ...]
    tensors: Mapping[str, torch.Tensor]
    provenance: Mapping[str, Any]
    resources: ConditioningResources = field(default_factory=ConditioningResources, compare=False)
    bindings: Mapping[str, torch.Tensor] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.source_family_id or not self.source_snapshot_ids:
            raise ValueError("training conditioning source identity is required")
        tensors = _validate_tensor_mapping(self.tensors)
        required = {"source_index", "wo"}
        missing = required - set(tensors)
        if missing:
            raise ValueError(f"training conditioning is missing tensors: {sorted(missing)}")
        source_index = tensors["source_index"]
        wo = tensors["wo"]
        if source_index.ndim != 1 or source_index.dtype != torch.int64:
            raise ValueError("source_index must be int64 [batch]")
        batch_size = int(source_index.shape[0])
        if any(value.ndim == 0 or value.shape[0] != batch_size for value in tensors.values()):
            raise ValueError("conditioning tensors must have one row per query")
        bindings = dict(self.bindings)
        for entry in self.resources.entries:
            if entry.device != wo.device:
                raise ValueError("conditioning resource must share the query device")
        for name, indices in bindings.items():
            if not name or indices.dtype != torch.int64 or indices.shape != (batch_size,):
                raise ValueError("conditioning bindings must be named int64 [batch]")
            if indices.device != wo.device:
                raise ValueError("conditioning binding must share the query device")
            in_bounds = ((indices >= 0) & (indices < len(self.resources))).all()
            if indices.device.type == "cuda":
                torch._assert_async(in_bounds)
            elif not bool(in_bounds):
                raise ValueError("conditioning binding is outside resources")
        source_bounds = torch.all(source_index >= 0) & torch.all(
            source_index < len(self.source_snapshot_ids)
        )
        if source_bounds.device.type == "cuda":
            torch._assert_async(source_bounds)
        elif not bool(source_bounds):
            raise ValueError("source_index is outside source_snapshot_ids")
        if wo.shape != (batch_size, 3) or not wo.is_floating_point():
            raise ValueError("wo must be floating [batch,3]")
        spatial_shapes = {
            "uv": (batch_size, 2),
            "uv_dx": (batch_size, 2),
            "uv_dy": (batch_size, 2),
            "paired_uv": (batch_size, 2),
            "paired_uv_dx": (batch_size, 2),
            "paired_uv_dy": (batch_size, 2),
            "mip_level": (batch_size,),
            "filter_random": (batch_size,),
        }
        for name, shape in spatial_shapes.items():
            if name in tensors and tensors[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        paired_spatial = {"paired_uv", "paired_uv_dx", "paired_uv_dy"} & set(tensors)
        if paired_spatial and paired_spatial != {
            "paired_uv", "paired_uv_dx", "paired_uv_dy"
        }:
            raise ValueError("paired UV conditioning fields must be provided together")
        if "native_features" in tensors:
            native = tensors["native_features"]
            if native.ndim != 2 or native.shape[0] != batch_size:
                raise ValueError("native_features must have shape [batch,feature]")
            if not isinstance(self.provenance.get("native_feature_layout_id"), str):
                raise ValueError("native_features require a layout identity")
        _validate_finite(tuple(tensors.values()), "training conditioning contains non-finite values")
        object.__setattr__(self, "source_snapshot_ids", tuple(self.source_snapshot_ids))
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "bindings", bindings)

    def select_rows(self, indices: torch.Tensor) -> TrainingConditioning:
        resources = self.resources.retain()
        try:
            return replace(
                self,
                tensors={name: value.index_select(0, indices) for name, value in self.tensors.items()},
                bindings={name: value.index_select(0, indices) for name, value in self.bindings.items()},
                resources=resources,
            )
        except BaseException:
            resources.release()
            raise

    def retain(self) -> TrainingConditioning:
        resources = self.resources.retain()
        try:
            return replace(self, resources=resources)
        except BaseException:
            resources.release()
            raise

    @staticmethod
    def concatenate(
        values: tuple[TrainingConditioning, ...],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> TrainingConditioning:
        if not values:
            raise ValueError("conditioning concatenation requires inputs")
        first = values[0]
        for value in values:
            if (
                value.source_family_id != first.source_family_id
                or value.source_snapshot_ids != first.source_snapshot_ids
                or set(value.tensors) != set(first.tensors)
                or set(value.bindings) != set(first.bindings)
                or value.device != first.device
            ):
                raise ValueError("conditioning concatenation contracts disagree")
        resources, remaps = ConditioningResources.concatenate(tuple(value.resources for value in values))
        try:
            bindings = {
                name: torch.cat(
                    [
                        torch.tensor(remap, dtype=torch.int64, device=first.device)[value.bindings[name]]
                        for value, remap in zip(values, remaps, strict=True)
                    ]
                )
                for name in first.bindings
            }
            return replace(
                first,
                tensors={name: torch.cat([value.tensors[name] for value in values]) for name in first.tensors},
                provenance=first.provenance if provenance is None else provenance,
                resources=resources,
                bindings=bindings,
            )
        except BaseException:
            resources.release()
            raise

    def release(self) -> None:
        self.resources.release()

    @property
    def batch_size(self) -> int:
        return int(self.tensors["source_index"].shape[0])

    @property
    def device(self) -> torch.device:
        return self.tensors["source_index"].device


@dataclass(frozen=True)
class EvaluatorBatch:
    conditioning: TrainingConditioning
    wi: torch.Tensor
    target_f: torch.Tensor
    paired_target_f: torch.Tensor | None = None
    lease: BatchLease | None = field(default=None, compare=False, repr=False)
    schema_name: str = "ncls.evaluator-batch"
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.evaluator-batch" or self.schema_version != 3:
            raise ValueError("unsupported EvaluatorBatch schema")
        batch_size = self.conditioning.batch_size
        if self.wi.ndim != 3 or self.wi.shape[0] != batch_size or self.wi.shape[2] != 3:
            raise ValueError("EvaluatorBatch wi must have shape [batch,direction,3]")
        if self.target_f.shape != self.wi.shape:
            raise ValueError("EvaluatorBatch target_f must match wi shape")
        if self.wi.device != self.conditioning.device or self.target_f.device != self.conditioning.device:
            raise ValueError("EvaluatorBatch tensors must share the conditioning device")
        _validate_finite((self.wi, self.target_f), "EvaluatorBatch contains non-finite values")
        nonnegative = torch.all(self.target_f >= 0.0)
        if nonnegative.device.type == "cuda":
            torch._assert_async(nonnegative)
        elif not bool(nonnegative):
            raise ValueError("EvaluatorBatch target_f must be nonnegative")
        has_paired_conditioning = "paired_uv" in self.conditioning.tensors
        if has_paired_conditioning != (self.paired_target_f is not None):
            raise ValueError(
                "EvaluatorBatch paired target and paired UV conditioning must be provided together"
            )
        if self.paired_target_f is not None:
            if self.paired_target_f.shape != self.wi.shape:
                raise ValueError("EvaluatorBatch paired_target_f must match wi shape")
            if self.paired_target_f.device != self.conditioning.device:
                raise ValueError(
                    "EvaluatorBatch paired target must share the conditioning device"
                )
            _validate_finite(
                (self.paired_target_f,),
                "EvaluatorBatch paired target contains non-finite values",
            )
            paired_nonnegative = torch.all(self.paired_target_f >= 0.0)
            if paired_nonnegative.device.type == "cuda":
                torch._assert_async(paired_nonnegative)
            elif not bool(paired_nonnegative):
                raise ValueError("EvaluatorBatch paired_target_f must be nonnegative")

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        values = {
            **self.conditioning.tensors,
            "wi": self.wi,
            "target_f": self.target_f,
        }
        if self.paired_target_f is not None:
            values["paired_target_f"] = self.paired_target_f
        return values

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.conditioning.provenance

    def release(self) -> None:
        try:
            if self.lease is not None:
                self.lease.release()
        finally:
            self.conditioning.release()


@dataclass(frozen=True)
class MethodSamplerBatch:
    conditioning: TrainingConditioning
    sample_u: torch.Tensor
    lease: BatchLease | None = field(default=None, compare=False, repr=False)
    schema_name: str = "ncls.method-sampler-batch"
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.method-sampler-batch" or self.schema_version != 3:
            raise ValueError("unsupported MethodSamplerBatch schema")
        if self.sample_u.shape != (self.conditioning.batch_size, 2):
            raise ValueError("MethodSamplerBatch sample_u must have shape [batch,2]")
        if self.sample_u.device != self.conditioning.device:
            raise ValueError("MethodSamplerBatch tensors must share the conditioning device")
        _validate_finite((self.sample_u,), "MethodSamplerBatch contains non-finite values")

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {**self.conditioning.tensors, "sample_u": self.sample_u}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.conditioning.provenance

    def release(self) -> None:
        try:
            if self.lease is not None:
                self.lease.release()
        finally:
            self.conditioning.release()


@dataclass(frozen=True)
class AssetTileBatch:
    asset_descriptors: tuple[NativeAssetDescriptor, ...]
    tiles: tuple[NativeAssetTile, ...]
    provenance: Mapping[str, Any]
    lease: BatchLease | None = field(default=None, compare=False, repr=False)
    schema_name: str = "ncls.asset-tile-batch"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.asset-tile-batch" or self.schema_version != 1:
            raise ValueError("unsupported AssetTileBatch schema")
        descriptors = tuple(self.asset_descriptors)
        tiles = tuple(self.tiles)
        asset_ids = tuple(descriptor.asset_id for descriptor in descriptors)
        if not descriptors or len(set(asset_ids)) != len(asset_ids) or not tiles:
            raise ValueError("AssetTileBatch requires unique assets and nonempty tiles")
        if any(
            tile.asset_index >= len(descriptors)
            or tile.request.asset_id != descriptors[tile.asset_index].asset_id
            or tile.request.schema_id != descriptors[tile.asset_index].schema_id
            for tile in tiles
        ):
            raise ValueError("AssetTileBatch tile asset_index is outside asset_ids")
        devices = {tile.values.device for tile in tiles}
        if len(devices) != 1:
            raise ValueError("AssetTileBatch tiles must share one device")
        _validate_finite(
            tuple(tile.values for tile in tiles),
            "AssetTileBatch contains non-finite values",
        )
        object.__setattr__(self, "asset_descriptors", descriptors)
        object.__setattr__(self, "tiles", tiles)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def device(self) -> torch.device:
        return self.tiles[0].values.device

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            (
                f"asset_{tile.asset_index}_domain_{tile.request.domain_id}_"
                f"mip_{tile.mip_level}_tile_{index}_role_{role.role_id}"
            ): tile.role_values(role.role_id)
            for index, tile in enumerate(self.tiles)
            for role in tile.roles
        }

    def release(self) -> None:
        for tile in reversed(self.tiles):
            tile.release()
        if self.lease is not None:
            self.lease.release()


OnlineTrainingBatch: TypeAlias = AssetTileBatch | EvaluatorBatch | MethodSamplerBatch


__all__ = [
    "BatchLease",
    "AssetTileBatch",
    "EvaluatorBatch",
    "MethodSamplerBatch",
    "OnlineTrainingBatch",
    "TrainingConditioning",
    "TrainingRouteKind",
    "TrainingRouteRequest",
]
