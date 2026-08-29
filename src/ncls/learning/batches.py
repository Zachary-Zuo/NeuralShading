from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, TypeAlias

import torch

from ncls.learning.source_adaptation import NativeAssetDescriptor, NativeAssetTile


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
            "mip_level": (batch_size,),
        }
        for name, shape in spatial_shapes.items():
            if name in tensors and tensors[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
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

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {**self.conditioning.tensors, "wi": self.wi, "target_f": self.target_f}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.conditioning.provenance

    def release(self) -> None:
        if self.lease is not None:
            self.lease.release()


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
        if self.lease is not None:
            self.lease.release()


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
