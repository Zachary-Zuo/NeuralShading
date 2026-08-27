from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch


REQUIRED_TRAINING_TENSORS = (
    "source_index",
    "wo",
    "wi",
    "target",
    "solid_angle_weight",
    "reference_pdf",
    "sample_count",
    "rng_seed",
    "query_role",
)


@dataclass(frozen=True)
class TrainingRouteRequest:
    """一次命名训练 query stream 请求；runner 与 producer 都不解释方法名称。"""

    name: str
    batch_size: int
    direction_count: int
    global_step: int
    query_role: int
    seed: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.batch_size < 1 or self.direction_count < 1:
            raise ValueError("training route identity and sizes must be positive")
        if self.global_step < 0 or self.query_role < 0 or self.seed < 0:
            raise ValueError("training route step, role and seed must be nonnegative")
        object.__setattr__(self, "options", dict(self.options))


class BatchLease(Protocol):
    def release(self) -> None: ...


@dataclass(frozen=True)
class TrainingBatch:
    source_family_id: str
    source_state_ids: tuple[str, ...]
    response_measure: str
    tensors: Mapping[str, torch.Tensor]
    provenance: Mapping[str, Any]
    lease: BatchLease | None = field(default=None, compare=False, repr=False)
    schema_name: str = "ncls.training-batch"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.training-batch" or self.schema_version != 1:
            raise ValueError("unsupported TrainingBatch schema")
        if not self.source_family_id or not self.source_state_ids or not self.response_measure:
            raise ValueError("TrainingBatch source identity and response measure are required")
        tensors = dict(self.tensors)
        missing = set(REQUIRED_TRAINING_TENSORS) - set(tensors)
        if missing:
            raise ValueError(f"TrainingBatch is missing tensors: {sorted(missing)}")
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise ValueError("TrainingBatch values must all be torch tensors")
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("TrainingBatch tensors must share one device")
        source_index = tensors["source_index"]
        wo = tensors["wo"]
        wi = tensors["wi"]
        target = tensors["target"]
        batch_size = int(source_index.shape[0])
        if source_index.ndim != 1 or source_index.dtype != torch.int64:
            raise ValueError("TrainingBatch source_index must be int64 [batch]")
        if len(self.source_state_ids) != batch_size:
            raise ValueError("TrainingBatch source_state_ids must match batch size")
        if wo.shape != (batch_size, 3):
            raise ValueError("TrainingBatch wo must have shape [batch, 3]")
        if wi.ndim != 3 or wi.shape[0] != batch_size or wi.shape[2] != 3:
            raise ValueError("TrainingBatch wi must have shape [batch, direction, 3]")
        if target.shape != wi.shape:
            raise ValueError("TrainingBatch target must match wi shape")
        scalar_shape = wi.shape[:-1]
        for name in ("solid_angle_weight", "reference_pdf", "sample_count", "rng_seed"):
            if tensors[name].shape != scalar_shape:
                raise ValueError(f"TrainingBatch {name} must have shape {tuple(scalar_shape)}")
        if tensors["query_role"].shape != (batch_size,):
            raise ValueError("TrainingBatch query_role must have shape [batch]")
        spatial_shapes = {
            "uv": (batch_size, 2),
            "uv_dx": (batch_size, 2),
            "uv_dy": (batch_size, 2),
            "mip_level": (batch_size,),
        }
        for name, shape in spatial_shapes.items():
            if name in tensors and tensors[name].shape != shape:
                raise ValueError(f"TrainingBatch {name} must have shape {shape}")
        if "native_features" in tensors:
            native_features = tensors["native_features"]
            if native_features.ndim != 2 or native_features.shape[0] != batch_size:
                raise ValueError("TrainingBatch native_features must have shape [batch, feature]")
            if not isinstance(self.provenance.get("native_feature_layout_id"), str):
                raise ValueError("TrainingBatch native features require a layout identity in provenance")
        if "sample_u" in tensors and tensors["sample_u"].shape != (batch_size, 2):
            raise ValueError("TrainingBatch sample_u must have shape [batch, 2]")
        floating = ("wo", "wi", "target", "solid_angle_weight", "reference_pdf")
        optional_floating = tuple(
            name for name in ("uv", "uv_dx", "uv_dy", "mip_level", "native_features", "sample_u")
            if name in tensors
        )
        value_checks = torch.stack(
            tuple(
                torch.isfinite(tensors[name]).all()
                for name in floating + optional_floating
            )
            + (
                torch.all(tensors["reference_pdf"] >= 0),
                torch.all(tensors["solid_angle_weight"] >= 0),
                torch.all(tensors["sample_count"] >= 1),
            )
        ).all()
        if value_checks.device.type == "cuda":
            # live producer的数值不回读host；shape/dtype/device仍在CPU同步拒绝。
            torch._assert_async(value_checks)
        elif not bool(value_checks):
            raise ValueError(
                "TrainingBatch floating values, weights, PDF or sample counts are invalid"
            )
        object.__setattr__(self, "source_state_ids", tuple(self.source_state_ids))
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def device(self) -> torch.device:
        return self.tensors["target"].device

    @property
    def batch_size(self) -> int:
        return int(self.tensors["source_index"].shape[0])

    def to(self, device: torch.device | str) -> "TrainingBatch":
        target = torch.device(device)
        if target == self.device:
            return self
        if self.lease is not None:
            raise ValueError("a live TrainingBatch lease cannot be copied to another device")
        return TrainingBatch(
            self.source_family_id,
            self.source_state_ids,
            self.response_measure,
            {name: value.to(target) for name, value in self.tensors.items()},
            self.provenance,
        )

    def release(self) -> None:
        if self.lease is not None:
            self.lease.release()
