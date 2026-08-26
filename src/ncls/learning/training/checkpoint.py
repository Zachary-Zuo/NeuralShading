from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ncls.core.identity import require_sha256, sha256_file, sha256_json
from ncls.learning.method import MethodDescriptor


@dataclass(frozen=True)
class TrainingCheckpoint:
    method_key: str
    method_descriptor_sha256: str
    implementation_identity: str
    training_config: Mapping[str, Any]
    training_config_sha256: str
    data_source_identity: str
    source_contracts: tuple[Mapping[str, Any], ...]
    source_state_ids: tuple[str, ...]
    step: int
    phase: str
    selection_evidence: Mapping[str, Any]
    model_state: Mapping[str, torch.Tensor]
    optimizer_state: Mapping[str, Any] = field(default_factory=dict)
    scheduler_state: Mapping[str, Any] = field(default_factory=dict)
    scaler_state: Mapping[str, Any] = field(default_factory=dict)
    rng_state: Mapping[str, Any] = field(default_factory=dict)
    format_name: str = "ncls.training-checkpoint"
    format_version: int = 2

    def __post_init__(self) -> None:
        if self.format_name != "ncls.training-checkpoint" or self.format_version != 2:
            raise ValueError("unsupported TrainingCheckpoint format")
        if not self.method_key or not self.implementation_identity or not self.data_source_identity:
            raise ValueError("TrainingCheckpoint identities must be nonempty")
        require_sha256("method_descriptor_sha256", self.method_descriptor_sha256)
        require_sha256("training_config_sha256", self.training_config_sha256)
        if sha256_json(self.training_config) != self.training_config_sha256:
            raise ValueError("TrainingCheckpoint training config hash mismatch")
        if self.step < 0 or self.phase not in {"evaluator", "joint", "sampler", "complete"}:
            raise ValueError("TrainingCheckpoint step or phase is invalid")
        if not self.source_contracts or not self.source_state_ids:
            raise ValueError("TrainingCheckpoint source contracts and state IDs are required")
        for state_id in self.source_state_ids:
            require_sha256("source_state_id", state_id)
        state = dict(self.model_state)
        if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("TrainingCheckpoint model_state must be a nonempty tensor mapping")
        if any(not bool(torch.isfinite(value).all()) for value in state.values() if value.is_floating_point()):
            raise ValueError("TrainingCheckpoint model tensors must be finite")
        object.__setattr__(self, "training_config", dict(self.training_config))
        object.__setattr__(self, "source_contracts", tuple(dict(value) for value in self.source_contracts))
        object.__setattr__(self, "source_state_ids", tuple(self.source_state_ids))
        object.__setattr__(self, "selection_evidence", dict(self.selection_evidence))
        object.__setattr__(self, "model_state", state)
        object.__setattr__(self, "optimizer_state", dict(self.optimizer_state))
        object.__setattr__(self, "scheduler_state", dict(self.scheduler_state))
        object.__setattr__(self, "scaler_state", dict(self.scaler_state))
        object.__setattr__(self, "rng_state", dict(self.rng_state))

    def validate_method(self, descriptor: MethodDescriptor) -> None:
        if descriptor.method_key != self.method_key or descriptor.descriptor_sha256 != self.method_descriptor_sha256:
            raise ValueError("TrainingCheckpoint method descriptor identity mismatch")
        fields = {field.name: field for field in descriptor.tensor_state_schema}
        if set(self.model_state) != set(fields):
            raise ValueError("TrainingCheckpoint tensor keys disagree with method schema")
        symbols: dict[str, int] = {}
        for name, tensor in self.model_state.items():
            field = fields[name]
            if field.dtype != str(tensor.dtype).removeprefix("torch."):
                raise ValueError(f"TrainingCheckpoint tensor {name!r} dtype mismatch")
            if len(field.shape) != tensor.ndim:
                raise ValueError(f"TrainingCheckpoint tensor {name!r} rank mismatch")
            for expected, actual in zip(field.shape, tensor.shape, strict=True):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(f"TrainingCheckpoint tensor {name!r} shape mismatch")
                if isinstance(expected, str):
                    previous = symbols.setdefault(expected, int(actual))
                    if previous != actual:
                        raise ValueError(f"TrainingCheckpoint symbolic dimension {expected!r} mismatch")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name, "format_version": self.format_version,
            "method_key": self.method_key, "method_descriptor_sha256": self.method_descriptor_sha256,
            "implementation_identity": self.implementation_identity,
            "training_config": dict(self.training_config), "training_config_sha256": self.training_config_sha256,
            "data_source_identity": self.data_source_identity,
            "source_contracts": [dict(value) for value in self.source_contracts],
            "source_state_ids": list(self.source_state_ids), "step": self.step, "phase": self.phase,
            "selection_evidence": dict(self.selection_evidence), "model_state": dict(self.model_state),
            "optimizer_state": dict(self.optimizer_state), "scheduler_state": dict(self.scheduler_state),
            "scaler_state": dict(self.scaler_state), "rng_state": dict(self.rng_state),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "TrainingCheckpoint":
        required = {
            "format_name", "format_version", "method_key", "method_descriptor_sha256",
            "implementation_identity", "training_config", "training_config_sha256",
            "data_source_identity", "source_contracts", "source_state_ids", "step", "phase",
            "selection_evidence", "model_state", "optimizer_state", "scheduler_state",
            "scaler_state", "rng_state",
        }
        if set(value) != required:
            raise ValueError(f"TrainingCheckpoint fields must be exactly {sorted(required)}")
        return cls(
            str(value["method_key"]), str(value["method_descriptor_sha256"]),
            str(value["implementation_identity"]), value["training_config"],
            str(value["training_config_sha256"]), str(value["data_source_identity"]),
            tuple(value["source_contracts"]), tuple(str(item) for item in value["source_state_ids"]),
            int(value["step"]), str(value["phase"]), value["selection_evidence"], value["model_state"],
            value["optimizer_state"], value["scheduler_state"], value["scaler_state"], value["rng_state"],
            str(value["format_name"]), int(value["format_version"]),
        )


def save_checkpoint(path: Path | str, checkpoint: TrainingCheckpoint) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint.to_payload(), temporary)
    os.replace(temporary, target)
    digest = sha256_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(digest + "\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def load_checkpoint(
    path: Path | str,
    *,
    descriptor: MethodDescriptor | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("TrainingCheckpoint SHA-256 sidecar is missing")
    expected = sidecar.read_text(encoding="ascii").strip()
    require_sha256("TrainingCheckpoint sidecar", expected)
    if sha256_file(target) != expected:
        raise ValueError("TrainingCheckpoint file hash mismatch")
    value = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("TrainingCheckpoint payload root must be an object")
    checkpoint = TrainingCheckpoint.from_payload(value)
    if descriptor is not None:
        checkpoint.validate_method(descriptor)
    return checkpoint
