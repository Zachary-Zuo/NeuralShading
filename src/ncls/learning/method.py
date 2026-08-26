from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn

from ncls.core.identity import require_sha256, sha256_json
from ncls.core.scattering import MaterialPayload, RuntimePayload
from ncls.core.source import SourceEditResult, SourceSnapshot
from ncls.data.training_batch import TrainingBatch


AdaptationAction = Literal["unchanged", "runtime-patch", "recompile", "unsupported"]


@dataclass(frozen=True)
class TensorField:
    name: str
    dtype: str
    shape: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.shape:
            raise ValueError("tensor field name, dtype and shape are required")
        if any(not isinstance(value, (int, str)) or value == 0 for value in self.shape):
            raise ValueError("tensor field shape entries must be nonzero integers or symbols")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class SourceAdaptationContract:
    family_id: str
    source_contract_version: int
    supported_paths: tuple[str, ...]
    default_action: AdaptationAction

    def __post_init__(self) -> None:
        if not self.family_id or self.source_contract_version < 1:
            raise ValueError("source adaptation contract identity is invalid")
        if self.default_action not in {"unchanged", "runtime-patch", "recompile", "unsupported"}:
            raise ValueError("source adaptation default action is unsupported")
        if any(not path.startswith("/") for path in self.supported_paths):
            raise ValueError("source adaptation paths must be absolute")

    def classify(self, edit: SourceEditResult) -> AdaptationAction:
        if self.default_action == "unsupported":
            return "unsupported"
        if not self.supported_paths:
            return self.default_action
        supported = all(
            any(path == root or path.startswith(root.rstrip("/") + "/") for root in self.supported_paths)
            for path in edit.changed_paths
        )
        return self.default_action if supported else "unsupported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "supported_paths": list(self.supported_paths),
            "default_action": self.default_action,
        }


@dataclass(frozen=True)
class MethodDescriptor:
    method_key: str
    version: int
    display_name: str
    implementation_sha256: str
    supported_sources: tuple[SourceAdaptationContract, ...]
    training_batch_requirements: tuple[str, ...]
    tensor_state_schema: tuple[TensorField, ...]
    runtime_abi: str
    capabilities: int
    bounded_execution: Mapping[str, int]
    cost_claims: Mapping[str, int | float | str | bool]

    def __post_init__(self) -> None:
        if not self.method_key or self.version < 1 or not self.display_name or not self.runtime_abi:
            raise ValueError("method descriptor identity fields are invalid")
        require_sha256("method implementation_sha256", self.implementation_sha256)
        if not self.supported_sources:
            raise ValueError("method descriptor requires at least one source contract")
        source_keys = {(item.family_id, item.source_contract_version) for item in self.supported_sources}
        if len(source_keys) != len(self.supported_sources):
            raise ValueError("method source adaptation contracts must be unique")
        if not self.training_batch_requirements:
            raise ValueError("method descriptor requires training batch fields")
        if len({field.name for field in self.tensor_state_schema}) != len(self.tensor_state_schema):
            raise ValueError("method tensor state fields must be unique")
        if self.capabilities <= 0:
            raise ValueError("method descriptor capabilities must be nonzero")
        required_bounds = {"maximum_prepare_steps", "maximum_evaluate_steps", "maximum_state_bytes", "maximum_reads"}
        if set(self.bounded_execution) != required_bounds or any(int(value) < 1 for value in self.bounded_execution.values()):
            raise ValueError(f"bounded_execution fields must be exactly {sorted(required_bounds)}")
        object.__setattr__(self, "supported_sources", tuple(self.supported_sources))
        object.__setattr__(self, "training_batch_requirements", tuple(self.training_batch_requirements))
        object.__setattr__(self, "tensor_state_schema", tuple(self.tensor_state_schema))
        object.__setattr__(self, "bounded_execution", dict(self.bounded_execution))
        object.__setattr__(self, "cost_claims", dict(self.cost_claims))

    @property
    def descriptor_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_key": self.method_key,
            "version": self.version,
            "display_name": self.display_name,
            "implementation_sha256": self.implementation_sha256,
            "supported_sources": [item.to_dict() for item in self.supported_sources],
            "training_batch_requirements": list(self.training_batch_requirements),
            "tensor_state_schema": [item.to_dict() for item in self.tensor_state_schema],
            "runtime_abi": self.runtime_abi,
            "capabilities": self.capabilities,
            "bounded_execution": dict(self.bounded_execution),
            "cost_claims": dict(self.cost_claims),
        }

    def adaptation_contract(self, snapshot: SourceSnapshot) -> SourceAdaptationContract:
        for contract in self.supported_sources:
            if (
                contract.family_id == snapshot.family_id
                and contract.source_contract_version == snapshot.source_contract_version
            ):
                return contract
        raise ValueError(f"method {self.method_key!r} does not support source {snapshot.family_id!r}")


class MethodDefinition(ABC):
    descriptor: MethodDescriptor

    @abstractmethod
    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def training_objective(
        self,
        model: nn.Module,
        batch: TrainingBatch,
        phase: str,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        raise NotImplementedError

    @abstractmethod
    def export_training_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def restore_training_state(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        raise NotImplementedError

    @abstractmethod
    def compile_runtime(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        raise NotImplementedError

    @abstractmethod
    def compile_material(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> MaterialPayload:
        raise NotImplementedError

    def classify_edit(self, snapshot: SourceSnapshot, edit: SourceEditResult) -> AdaptationAction:
        return self.descriptor.adaptation_contract(snapshot).classify(edit)

    def configure_phase(self, model: nn.Module, phase: str) -> None:
        if phase not in {"evaluator", "joint", "sampler"}:
            raise ValueError(f"unsupported training phase {phase!r}")
        for parameter in model.parameters():
            parameter.requires_grad_(True)
