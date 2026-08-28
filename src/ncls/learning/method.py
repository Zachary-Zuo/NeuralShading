from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn

from ncls.core.identity import require_sha256, sha256_json
from ncls.core.scattering import MaterialPayload, RuntimePayload
from ncls.core.source import SourceEditResult, SourceSnapshot
from ncls.learning.source_adaptation import NativeFeaturePyramid
from ncls.learning.batches import OnlineTrainingBatch


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
    training_batch_requirements: Mapping[str, tuple[str, ...]]
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
        requirements = {
            str(kind): tuple(fields)
            for kind, fields in self.training_batch_requirements.items()
        }
        if set(requirements) != {"reference-evaluator", "method-sampler"}:
            raise ValueError("method descriptor requires both typed training routes")
        if any(not fields or len(set(fields)) != len(fields) for fields in requirements.values()):
            raise ValueError("method typed route fields must be nonempty and unique")
        if "target_f" not in requirements["reference-evaluator"]:
            raise ValueError("reference-evaluator requirements must include target_f")
        if "sample_u" not in requirements["method-sampler"]:
            raise ValueError("method-sampler requirements must include sample_u")
        if any(
            legacy in fields
            for fields in requirements.values()
            for legacy in ("target", "query_role", "reference_pdf", "solid_angle_weight")
        ):
            raise ValueError("method descriptor contains a removed training batch field")
        if len({field.name for field in self.tensor_state_schema}) != len(self.tensor_state_schema):
            raise ValueError("method tensor state fields must be unique")
        if self.capabilities <= 0:
            raise ValueError("method descriptor capabilities must be nonzero")
        required_bounds = {"maximum_prepare_steps", "maximum_evaluate_steps", "maximum_state_bytes", "maximum_reads"}
        if set(self.bounded_execution) != required_bounds or any(int(value) < 1 for value in self.bounded_execution.values()):
            raise ValueError(f"bounded_execution fields must be exactly {sorted(required_bounds)}")
        object.__setattr__(self, "supported_sources", tuple(self.supported_sources))
        object.__setattr__(self, "training_batch_requirements", requirements)
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
            "training_batch_requirements": {
                kind: list(fields)
                for kind, fields in self.training_batch_requirements.items()
            },
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
        batches: Mapping[str, OnlineTrainingBatch],
        lifecycle: Mapping[str, Any],
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

    def package_validation(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """返回随包冻结、供部署端执行的方法专属验证合同。"""

        del snapshot, checkpoint
        return {"status": "unverified"}

    def classify_edit(self, snapshot: SourceSnapshot, edit: SourceEditResult) -> AdaptationAction:
        return self.descriptor.adaptation_contract(snapshot).classify(edit)

    def validate_training_config(self, config: Mapping[str, Any]) -> None:
        del config

    def configure_lifecycle(self, model: nn.Module, lifecycle: Mapping[str, Any]) -> None:
        del lifecycle
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def materialize_latent(
        self,
        model: nn.Module,
        native_feature_pyramid: NativeFeaturePyramid,
    ) -> None:
        del model, native_feature_pyramid
        raise RuntimeError("method does not implement a latent materialization transition")
