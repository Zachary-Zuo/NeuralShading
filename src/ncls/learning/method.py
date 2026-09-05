from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch
from torch import nn

from ncls.core.identity import require_sha256, sha256_json
from ncls.core.scattering import InstancePayload, MaterialPayload, RuntimePayload
from ncls.core.source import SourceEditResult, SourceSnapshot
from ncls.learning.source_adaptation import NativeAssetCollection
from ncls.learning.batches import OnlineTrainingBatch


AdaptationAction = Literal["unchanged", "runtime-patch", "recompile", "unsupported"]


@dataclass(frozen=True)
class TrainingInitializationRequest:
    """一次发生在任何模型前向之前的 train-only online 初始化请求。

    ``sample_count``是整个distributed job的全局样本数；engine以确定性连续
    分片分给各rank，再按rank顺序合并后交给method lifecycle。
    """

    name: str
    phase_name: str
    route_name: str
    sample_count: int
    seed: int
    tensor_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        fields = tuple(str(value) for value in self.tensor_fields)
        if (
            not self.name
            or not self.phase_name
            or not self.route_name
            or self.sample_count < 1
            or self.seed < 0
            or not fields
            or len(set(fields)) != len(fields)
            or any(not value for value in fields)
        ):
            raise ValueError("training initialization request is invalid")
        object.__setattr__(self, "tensor_fields", fields)


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
class ComponentContract:
    component_id: str
    required: bool
    parameter_groups: tuple[str, ...]
    active_phases: tuple[str, ...]
    batch_dependencies: tuple[str, ...]
    python_outputs: tuple[str, ...]
    runtime_artifacts: tuple[str, ...]
    slang_entry_points: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("method component identity is required")
        for name, values in (
            ("parameter_groups", self.parameter_groups),
            ("active_phases", self.active_phases),
            ("batch_dependencies", self.batch_dependencies),
            ("python_outputs", self.python_outputs),
            ("runtime_artifacts", self.runtime_artifacts),
            ("slang_entry_points", self.slang_entry_points),
        ):
            normalized = tuple(str(value) for value in values)
            if len(set(normalized)) != len(normalized) or any(not value for value in normalized):
                raise ValueError(f"method component {name} must be unique and nonempty")
            object.__setattr__(self, name, normalized)
        if self.required and not (
            self.parameter_groups or self.python_outputs or self.runtime_artifacts or self.slang_entry_points
        ):
            raise ValueError("required method component has no observable implementation contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "required": self.required,
            "parameter_groups": list(self.parameter_groups),
            "active_phases": list(self.active_phases),
            "batch_dependencies": list(self.batch_dependencies),
            "python_outputs": list(self.python_outputs),
            "runtime_artifacts": list(self.runtime_artifacts),
            "slang_entry_points": list(self.slang_entry_points),
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
    parameter_groups: Mapping[str, tuple[str, ...]]
    components: tuple[ComponentContract, ...]
    training_resource_requirements: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema_name: str = "ncls.method-descriptor"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.method-descriptor" or self.schema_version != 2:
            raise ValueError("unsupported MethodDescriptor schema")
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
        allowed_batches = {"asset-tile", "reference-evaluator", "method-sampler"}
        if not requirements or not set(requirements).issubset(allowed_batches):
            raise ValueError("method descriptor contains unsupported typed training routes")
        if any(not fields or len(set(fields)) != len(fields) for fields in requirements.values()):
            raise ValueError("method typed route fields must be nonempty and unique")
        if "reference-evaluator" in requirements and "target_f" not in requirements["reference-evaluator"]:
            raise ValueError("reference-evaluator requirements must include target_f")
        if "method-sampler" in requirements and "sample_u" not in requirements["method-sampler"]:
            raise ValueError("method-sampler requirements must include sample_u")
        resources = {str(kind): tuple(names) for kind, names in self.training_resource_requirements.items()}
        if not set(resources).issubset(requirements) or any(
            not names or len(set(names)) != len(names) or any(not name for name in names)
            for names in resources.values()
        ):
            raise ValueError("method resource bindings must be unique and reference a declared route")
        if len({field.name for field in self.tensor_state_schema}) != len(self.tensor_state_schema):
            raise ValueError("method tensor state fields must be unique")
        if self.capabilities <= 0:
            raise ValueError("method descriptor capabilities must be nonzero")
        required_bounds = {"maximum_prepare_steps", "maximum_evaluate_steps", "maximum_state_bytes", "maximum_reads"}
        if set(self.bounded_execution) != required_bounds or any(int(value) < 1 for value in self.bounded_execution.values()):
            raise ValueError(f"bounded_execution fields must be exactly {sorted(required_bounds)}")
        groups = {str(name): tuple(str(value) for value in values) for name, values in self.parameter_groups.items()}
        if not groups or any(not name or not values or len(set(values)) != len(values) for name, values in groups.items()):
            raise ValueError("method parameter groups must be nonempty and contain unique parameter names")
        flattened = [name for names in groups.values() for name in names]
        if len(set(flattened)) != len(flattened):
            raise ValueError("method trainable parameters must belong to exactly one parameter group")
        components = tuple(self.components)
        if not components or len({component.component_id for component in components}) != len(components):
            raise ValueError("method descriptor requires unique component contracts")
        for component in components:
            if not set(component.parameter_groups).issubset(groups):
                raise ValueError("method component references an unknown parameter group")
            if not set(component.batch_dependencies).issubset(requirements):
                raise ValueError("method component references an unknown typed batch")
        required_groups = {
            group
            for component in components
            if component.required
            for group in component.parameter_groups
        }
        if required_groups != set(groups):
            raise ValueError("every method parameter group must belong to a required component")
        object.__setattr__(self, "supported_sources", tuple(self.supported_sources))
        object.__setattr__(self, "training_batch_requirements", requirements)
        object.__setattr__(self, "training_resource_requirements", resources)
        object.__setattr__(self, "tensor_state_schema", tuple(self.tensor_state_schema))
        object.__setattr__(self, "bounded_execution", dict(self.bounded_execution))
        object.__setattr__(self, "cost_claims", dict(self.cost_claims))
        object.__setattr__(self, "parameter_groups", groups)
        object.__setattr__(self, "components", components)

    @property
    def descriptor_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "method_key": self.method_key,
            "version": self.version,
            "display_name": self.display_name,
            "implementation_sha256": self.implementation_sha256,
            "supported_sources": [item.to_dict() for item in self.supported_sources],
            "training_batch_requirements": {
                kind: list(fields)
                for kind, fields in self.training_batch_requirements.items()
            },
            "training_resource_requirements": {kind: list(names) for kind, names in self.training_resource_requirements.items()},
            "tensor_state_schema": [item.to_dict() for item in self.tensor_state_schema],
            "runtime_abi": self.runtime_abi,
            "capabilities": self.capabilities,
            "bounded_execution": dict(self.bounded_execution),
            "cost_claims": dict(self.cost_claims),
            "parameter_groups": {name: list(values) for name, values in self.parameter_groups.items()},
            "components": [component.to_dict() for component in self.components],
        }
        return result

    def adaptation_contract(self, snapshot: SourceSnapshot) -> SourceAdaptationContract:
        for contract in self.supported_sources:
            if (
                contract.family_id == snapshot.family_id
                and contract.source_contract_version == snapshot.source_contract_version
            ):
                return contract
        raise ValueError(f"method {self.method_key!r} does not support source {snapshot.family_id!r}")


class Method(ABC):
    key: str
    descriptor: MethodDescriptor

    def requirements(self, config: Mapping[str, Any] | None = None):
        from ncls.data import DataRequirement

        selected = set(self.descriptor.training_batch_requirements)
        if config is not None:
            self.validate_training_config(config)
            selected = {str(route["kind"]) for phase in config["phases"] for route in phase["routes"]}
            if not selected.issubset(self.descriptor.training_batch_requirements):
                raise ValueError("training config contains a route not supported by the method")
        return tuple(DataRequirement(kind, tuple(fields)) for kind, fields in self.descriptor.training_batch_requirements.items()
                     if kind in selected)

    @abstractmethod
    def create_source_adapter(self, snapshots, device):
        raise NotImplementedError


    @abstractmethod
    def create_trainable(self, context: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def training_objective(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        raise NotImplementedError

    @abstractmethod
    def export_training_state(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def restore_training_state(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def prepare_export(self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        """需要时将当前训练表示编译为部署状态；不修改训练模型或 checkpoint。"""
        return checkpoint

    @abstractmethod
    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        raise NotImplementedError

    @abstractmethod
    def compile_asset(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> MaterialPayload:
        raise NotImplementedError

    def compile_instance(
        self,
        snapshot: SourceSnapshot,
        checkpoint: Mapping[str, Any],
    ) -> InstancePayload:
        del snapshot, checkpoint
        return InstancePayload({"compiled_material_index": 0})

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

    def initialization_requests(
        self, config: Mapping[str, Any]
    ) -> tuple[TrainingInitializationRequest, ...]:
        """声明 fresh run 的 train-only 在线初始化；默认方法不需要该阶段。"""

        del config
        return ()

    def initialize_training_state(
        self,
        model: nn.Module,
        values: Mapping[str, Mapping[str, torch.Tensor]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """消费已搬到 CPU 的初始化张量并写入 checkpoint-visible model state。"""

        del model, metadata
        if values:
            raise RuntimeError("method declared initialization values but did not consume them")
        return {}

    def parameter_registry(self, model: nn.Module) -> Mapping[str, tuple[nn.Parameter, ...]]:
        named = dict(model.named_parameters())
        declared = {
            name
            for values in self.descriptor.parameter_groups.values()
            for name in values
        }
        if set(named) != declared:
            missing = sorted(declared - set(named))
            orphan = sorted(set(named) - declared)
            raise ValueError(
                f"method parameter registry mismatch; missing={missing}, orphan={orphan}"
            )
        return {
            group: tuple(named[name] for name in names)
            for group, names in self.descriptor.parameter_groups.items()
        }

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None:
        active = set(str(value) for value in phase.get("parameter_groups", ()))
        registry = self.parameter_registry(model)
        if not active or not active.issubset(registry):
            raise ValueError("training phase references invalid parameter groups")
        for group, parameters in registry.items():
            for parameter in parameters:
                parameter.requires_grad_(group in active)

    def materialize_assets(
        self,
        model: nn.Module,
        native_assets: NativeAssetCollection,
    ) -> None:
        del model, native_assets
        raise RuntimeError("method does not implement an asset materialization transition")

    def apply_phase_transition(
        self,
        model: nn.Module,
        transition: str,
        native_assets: NativeAssetCollection,
    ) -> None:
        del model, native_assets
        raise RuntimeError(
            f"method {self.descriptor.method_key!r} does not implement transition {transition!r}"
        )
