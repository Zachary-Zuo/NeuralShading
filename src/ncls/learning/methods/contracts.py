from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import torch
from torch import nn

from ncls.core.identity import require_sha256, sha256_json
from ncls.core.scattering import InstancePayload, MaterialPayload, RuntimePayload
from ncls.core.source import SourceSnapshot
from ncls.data import DataRequirement, TrainingRouteKind
from ncls.learning.batches import OnlineTrainingBatch
from ncls.learning.method import (
    MethodDefinition,
    MethodDescriptor,
    TrainingInitializationRequest,
)
from ncls.learning.source_adaptation import NativeAssetCollection


_PUBLIC_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ModelFactory(Protocol):
    implementation_sha256: str

    def create(self, context: Mapping[str, Any]) -> nn.Module: ...


class MethodDataFacet(Protocol):
    implementation_sha256: str

    def requirements(self) -> tuple[DataRequirement, ...]: ...

    def create_source_adapter(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> Any: ...


class ObjectiveFacet(Protocol):
    implementation_sha256: str

    def compute(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]: ...


class LifecycleFacet(Protocol):
    implementation_sha256: str

    def validate_training_plan(self, config: Mapping[str, Any]) -> None: ...

    def initialization_requests(
        self, config: Mapping[str, Any]
    ) -> tuple[TrainingInitializationRequest, ...]: ...

    def initialize_training_state(
        self,
        model: nn.Module,
        values: Mapping[str, Mapping[str, torch.Tensor]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None: ...

    def parameter_registry(
        self, model: nn.Module
    ) -> Mapping[str, tuple[nn.Parameter, ...]]: ...

    def apply_transition(
        self, model: nn.Module, transition: str, assets: NativeAssetCollection
    ) -> None: ...


class CheckpointCodec(Protocol):
    implementation_sha256: str

    def encode(self, model: nn.Module) -> Mapping[str, torch.Tensor]: ...

    def restore(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None: ...


class DeploymentCompiler(Protocol):
    implementation_sha256: str

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload: ...

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload: ...

    def compile_instance(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> InstancePayload: ...

    def package_validation(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def _facet_identity(definition: MethodDefinition, facet: str) -> str:
    return sha256_json(
        {
            "method_implementation_sha256": definition.descriptor.implementation_sha256,
            "facet": facet,
            "adapter_version": 1,
        }
    )


@dataclass(frozen=True)
class _DefinitionModelFactory:
    definition: MethodDefinition
    implementation_sha256: str

    def create(self, context: Mapping[str, Any]) -> nn.Module:
        return self.definition.create_trainable(context)


@dataclass(frozen=True)
class _DefinitionDataFacet:
    definition: MethodDefinition
    source_adapter_factory: Callable[[Sequence[SourceSnapshot], torch.device], Any]
    implementation_sha256: str

    def requirements(self) -> tuple[DataRequirement, ...]:
        return tuple(
            DataRequirement(cast(TrainingRouteKind, kind), tuple(fields))
            for kind, fields in self.definition.descriptor.training_batch_requirements.items()
        )

    def create_source_adapter(
        self, snapshots: Sequence[SourceSnapshot], device: torch.device
    ) -> Any:
        return self.source_adapter_factory(snapshots, device)


@dataclass(frozen=True)
class _DefinitionObjectiveFacet:
    definition: MethodDefinition
    implementation_sha256: str

    def compute(
        self,
        model: nn.Module,
        batches: Mapping[str, OnlineTrainingBatch],
        phase: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | float]]:
        return self.definition.training_objective(model, batches, phase)


@dataclass(frozen=True)
class _DefinitionLifecycleFacet:
    definition: MethodDefinition
    implementation_sha256: str

    def validate_training_plan(self, config: Mapping[str, Any]) -> None:
        self.definition.validate_training_config(config)

    def initialization_requests(
        self, config: Mapping[str, Any]
    ) -> tuple[TrainingInitializationRequest, ...]:
        return self.definition.initialization_requests(config)

    def initialize_training_state(
        self,
        model: nn.Module,
        values: Mapping[str, Mapping[str, torch.Tensor]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self.definition.initialize_training_state(model, values, metadata)

    def configure_phase(self, model: nn.Module, phase: Mapping[str, Any]) -> None:
        self.definition.configure_phase(model, phase)

    def parameter_registry(
        self, model: nn.Module
    ) -> Mapping[str, tuple[nn.Parameter, ...]]:
        return self.definition.parameter_registry(model)

    def apply_transition(
        self, model: nn.Module, transition: str, assets: NativeAssetCollection
    ) -> None:
        self.definition.apply_phase_transition(model, transition, assets)


@dataclass(frozen=True)
class _DefinitionCheckpointCodec:
    definition: MethodDefinition
    implementation_sha256: str

    def encode(self, model: nn.Module) -> Mapping[str, torch.Tensor]:
        return self.definition.export_training_state(model)

    def restore(self, model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        self.definition.restore_training_state(model, state)


@dataclass(frozen=True)
class _DefinitionDeploymentCompiler:
    definition: MethodDefinition
    implementation_sha256: str

    def compile_program(self, checkpoint: Mapping[str, Any]) -> RuntimePayload:
        return self.definition.compile_program(checkpoint)

    def compile_asset(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> MaterialPayload:
        return self.definition.compile_asset(snapshot, checkpoint)

    def compile_instance(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> InstancePayload:
        return self.definition.compile_instance(snapshot, checkpoint)

    def package_validation(
        self, snapshot: SourceSnapshot, checkpoint: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self.definition.package_validation(snapshot, checkpoint)


@dataclass(frozen=True)
class MethodPlugin:
    key: str
    descriptor: MethodDescriptor
    model_factory: ModelFactory
    data: MethodDataFacet
    objective: ObjectiveFacet
    lifecycle: LifecycleFacet
    checkpoint: CheckpointCodec
    deployment: DeploymentCompiler

    def __post_init__(self) -> None:
        if not _PUBLIC_KEY.fullmatch(self.key) or "@" in self.key:
            raise ValueError("method plugin key must be lower-kebab without a version suffix")
        facets = {
            "model": self.model_factory,
            "data": self.data,
            "objective": self.objective,
            "lifecycle": self.lifecycle,
            "checkpoint": self.checkpoint,
            "deployment": self.deployment,
        }
        for name, facet in facets.items():
            require_sha256(
                f"method {self.key} {name} facet implementation",
                facet.implementation_sha256,
            )
        expected = {
            kind: tuple(fields)
            for kind, fields in self.descriptor.training_batch_requirements.items()
        }
        actual = {
            item.route_kind: item.fields
            for item in self.data.requirements()
        }
        if actual != expected:
            raise ValueError("method data facet requirements disagree with descriptor")

    @property
    def facet_identities(self) -> Mapping[str, str]:
        return {
            "model": self.model_factory.implementation_sha256,
            "data": self.data.implementation_sha256,
            "objective": self.objective.implementation_sha256,
            "lifecycle": self.lifecycle.implementation_sha256,
            "checkpoint": self.checkpoint.implementation_sha256,
            "deployment": self.deployment.implementation_sha256,
        }

    @classmethod
    def adapt_definition(
        cls,
        key: str,
        definition: MethodDefinition,
        *,
        source_adapter_factory: Callable[
            [Sequence[SourceSnapshot], torch.device], Any
        ],
    ) -> "MethodPlugin":
        return cls(
            key,
            definition.descriptor,
            _DefinitionModelFactory(definition, _facet_identity(definition, "model")),
            _DefinitionDataFacet(
                definition,
                source_adapter_factory,
                _facet_identity(definition, "data"),
            ),
            _DefinitionObjectiveFacet(
                definition, _facet_identity(definition, "objective")
            ),
            _DefinitionLifecycleFacet(
                definition, _facet_identity(definition, "lifecycle")
            ),
            _DefinitionCheckpointCodec(
                definition, _facet_identity(definition, "checkpoint")
            ),
            _DefinitionDeploymentCompiler(
                definition, _facet_identity(definition, "deployment")
            ),
        )


__all__ = [
    "CheckpointCodec",
    "DeploymentCompiler",
    "LifecycleFacet",
    "MethodDataFacet",
    "MethodPlugin",
    "ModelFactory",
    "ObjectiveFacet",
]
