from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ncls.core.identity import require_sha256, sha256_json
from ncls.core.scattering.contract import BackendCapability, REQUIRED_PATH_TRACING_CAPABILITIES
from ncls.core.source import SourceSnapshot


@dataclass(frozen=True)
class RuntimePayload:
    program_module: str
    module_closure: Mapping[str, bytes]
    blobs: Mapping[str, bytes]
    blob_descriptors: Mapping[str, Mapping[str, Any]]
    capabilities: int
    defines: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.program_module or self.program_module not in self.module_closure:
            raise ValueError("runtime payload program module must exist in module closure")
        if self.capabilities <= 0:
            raise ValueError("runtime payload capabilities must be nonzero")
        if set(self.blobs) != set(self.blob_descriptors):
            raise ValueError("runtime blob descriptors must cover blobs exactly")
        object.__setattr__(self, "module_closure", dict(self.module_closure))
        object.__setattr__(self, "blobs", dict(self.blobs))
        object.__setattr__(self, "blob_descriptors", {name: dict(value) for name, value in self.blob_descriptors.items()})
        if any(not name or not value for name, value in self.defines.items()):
            raise ValueError("runtime payload defines must have nonempty names and values")
        object.__setattr__(self, "defines", dict(self.defines))


@dataclass(frozen=True)
class MaterialPayload:
    source_snapshot_id: str
    blobs: Mapping[str, bytes]
    blob_descriptors: Mapping[str, Mapping[str, Any]]
    resources: Mapping[str, bytes] = field(default_factory=dict)
    resource_descriptors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_sha256("material payload source_snapshot_id", self.source_snapshot_id)
        if set(self.blobs) != set(self.blob_descriptors):
            raise ValueError("material blob descriptors must cover blobs exactly")
        if set(self.resources) != set(self.resource_descriptors) and self.resource_descriptors:
            raise ValueError("material resource descriptors must cover resources exactly")
        object.__setattr__(self, "blobs", dict(self.blobs))
        object.__setattr__(self, "blob_descriptors", {name: dict(value) for name, value in self.blob_descriptors.items()})
        object.__setattr__(self, "resources", dict(self.resources))
        object.__setattr__(self, "resource_descriptors", {name: dict(value) for name, value in self.resource_descriptors.items()})


@dataclass(frozen=True)
class ReferenceProgramDescriptor:
    program_key: str
    version: int
    display_name: str
    family_id: str
    source_contract_version: int
    implementation_sha256: str
    runtime_abi: str
    capabilities: int
    bounded_execution: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.program_key or self.version < 1 or not self.display_name or not self.family_id:
            raise ValueError("reference program descriptor identity is invalid")
        if self.source_contract_version < 1 or not self.runtime_abi or self.capabilities <= 0:
            raise ValueError("reference program contract is invalid")
        capabilities = BackendCapability(self.capabilities)
        if (capabilities & REQUIRED_PATH_TRACING_CAPABILITIES) != REQUIRED_PATH_TRACING_CAPABILITIES:
            missing = REQUIRED_PATH_TRACING_CAPABILITIES & ~capabilities
            raise ValueError(f"reference program is missing required path-tracing capabilities: {missing!s}")
        require_sha256("reference implementation_sha256", self.implementation_sha256)
        required = {"maximum_prepare_steps", "maximum_evaluate_steps", "maximum_state_bytes", "maximum_reads"}
        if set(self.bounded_execution) != required or any(int(value) < 1 for value in self.bounded_execution.values()):
            raise ValueError(f"bounded_execution fields must be exactly {sorted(required)}")
        object.__setattr__(self, "bounded_execution", dict(self.bounded_execution))

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_key": self.program_key,
            "version": self.version,
            "display_name": self.display_name,
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "implementation_sha256": self.implementation_sha256,
            "runtime_abi": self.runtime_abi,
            "capabilities": self.capabilities,
            "bounded_execution": dict(self.bounded_execution),
        }

    @property
    def descriptor_sha256(self) -> str:
        return sha256_json(self.to_dict())


class ReferenceProgramDefinition(ABC):
    descriptor: ReferenceProgramDescriptor

    def validate_snapshot(self, snapshot: SourceSnapshot) -> None:
        if (
            snapshot.family_id != self.descriptor.family_id
            or snapshot.source_contract_version != self.descriptor.source_contract_version
        ):
            raise ValueError("source snapshot is incompatible with reference program")

    @abstractmethod
    def compile_runtime(self) -> RuntimePayload:
        raise NotImplementedError

    @abstractmethod
    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        raise NotImplementedError
