from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ncls.core.identity import require_sha256, sha256_bytes, sha256_file, sha256_json
from ncls.core.scattering.contract import BackendCapability, REQUIRED_PATH_TRACING_CAPABILITIES
from ncls.core.source import SourceSnapshot


@dataclass(frozen=True)
class FileResourcePayload:
    """内容寻址的只读大资源；plan常驻identity，group materialize时才读取。"""

    path: Path = field(compare=False)
    content_sha256: str
    size: int

    def __post_init__(self) -> None:
        path = self.path.resolve()
        require_sha256("file resource content_sha256", self.content_sha256)
        if self.size < 0 or not path.is_file() or path.stat().st_size != self.size:
            raise ValueError("file resource payload path/size is invalid")
        object.__setattr__(self, "path", path)

    @classmethod
    def from_path(cls, path: Path) -> "FileResourcePayload":
        resolved = path.resolve()
        return cls(resolved, sha256_file(resolved), resolved.stat().st_size)

    def read_bytes(self) -> bytes:
        payload = self.path.read_bytes()
        if len(payload) != self.size or sha256_bytes(payload) != self.content_sha256:
            raise ValueError("file resource payload changed after plan compilation")
        return payload


ResourcePayload = bytes | FileResourcePayload


def resource_payload_sha256(payload: ResourcePayload) -> str:
    return (
        payload.content_sha256
        if isinstance(payload, FileResourcePayload)
        else sha256_bytes(payload)
    )


def read_resource_payload(payload: ResourcePayload) -> bytes:
    return payload.read_bytes() if isinstance(payload, FileResourcePayload) else payload


def _validate_sampler_descriptors(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {str(name): dict(value) for name, value in values.items()}
    required = {"kind", "usage", "filter", "address_mode"}
    for name, descriptor in result.items():
        if (
            not name
            or set(descriptor) != required
            or descriptor["kind"] != "sampler"
            or not str(descriptor["usage"])
            or descriptor["filter"] not in {"point", "linear", "anisotropic"}
            or descriptor["address_mode"] not in {"clamp", "wrap"}
        ):
            raise ValueError("typed sampler descriptor is invalid")
    return result


@dataclass(frozen=True)
class RuntimePayload:
    program_module: str
    module_closure: Mapping[str, bytes]
    blobs: Mapping[str, bytes]
    blob_descriptors: Mapping[str, Mapping[str, Any]]
    capabilities: int
    defines: Mapping[str, str] = field(default_factory=dict)
    sampler_descriptors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

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
        object.__setattr__(
            self,
            "sampler_descriptors",
            _validate_sampler_descriptors(self.sampler_descriptors),
        )


@dataclass(frozen=True)
class MaterialPayload:
    source_snapshot_id: str
    blobs: Mapping[str, bytes]
    blob_descriptors: Mapping[str, Mapping[str, Any]]
    resources: Mapping[str, ResourcePayload] = field(default_factory=dict)
    resource_descriptors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    sampler_descriptors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_sha256("material payload source_snapshot_id", self.source_snapshot_id)
        if set(self.blobs) != set(self.blob_descriptors):
            raise ValueError("material blob descriptors must cover blobs exactly")
        if set(self.resources) != set(self.resource_descriptors):
            raise ValueError("material resource descriptors must cover resources exactly")
        if any(
            not isinstance(payload, (bytes, FileResourcePayload))
            for payload in self.resources.values()
        ):
            raise TypeError("material resources must be bytes or FileResourcePayload")
        object.__setattr__(self, "blobs", dict(self.blobs))
        object.__setattr__(self, "blob_descriptors", {name: dict(value) for name, value in self.blob_descriptors.items()})
        object.__setattr__(self, "resources", dict(self.resources))
        object.__setattr__(self, "resource_descriptors", {name: dict(value) for name, value in self.resource_descriptors.items()})
        object.__setattr__(
            self,
            "sampler_descriptors",
            _validate_sampler_descriptors(self.sampler_descriptors),
        )


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


@dataclass(frozen=True)
class ReferenceProgramProviderStatus:
    requirement_id: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if not self.requirement_id or self.status not in {"ready", "missing", "invalid"}:
            raise ValueError("reference program provider status is invalid")


class ReferenceProgramDefinition(ABC):
    descriptor: ReferenceProgramDescriptor

    def validate_snapshot(self, snapshot: SourceSnapshot) -> None:
        if (
            snapshot.family_id != self.descriptor.family_id
            or snapshot.source_contract_version != self.descriptor.source_contract_version
        ):
            raise ValueError("source snapshot is incompatible with reference program")

    def preflight_provider(
        self, *, platform_id: str, project_root: Path
    ) -> tuple[ReferenceProgramProviderStatus, ...]:
        """报告 program 私有编译器要求；不得检查或获取 source assets。"""

        del platform_id, project_root
        return ()

    def execution_group_key(
        self, snapshot: SourceSnapshot, material: MaterialPayload
    ) -> str:
        """返回可共用一次 runtime、generated module 与资源表的 group 身份。"""

        self.validate_snapshot(snapshot)
        if material.source_snapshot_id != snapshot.snapshot_id:
            raise ValueError("reference material payload belongs to another snapshot")
        module_sources = {
            name: {
                "sha256": sha256_bytes(material.blobs[name]),
                "descriptor": dict(descriptor),
            }
            for name, descriptor in material.blob_descriptors.items()
            if descriptor.get("kind") == "slang-module-source"
        }
        resources = {
            name: {
                "sha256": resource_payload_sha256(payload),
                "descriptor": dict(material.resource_descriptors[name]),
            }
            for name, payload in material.resources.items()
        }
        return sha256_json(
            {
                "reference_program_descriptor": self.descriptor.descriptor_sha256,
                "runtime_abi": self.descriptor.runtime_abi,
                "material_modules": module_sources,
                "resource_table": resources,
                "samplers": {
                    name: dict(value)
                    for name, value in material.sampler_descriptors.items()
                },
            }
        )

    def execution_group_layout(
        self, materials: Sequence[MaterialPayload]
    ) -> tuple[tuple[int, int], ...]:
        """返回每个 group-local material 的 argument/RO byte offsets。"""

        return tuple((0, 0) for _ in materials)

    def compile_execution_group_bindings(
        self,
        materials: Sequence[MaterialPayload],
        layouts: Sequence[tuple[int, int]],
    ) -> tuple[Mapping[str, bytes], Mapping[str, Mapping[str, Any]]]:
        """生成只在一个 execution group 内存在的 typed bindings。"""

        if len(materials) != len(layouts):
            raise ValueError("reference group materials/layouts must align")
        return {}, {}

    @abstractmethod
    def compile_runtime(self) -> RuntimePayload:
        raise NotImplementedError

    @abstractmethod
    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        raise NotImplementedError
