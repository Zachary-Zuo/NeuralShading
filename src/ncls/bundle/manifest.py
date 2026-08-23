from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from ncls.core.scattering import BackendDescriptor


FORMAT_NAME = "ncls.method-bundle"
FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _relative_uri(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"bundle URI must be a safe POSIX-relative path: {value!r}")
    return value


@dataclass(frozen=True)
class MethodBundleManifest:
    method_id: str
    display_name: str
    created_at: str
    source_git_commit: str
    material_program_schema_versions: tuple[int, ...]
    supported_ir_ids: tuple[str, ...]
    scattering_contract_version: int
    backend_id: str
    backend_version: int
    backend_descriptor: Mapping[str, Any]
    runtime_class: str
    compiler: Mapping[str, Any]
    runtime: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    cost_claims: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    validation_provenance: Mapping[str, Any]
    files: Mapping[str, str]
    content_hashes: Mapping[str, str]
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_name != FORMAT_NAME or self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported MethodBundle format")
        if not _SHA256.fullmatch(self.method_id):
            raise ValueError("method_id must be a lowercase SHA-256 digest")
        if not self.display_name or not self.created_at or not self.source_git_commit:
            raise ValueError("MethodBundle identity fields must be nonempty")
        if self.runtime_class not in {"realtime", "diagnostic"}:
            raise ValueError("runtime_class must be realtime or diagnostic")
        if self.scattering_contract_version != 1:
            raise ValueError("unsupported scattering contract version")
        if not self.material_program_schema_versions or not self.supported_ir_ids:
            raise ValueError("MethodBundle must declare supported material contracts")
        descriptor = BackendDescriptor.from_dict(self.backend_descriptor)
        if descriptor.backend_id != self.backend_id or descriptor.backend_version != self.backend_version:
            raise ValueError("backend descriptor identity disagrees with manifest")
        if descriptor.scattering_contract_version != self.scattering_contract_version:
            raise ValueError("backend descriptor scattering contract disagrees with manifest")
        if tuple(descriptor.supported_ir_ids) != tuple(self.supported_ir_ids):
            raise ValueError("backend descriptor IR support disagrees with manifest")
        if self.runtime_class == "realtime" and not descriptor.is_complete_realtime_backend:
            raise ValueError("realtime MethodBundle requires a complete bounded backend")
        allowed_compilers = {"analytic", "parameter-network", "latent", "direct-neural"}
        if self.compiler.get("kind") not in allowed_compilers:
            raise ValueError("unsupported MethodBundle compiler kind")
        if set(self.content_hashes) != set(self.files.values()):
            raise ValueError("content_hashes must cover every bundle file exactly once")
        for uri in self.files.values():
            _relative_uri(uri)
        for uri, digest in self.content_hashes.items():
            _relative_uri(uri)
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"invalid content hash for {uri}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "method_id": self.method_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "source_git_commit": self.source_git_commit,
            "material_program_schema_versions": list(self.material_program_schema_versions),
            "supported_ir_ids": list(self.supported_ir_ids),
            "scattering_contract_version": self.scattering_contract_version,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "backend_descriptor": dict(self.backend_descriptor),
            "runtime_class": self.runtime_class,
            "compiler": dict(self.compiler),
            "runtime": dict(self.runtime),
            "capabilities": dict(self.capabilities),
            "cost_claims": dict(self.cost_claims),
            "training_provenance": dict(self.training_provenance),
            "validation_provenance": dict(self.validation_provenance),
            "files": dict(self.files),
            "content_hashes": dict(self.content_hashes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MethodBundleManifest:
        return cls(
            method_id=str(value["method_id"]),
            display_name=str(value["display_name"]),
            created_at=str(value["created_at"]),
            source_git_commit=str(value["source_git_commit"]),
            material_program_schema_versions=tuple(int(item) for item in value["material_program_schema_versions"]),
            supported_ir_ids=tuple(str(item) for item in value["supported_ir_ids"]),
            scattering_contract_version=int(value["scattering_contract_version"]),
            backend_id=str(value["backend_id"]),
            backend_version=int(value["backend_version"]),
            backend_descriptor=value["backend_descriptor"],
            runtime_class=str(value["runtime_class"]),
            compiler=value["compiler"],
            runtime=value["runtime"],
            capabilities=value["capabilities"],
            cost_claims=value["cost_claims"],
            training_provenance=value["training_provenance"],
            validation_provenance=value["validation_provenance"],
            files={str(k): str(v) for k, v in value["files"].items()},
            content_hashes={str(k): str(v) for k, v in value["content_hashes"].items()},
            format_name=str(value["format_name"]),
            format_version=int(value["format_version"]),
        )

    @classmethod
    def from_json(cls, text: str) -> MethodBundleManifest:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("MethodBundle manifest root must be an object")
        return cls.from_dict(value)
