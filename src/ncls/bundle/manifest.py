from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from ncls.core.identity import require_sha256, safe_relative_uri, sha256_json


FORMAT_NAME = "ncls.scattering-package"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class ScatteringPackageManifest:
    package_id: str
    program_runtime_id: str
    material_asset_id: str
    program_kind: str
    program_key: str
    program_version: int
    program_descriptor_sha256: str
    source_family_id: str
    source_contract_version: int
    source_snapshot_id: str
    scattering_contract_version: int
    runtime_abi: str
    capabilities: int
    program: Mapping[str, Any]
    material: Mapping[str, Any]
    validation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    files: Mapping[str, str]
    content_hashes: Mapping[str, str]
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_name != FORMAT_NAME or self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported ScatteringPackage format")
        for name in ("package_id", "program_runtime_id", "material_asset_id", "program_descriptor_sha256", "source_snapshot_id"):
            require_sha256(name, str(getattr(self, name)))
        if self.program_kind not in {"reference", "method"}:
            raise ValueError("program_kind must be reference or method")
        if not self.program_key or self.program_version < 1 or not self.source_family_id:
            raise ValueError("ScatteringPackage program/source identity is invalid")
        if self.source_contract_version < 1 or self.scattering_contract_version != 1:
            raise ValueError("unsupported source or scattering contract version")
        if not self.runtime_abi or self.capabilities <= 0:
            raise ValueError("ScatteringPackage runtime ABI and capabilities are required")
        program = dict(self.program)
        material = dict(self.material)
        if set(program) != {"module", "defines", "blobs"} or set(material) != {"blobs", "resources"}:
            raise ValueError("ScatteringPackage program/material descriptors have unknown or missing fields")
        if not all(isinstance(program[name], Mapping) for name in ("defines", "blobs")):
            raise ValueError("runtime defines and blobs must be objects")
        if not all(isinstance(material[name], Mapping) for name in ("blobs", "resources")):
            raise ValueError("material blobs and resources must be objects")
        files = {str(name): safe_relative_uri(str(uri)) for name, uri in self.files.items()}
        hashes = {safe_relative_uri(str(uri)): require_sha256(f"content hash {uri}", str(digest)) for uri, digest in self.content_hashes.items()}
        if set(hashes) != set(files.values()) or len(set(files.values())) != len(files):
            raise ValueError("content_hashes must cover unique package files exactly")
        module = str(program["module"])
        if module not in files:
            raise ValueError("program.module must name a logical package file")
        for group in (program["blobs"], material["blobs"], material["resources"]):
            for logical_name, descriptor in group.items():
                if logical_name not in files or not isinstance(descriptor, Mapping):
                    raise ValueError("blob/resource descriptor must reference a logical package file")
                required = {"dtype", "shape", "stride", "alignment", "usage"}
                optional = {"kind", "module_name", "format", "color_space"}
                fields = set(descriptor)
                if not required.issubset(fields) or not fields.issubset(required | optional):
                    raise ValueError(
                        "typed payload descriptor fields must contain "
                        f"{sorted(required)} and only use optional fields {sorted(optional)}"
                    )
                if int(descriptor["stride"]) < 1 or int(descriptor["alignment"]) < 1:
                    raise ValueError("typed blob stride and alignment must be positive")
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "validation", dict(self.validation))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "content_hashes", hashes)
        program_identity = {
            "program_kind": self.program_kind,
            "program_key": self.program_key,
            "program_version": self.program_version,
            "program_descriptor_sha256": self.program_descriptor_sha256,
            "scattering_contract_version": self.scattering_contract_version,
            "runtime_abi": self.runtime_abi,
            "capabilities": self.capabilities,
            "program": program,
            "content_hashes": {files[name]: hashes[files[name]] for name in files if name == module or name.startswith("runtime/")},
        }
        if sha256_json(program_identity) != self.program_runtime_id:
            raise ValueError("program_runtime_id does not match runtime semantics")
        material_identity = {
            "source_family_id": self.source_family_id,
            "source_contract_version": self.source_contract_version,
            "source_snapshot_id": self.source_snapshot_id,
            "program_descriptor_sha256": self.program_descriptor_sha256,
            "material": material,
            "content_hashes": {files[name]: hashes[files[name]] for name in files if name.startswith("material/")},
        }
        if sha256_json(material_identity) != self.material_asset_id:
            raise ValueError("material_asset_id does not match material semantics")
        package_identity = {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "program_runtime_id": self.program_runtime_id,
            "material_asset_id": self.material_asset_id,
            "validation": dict(self.validation),
            "provenance": dict(self.provenance),
            "content_hashes": {uri: digest for uri, digest in hashes.items() if uri.startswith("validation/") or uri.startswith("provenance/")},
        }
        if sha256_json(package_identity) != self.package_id:
            raise ValueError("package_id does not match package semantics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name, "format_version": self.format_version,
            "package_id": self.package_id, "program_runtime_id": self.program_runtime_id,
            "material_asset_id": self.material_asset_id, "program_kind": self.program_kind,
            "program_key": self.program_key, "program_version": self.program_version,
            "program_descriptor_sha256": self.program_descriptor_sha256,
            "source_family_id": self.source_family_id, "source_contract_version": self.source_contract_version,
            "source_snapshot_id": self.source_snapshot_id, "scattering_contract_version": self.scattering_contract_version,
            "runtime_abi": self.runtime_abi, "capabilities": self.capabilities,
            "program": dict(self.program), "material": dict(self.material),
            "validation": dict(self.validation), "provenance": dict(self.provenance),
            "files": dict(self.files), "content_hashes": dict(self.content_hashes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScatteringPackageManifest":
        required = {
            "format_name", "format_version", "package_id", "program_runtime_id", "material_asset_id",
            "program_kind", "program_key", "program_version", "program_descriptor_sha256",
            "source_family_id", "source_contract_version", "source_snapshot_id",
            "scattering_contract_version", "runtime_abi", "capabilities", "program", "material",
            "validation", "provenance", "files", "content_hashes",
        }
        if set(value) != required:
            raise ValueError(f"ScatteringPackage manifest fields must be exactly {sorted(required)}")
        return cls(
            str(value["package_id"]), str(value["program_runtime_id"]), str(value["material_asset_id"]),
            str(value["program_kind"]), str(value["program_key"]), int(value["program_version"]),
            str(value["program_descriptor_sha256"]), str(value["source_family_id"]),
            int(value["source_contract_version"]), str(value["source_snapshot_id"]),
            int(value["scattering_contract_version"]), str(value["runtime_abi"]), int(value["capabilities"]),
            value["program"], value["material"], value["validation"], value["provenance"],
            value["files"], value["content_hashes"], str(value["format_name"]), int(value["format_version"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "ScatteringPackageManifest":
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("ScatteringPackage manifest root must be an object")
        return cls.from_dict(value)
