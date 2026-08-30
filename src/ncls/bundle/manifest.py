from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from ncls.core.identity import require_sha256, safe_relative_uri, sha256_json
from ncls.core.scattering import validate_typed_parameter_view


FORMAT_NAME = "ncls.scattering-package"
FORMAT_VERSION = 2


def _validate_typed_group(
    group: Mapping[str, Any], files: Mapping[str, str], label: str
) -> dict[str, Any]:
    result = {str(name): dict(value) for name, value in group.items()}
    required = {"dtype", "shape", "stride", "alignment", "usage"}
    optional = {"kind", "module_name", "format", "color_space"}
    for logical_name, descriptor in result.items():
        if logical_name not in files:
            raise ValueError(f"{label} descriptor references an unknown logical file")
        fields = set(descriptor)
        if not required.issubset(fields) or not fields.issubset(required | optional):
            raise ValueError(
                f"{label} descriptors require {sorted(required)} and only {sorted(optional)}"
            )
        if int(descriptor["stride"]) < 1 or int(descriptor["alignment"]) < 1:
            raise ValueError(f"{label} descriptor stride/alignment must be positive")
        if descriptor.get("kind") == "slang-module-source":
            raise ValueError(
                f"{label} cannot contain Slang source; module_closure owns program source"
            )
        shape = descriptor["shape"]
        if not isinstance(shape, (list, tuple)) or not shape or any(int(value) < 1 for value in shape):
            raise ValueError(f"{label} descriptor shape must be positive")
    return result


def _validate_sampler_group(
    group: Mapping[str, Any], label: str
) -> dict[str, Any]:
    result = {str(name): dict(value) for name, value in group.items()}
    required = {"kind", "usage", "filter", "address_mode"}
    for logical_name, descriptor in result.items():
        if not logical_name or set(descriptor) != required:
            raise ValueError(f"{label} sampler descriptors require exactly {sorted(required)}")
        if (
            descriptor["kind"] != "sampler"
            or not str(descriptor["usage"])
            or descriptor["filter"] not in {"point", "linear", "anisotropic"}
            or descriptor["address_mode"] not in {"clamp", "wrap"}
        ):
            raise ValueError(f"{label} sampler descriptor is invalid")
    return result


@dataclass(frozen=True)
class ScatteringPackageManifest:
    package_id: str
    program_id: str
    asset_id: str
    instance_id: str
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
    asset: Mapping[str, Any]
    instance: Mapping[str, Any]
    validation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    files: Mapping[str, str]
    content_hashes: Mapping[str, str]
    format_name: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_name != FORMAT_NAME or self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported ScatteringPackage format")
        for name in (
            "package_id", "program_id", "asset_id", "instance_id",
            "program_descriptor_sha256", "source_snapshot_id",
        ):
            require_sha256(name, str(getattr(self, name)))
        if self.program_kind not in {"reference", "method"}:
            raise ValueError("program_kind must be reference or method")
        if not self.program_key or self.program_version < 1 or not self.source_family_id:
            raise ValueError("ScatteringPackage program/source identity is invalid")
        if self.source_contract_version < 1 or self.scattering_contract_version != 1:
            raise ValueError("unsupported source or scattering contract version")
        if not self.runtime_abi or self.capabilities <= 0:
            raise ValueError("ScatteringPackage runtime ABI and capabilities are required")

        files = {str(name): safe_relative_uri(str(uri)) for name, uri in self.files.items()}
        hashes = {
            safe_relative_uri(str(uri)): require_sha256(f"content hash {uri}", str(digest))
            for uri, digest in self.content_hashes.items()
        }
        if set(hashes) != set(files.values()) or len(set(files.values())) != len(files):
            raise ValueError("content_hashes must cover unique package files exactly")
        program = dict(self.program)
        asset = dict(self.asset)
        instance = dict(self.instance)
        if set(program) != {"module", "defines", "blobs", "samplers"}:
            raise ValueError("ScatteringPackage program fields are invalid")
        if set(asset) != {"blobs", "resources", "samplers"}:
            raise ValueError("ScatteringPackage asset fields are invalid")
        if set(instance) != {"bindings", "parameters", "blobs", "editor", "compiler"}:
            raise ValueError("ScatteringPackage instance fields are invalid")
        if not isinstance(program["defines"], Mapping):
            raise ValueError("ScatteringPackage program defines must be an object")
        defines = {str(name): str(value) for name, value in program["defines"].items()}
        if (
            len(defines) != len(program["defines"])
            or any(not name or not value for name, value in defines.items())
        ):
            raise ValueError("ScatteringPackage program defines must be unique strings")
        program["defines"] = defines
        module = str(program["module"])
        if module not in files:
            raise ValueError("ScatteringPackage program module is absent")
        program["blobs"] = _validate_typed_group(program["blobs"], files, "program blob")
        program["samplers"] = _validate_sampler_group(
            program["samplers"], "program"
        )
        asset["blobs"] = _validate_typed_group(asset["blobs"], files, "asset blob")
        asset["resources"] = _validate_typed_group(asset["resources"], files, "asset resource")
        asset["samplers"] = _validate_sampler_group(asset["samplers"], "asset")
        instance["blobs"] = _validate_typed_group(
            instance["blobs"], files, "instance blob"
        )
        binding_descriptors = (
            *program["blobs"].values(),
            *program["samplers"].values(),
            *asset["blobs"].values(),
            *asset["resources"].values(),
            *asset["samplers"].values(),
            *instance["blobs"].values(),
        )
        usages = [str(descriptor["usage"]) for descriptor in binding_descriptors]
        if any(not usage for usage in usages) or len(set(usages)) != len(usages):
            raise ValueError("ScatteringPackage typed binding usages must be unique")
        bindings = dict(instance["bindings"])
        if bindings != {"program_id": self.program_id, "asset_id": self.asset_id}:
            raise ValueError("ScatteringPackage instance bindings are not atomic")
        parameters = dict(instance["parameters"])
        if set(parameters) != {"compiled_material_index"} or int(parameters["compiled_material_index"]) < 0:
            raise ValueError("ScatteringPackage instance parameters are invalid")
        editor = dict(instance["editor"])
        compiler = dict(instance["compiler"])
        if bool(editor) != bool(compiler):
            raise ValueError("ScatteringPackage editable instance contract is incomplete")
        if editor:
            if (
                set(editor) != {"schema", "parameter_view", "raw_usage", "compiled_usage"}
                or editor["schema"] != "ncls.typed-material-editor@1"
                or not isinstance(editor["parameter_view"], Mapping)
                or set(compiler) != {"entry_point", "thread_group_size"}
                or not str(compiler["entry_point"])
                or not isinstance(compiler["thread_group_size"], (list, tuple))
                or len(compiler["thread_group_size"]) != 3
                or any(int(value) < 1 for value in compiler["thread_group_size"])
            ):
                raise ValueError("ScatteringPackage editable instance contract is invalid")
            validate_typed_parameter_view(editor["parameter_view"])
            if editor["parameter_view"]["snapshot_id"] != self.source_snapshot_id:
                raise ValueError("ScatteringPackage editor parameter_view snapshot mismatch")
            usages = {
                descriptor["usage"]: descriptor for descriptor in instance["blobs"].values()
            }
            if editor["raw_usage"] not in usages or editor["compiled_usage"] not in usages:
                raise ValueError("ScatteringPackage editor usages do not name instance blobs")
            if any(
                usages[usage].get("kind") != "mutable-structured-buffer"
                for usage in (editor["raw_usage"], editor["compiled_usage"])
            ):
                raise ValueError("ScatteringPackage editor buffers must be mutable")
        instance = {
            "bindings": bindings,
            "parameters": parameters,
            "blobs": instance["blobs"],
            "editor": editor,
            "compiler": compiler,
        }
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "validation", dict(self.validation))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "content_hashes", hashes)

        def section_hashes(prefix: str) -> dict[str, str]:
            return {
                files[name]: hashes[files[name]]
                for name in files
                if name.startswith(prefix)
            }

        program_identity = {
            "program_kind": self.program_kind,
            "program_key": self.program_key,
            "program_version": self.program_version,
            "program_descriptor_sha256": self.program_descriptor_sha256,
            "scattering_contract_version": self.scattering_contract_version,
            "runtime_abi": self.runtime_abi,
            "capabilities": self.capabilities,
            "program": program,
            "content_hashes": section_hashes("program/"),
        }
        if sha256_json(program_identity) != self.program_id:
            raise ValueError("program_id does not match program semantics")
        asset_identity = {
            "source_family_id": self.source_family_id,
            "source_contract_version": self.source_contract_version,
            "source_snapshot_id": self.source_snapshot_id,
            "program_descriptor_sha256": self.program_descriptor_sha256,
            "asset": asset,
            "content_hashes": section_hashes("asset/"),
        }
        if sha256_json(asset_identity) != self.asset_id:
            raise ValueError("asset_id does not match asset semantics")
        instance_identity = {
            "program_id": self.program_id,
            "asset_id": self.asset_id,
            "source_snapshot_id": self.source_snapshot_id,
            "instance": instance,
            "content_hashes": section_hashes("instance/"),
        }
        if sha256_json(instance_identity) != self.instance_id:
            raise ValueError("instance_id does not match instance semantics")
        package_identity = {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "program_id": self.program_id,
            "asset_id": self.asset_id,
            "instance_id": self.instance_id,
            "validation": dict(self.validation),
            "provenance": dict(self.provenance),
            "content_hashes": {
                uri: digest
                for uri, digest in hashes.items()
                if uri.startswith("validation/") or uri.startswith("provenance/")
            },
        }
        if sha256_json(package_identity) != self.package_id:
            raise ValueError("package_id does not match package semantics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "package_id": self.package_id,
            "program_id": self.program_id,
            "asset_id": self.asset_id,
            "instance_id": self.instance_id,
            "program_kind": self.program_kind,
            "program_key": self.program_key,
            "program_version": self.program_version,
            "program_descriptor_sha256": self.program_descriptor_sha256,
            "source_family_id": self.source_family_id,
            "source_contract_version": self.source_contract_version,
            "source_snapshot_id": self.source_snapshot_id,
            "scattering_contract_version": self.scattering_contract_version,
            "runtime_abi": self.runtime_abi,
            "capabilities": self.capabilities,
            "program": dict(self.program),
            "asset": dict(self.asset),
            "instance": dict(self.instance),
            "validation": dict(self.validation),
            "provenance": dict(self.provenance),
            "files": dict(self.files),
            "content_hashes": dict(self.content_hashes),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        ) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScatteringPackageManifest":
        required = {
            "format_name", "format_version", "package_id", "program_id", "asset_id",
            "instance_id", "program_kind", "program_key", "program_version",
            "program_descriptor_sha256", "source_family_id", "source_contract_version",
            "source_snapshot_id", "scattering_contract_version", "runtime_abi",
            "capabilities", "program", "asset", "instance", "validation",
            "provenance", "files", "content_hashes",
        }
        if set(value) != required:
            raise ValueError(f"ScatteringPackage manifest fields must be exactly {sorted(required)}")
        return cls(
            str(value["package_id"]),
            str(value["program_id"]),
            str(value["asset_id"]),
            str(value["instance_id"]),
            str(value["program_kind"]),
            str(value["program_key"]),
            int(value["program_version"]),
            str(value["program_descriptor_sha256"]),
            str(value["source_family_id"]),
            int(value["source_contract_version"]),
            str(value["source_snapshot_id"]),
            int(value["scattering_contract_version"]),
            str(value["runtime_abi"]),
            int(value["capabilities"]),
            value["program"],
            value["asset"],
            value["instance"],
            value["validation"],
            value["provenance"],
            value["files"],
            value["content_hashes"],
            str(value["format_name"]),
            int(value["format_version"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "ScatteringPackageManifest":
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("ScatteringPackage manifest root must be an object")
        return cls.from_dict(value)
