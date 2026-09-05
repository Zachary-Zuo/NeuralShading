from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ncls.bundle import ScatteringPackage
from ncls.core.identity import sha256_file, sha256_json
from ncls.core.scattering import validate_typed_parameter_view


FORMAT_NAME = "ncls.viewer-material-catalog"
FORMAT_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_VALUE_TYPES = {"bool", "int", "enum", "float", "vector2", "color3"}
_RESPONSIBILITIES = {
    "coordinates",
    "frame",
    "metal-core",
    "finish-microstructure",
    "aging-contamination",
    "coating-composite",
}
RESPONSIBILITY_ORDER = (
    "metal-core",
    "finish-microstructure",
    "aging-contamination",
    "coating-composite",
    "coordinates",
    "frame",
)
_RESPONSIBILITY_LABELS = {
    "metal-core": "Metal core",
    "finish-microstructure": "Finish / microstructure",
    "aging-contamination": "Aging / contamination",
    "coating-composite": "Coating / composite",
    "coordinates": "Coordinates",
    "frame": "Frame",
}
_MDL_VALUE_TYPES = {
    "bool": "bool",
    "int": "int",
    "enum": "enum",
    "float": "float",
    "float2": "vector2",
    "color": "color3",
}


def _require_sha256(label: str, value: Any) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"ViewerMaterialCatalog {label} must be a lowercase SHA-256")
    return result


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    result = dict(value)
    if set(result) != fields:
        raise ValueError(
            f"ViewerMaterialCatalog {label} fields must be exactly {sorted(fields)}"
        )
    return result


def _contained(root: Path, uri: Any, label: str) -> Path:
    relative = Path(str(uri))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"ViewerMaterialCatalog {label} URI must stay below catalog root")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"ViewerMaterialCatalog {label} URI escapes catalog root"
        ) from error
    return candidate


def _walk_parameter_nodes(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    children = node.get("children", [])
    if not isinstance(children, list):
        raise ValueError("ViewerMaterialCatalog parameter children must be an array")
    if bool(node.get("editable", False)):
        result.append(dict(node))
    for child in children:
        if not isinstance(child, Mapping):
            raise ValueError("ViewerMaterialCatalog parameter nodes must be objects")
        result.extend(_walk_parameter_nodes(child))
    return result


def link_parameter_view(
    parameter_view: Mapping[str, Any],
    parameters: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Attach registry-authored MDL writes/responsibilities to one runtime editor view."""

    result = deepcopy(dict(parameter_view))
    validate_typed_parameter_view(result)
    nodes = {
        str(node["path"]).removeprefix("/arguments/"): node
        for node in _walk_parameter_nodes(result["root"])
    }
    editable_parameters = [dict(item) for item in parameters if bool(item.get("editable"))]
    names = [str(item.get("name", "")) for item in editable_parameters]
    if len(set(names)) != len(names) or set(nodes) != set(names):
        raise ValueError(
            "ViewerMaterialCatalog registry parameters do not match runtime editor paths"
        )
    grouped: dict[str, list[dict[str, Any]]] = {
        name: [] for name in RESPONSIBILITY_ORDER
    }
    for parameter in editable_parameters:
        name = str(parameter["name"])
        mdl_type = str(parameter.get("type", ""))
        expected_type = _MDL_VALUE_TYPES.get(mdl_type)
        node = nodes[name]
        if expected_type is None or node.get("value_type") != expected_type:
            raise ValueError(
                f"ViewerMaterialCatalog unsupported or mismatched MDL parameter {name!r}"
            )
        responsibility = str(parameter.get("responsibility", ""))
        if responsibility not in grouped:
            raise ValueError(
                f"ViewerMaterialCatalog parameter {name!r} has unknown responsibility"
            )
        size = int(parameter.get("size", 0))
        expected_size = {
            "bool": 1,
            "int": 4,
            "enum": 4,
            "float": 4,
            "float2": 8,
            "color": 12,
        }[mdl_type]
        if size != expected_size:
            raise ValueError(
                f"ViewerMaterialCatalog parameter {name!r} has an invalid MDL write size"
            )
        write: dict[str, Any] = {
            "offset": int(parameter.get("offset", -1)),
            "size": size,
            "mdl_type": mdl_type,
        }
        if mdl_type == "enum":
            choices = parameter.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"ViewerMaterialCatalog enum parameter {name!r} has no choices"
                )
            write["choices"] = {
                str(choice["name"]): int(choice["value"]) for choice in choices
            }
            if list(write["choices"]) != list(node.get("choices", [])):
                raise ValueError(
                    f"ViewerMaterialCatalog enum parameter {name!r} choices drifted"
                )
        metadata = dict(node.get("metadata", {}))
        metadata["responsibility"] = responsibility
        metadata["reference_write"] = write
        node["metadata"] = metadata
        grouped[responsibility].append(node)

    result["root"]["children"] = [
        {
            "path": f"/responsibilities/{responsibility}",
            "kind": "group",
            "label": _RESPONSIBILITY_LABELS[responsibility],
            "children": grouped[responsibility],
            "editable": False,
            "allowed_operations": [],
        }
        for responsibility in RESPONSIBILITY_ORDER
        if grouped[responsibility]
    ]
    validate_typed_parameter_view(result)
    return result


@dataclass(frozen=True)
class ViewerMaterialEntry:
    export_id: str
    display_name: str
    metal: str
    finish: str
    graph_id: str
    texture_set_id: str
    parameter_schema_id: str
    source_snapshot_id: str
    artifact_sha256: str
    artifact_root: Path
    package_id: str
    package_root: Path | None
    program_id: str
    asset_id: str
    instance_id: str
    parameter_view: dict[str, Any]


@dataclass(frozen=True)
class ViewerMaterialCatalog:
    source_path: Path
    catalog_id: str
    registry_identity: str
    registry_sha256: str
    opaque_entry_count: int
    rejected_cutout_count: int
    mdl_sdk: str
    target_code_types: Path
    renderer_runtime: Path
    default_export_id: str
    entries: tuple[ViewerMaterialEntry, ...]

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        source_path: Path | str,
        verify_payloads: bool = True,
    ) -> "ViewerMaterialCatalog":
        document = _exact(
            value,
            {
                "schema_name",
                "schema_version",
                "catalog_id",
                "registry",
                "reference_runtime",
                "default_export_id",
                "entries",
            },
            "root",
        )
        if (
            document["schema_name"] != FORMAT_NAME
            or document["schema_version"] != FORMAT_VERSION
        ):
            raise ValueError("unsupported ViewerMaterialCatalog format")
        declared_catalog_id = _require_sha256("catalog_id", document["catalog_id"])
        identity_document = dict(document)
        identity_document.pop("catalog_id")
        if sha256_json(identity_document) != declared_catalog_id:
            raise ValueError("ViewerMaterialCatalog catalog_id does not match semantics")

        registry_identity = registry_sha256 = ""
        opaque_count, rejected_count = len(document["entries"]), 0
        if document["registry"] is not None:
            registry = _exact(
                document["registry"],
                {"identity", "sha256", "opaque_entry_count", "rejected_cutout_count"},
                "registry",
            )
            registry_identity = _require_sha256("registry.identity", registry["identity"])
            registry_sha256 = _require_sha256("registry.sha256", registry["sha256"])
            opaque_count = int(registry["opaque_entry_count"])
            rejected_count = int(registry["rejected_cutout_count"])
            if opaque_count <= 0 or rejected_count < 0:
                raise ValueError("ViewerMaterialCatalog registry counts are invalid")

        source = Path(source_path).resolve()
        root = source.parent
        runtime = _exact(
            document["reference_runtime"],
            {"mdl_sdk", "target_code_types", "renderer_runtime"},
            "reference_runtime",
        )
        mdl_sdk = str(runtime["mdl_sdk"])
        if not mdl_sdk:
            raise ValueError("ViewerMaterialCatalog MDL SDK identity is empty")

        def runtime_file(name: str) -> Path:
            descriptor = _exact(runtime[name], {"path", "sha256"}, name)
            path = _contained(root, descriptor["path"], name)
            expected = _require_sha256(f"{name}.sha256", descriptor["sha256"])
            if verify_payloads and (
                not path.is_file() or sha256_file(path) != expected
            ):
                raise ValueError(f"ViewerMaterialCatalog {name} is missing or drifted")
            return path

        target_code_types = runtime_file("target_code_types")
        renderer_runtime = runtime_file("renderer_runtime")

        raw_entries = document["entries"]
        if not isinstance(raw_entries, list) or not raw_entries or len(raw_entries) != opaque_count:
            raise ValueError("ViewerMaterialCatalog entries do not match opaque count")
        entries: list[ViewerMaterialEntry] = []
        unique: dict[str, set[str]] = {
            name: set()
            for name in ("export_id", "source_snapshot_id", "package_id", "instance_id")
        }
        program_ids: set[str] = set()
        for raw in raw_entries:
            item = _exact(
                raw,
                {
                    "export_id",
                    "display_name",
                    "metal",
                    "finish",
                    "graph_id",
                    "texture_set_id",
                    "parameter_schema_id",
                    "source_snapshot_id",
                    "artifact_sha256",
                    "artifact_root",
                    "package_id",
                    "package_root",
                    "program_id",
                    "asset_id",
                    "instance_id",
                    "parameter_view",
                },
                "entry",
            )
            optional_ids = {"graph_id", "texture_set_id", "parameter_schema_id", "package_id", "program_id", "asset_id", "instance_id"}
            binding_fields = ("package_id", "package_root", "program_id", "asset_id", "instance_id")
            has_binding = item["package_id"] is not None
            if any((item[name] is not None) != has_binding for name in binding_fields):
                raise ValueError("ViewerMaterialCatalog partial package binding")
            identities = {
                name: "" if name in optional_ids and item[name] is None else _require_sha256(f"entry.{name}", item[name])
                for name in (
                    "export_id",
                    "graph_id",
                    "texture_set_id",
                    "parameter_schema_id",
                    "source_snapshot_id",
                    "artifact_sha256",
                    "package_id",
                    "program_id",
                    "asset_id",
                    "instance_id",
                )
            }
            for name in unique:
                if not identities[name]:
                    continue
                if identities[name] in unique[name]:
                    raise ValueError(f"ViewerMaterialCatalog duplicate {name}")
                unique[name].add(identities[name])
            if has_binding:
                program_ids.add(identities["program_id"])
            display_name = str(item["display_name"])
            metal = str(item["metal"] or "")
            finish = str(item["finish"] or "")
            if not display_name:
                raise ValueError("ViewerMaterialCatalog entry taxonomy is incomplete")
            artifact_root = _contained(root, item["artifact_root"], "artifact_root")
            package_root = _contained(root, item["package_root"], "package_root") if has_binding else None
            parameter_view: dict[str, Any] = {}
            parameter_nodes: list[dict[str, Any]] = []
            if item["parameter_view"] is not None:
                parameter_view = dict(item["parameter_view"])
                validate_typed_parameter_view(parameter_view)
                if parameter_view.get("snapshot_id") != identities["source_snapshot_id"]:
                    raise ValueError(
                        "ViewerMaterialCatalog parameter view snapshot does not match entry"
                    )
                parameter_paths: set[str] = set()
                parameter_nodes = _walk_parameter_nodes(parameter_view["root"])
                for node in parameter_nodes:
                    path = str(node.get("path", ""))
                    value_type = str(node.get("value_type", ""))
                    metadata = node.get("metadata")
                    if path in parameter_paths or not path.startswith("/arguments/"):
                        raise ValueError("ViewerMaterialCatalog parameter paths are invalid")
                    parameter_paths.add(path)
                    if value_type not in _SUPPORTED_VALUE_TYPES:
                        raise ValueError(
                            f"ViewerMaterialCatalog unsupported editable type: {value_type}"
                        )
                    if not isinstance(metadata, Mapping):
                        raise ValueError("ViewerMaterialCatalog editable parameter has no metadata")
                    responsibility = str(metadata.get("responsibility", ""))
                    reference_write = metadata.get("reference_write")
                    if responsibility not in _RESPONSIBILITIES:
                        raise ValueError(
                            "ViewerMaterialCatalog editable parameter responsibility is invalid"
                        )
                    if not isinstance(reference_write, Mapping) or set(reference_write) not in (
                        {"offset", "size", "mdl_type"},
                        {"offset", "size", "mdl_type", "choices"},
                    ):
                        raise ValueError(
                            "ViewerMaterialCatalog editable parameter reference write is invalid"
                        )
                    if int(reference_write["offset"]) < 0 or int(reference_write["size"]) <= 0:
                        raise ValueError(
                            "ViewerMaterialCatalog reference write range is invalid"
                        )
                    expected_write = {
                        "bool": ("bool", 1),
                        "int": ("int", 4),
                        "enum": ("enum", 4),
                        "float": ("float", 4),
                        "vector2": ("float2", 8),
                        "color3": ("color", 12),
                    }[value_type]
                    if (
                        str(reference_write["mdl_type"]) != expected_write[0]
                        or int(reference_write["size"]) != expected_write[1]
                    ):
                        raise ValueError(
                            "ViewerMaterialCatalog reference write type/size is invalid"
                        )
                    if value_type == "enum":
                        choices = reference_write.get("choices")
                        node_choices = node.get("choices")
                        if (
                            not isinstance(choices, Mapping)
                            or not isinstance(node_choices, list)
                            or list(choices) != node_choices
                            or not all(isinstance(item, int) for item in choices.values())
                        ):
                            raise ValueError(
                                "ViewerMaterialCatalog enum reference write is invalid"
                            )

            if verify_payloads:
                from ncls.references.mdl import MdlCompiledArtifact

                artifact = MdlCompiledArtifact.load(artifact_root)
                if artifact.artifact_sha256 != identities["artifact_sha256"]:
                    raise ValueError(
                        "ViewerMaterialCatalog MDL artifact identity mismatch"
                    )
                for node in parameter_nodes:
                    write = node["metadata"]["reference_write"]
                    if int(write["offset"]) + int(write["size"]) > len(
                        artifact.argument_block
                    ):
                        raise ValueError(
                            "ViewerMaterialCatalog reference write exceeds argument block"
                        )
                if has_binding:
                    package = ScatteringPackage.open(package_root)
                    manifest = package.manifest
                    if (
                        manifest.package_id != identities["package_id"]
                        or manifest.program_id != identities["program_id"]
                        or manifest.asset_id != identities["asset_id"]
                        or manifest.instance_id != identities["instance_id"]
                        or manifest.source_snapshot_id != identities["source_snapshot_id"]
                    ):
                        raise ValueError(
                            "ViewerMaterialCatalog package binding identity mismatch"
                        )

            entries.append(
                ViewerMaterialEntry(
                    identities["export_id"],
                    display_name,
                    metal,
                    finish,
                    identities["graph_id"],
                    identities["texture_set_id"],
                    identities["parameter_schema_id"],
                    identities["source_snapshot_id"],
                    identities["artifact_sha256"],
                    artifact_root,
                    identities["package_id"],
                    package_root,
                    identities["program_id"],
                    identities["asset_id"],
                    identities["instance_id"],
                    parameter_view,
                )
            )
        default_export_id = _require_sha256(
            "default_export_id", document["default_export_id"]
        )
        if default_export_id not in unique["export_id"]:
            raise ValueError("ViewerMaterialCatalog default export is absent")
        return cls(
            source,
            declared_catalog_id,
            registry_identity,
            registry_sha256,
            opaque_count,
            rejected_count,
            mdl_sdk,
            target_code_types,
            renderer_runtime,
            default_export_id,
            tuple(entries),
        )

    @classmethod
    def open(
        cls, path: Path | str, *, verify_payloads: bool = True
    ) -> "ViewerMaterialCatalog":
        source = Path(path).resolve()
        if not source.is_file():
            raise ValueError(f"ViewerMaterialCatalog is missing: {source}")
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("ViewerMaterialCatalog root must be an object")
        return cls.from_dict(
            value, source_path=source, verify_payloads=verify_payloads
        )


def finalize_catalog_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add the canonical catalog identity to an otherwise complete document."""

    result = dict(value)
    if "catalog_id" in result:
        raise ValueError("ViewerMaterialCatalog writer input already has catalog_id")
    result["catalog_id"] = sha256_json(result)
    return result


def source_catalog_entry(
    *, export_id: str, display_name: str, source_snapshot_id: str,
    artifact_sha256: str, artifact_root: str,
) -> dict[str, Any]:
    """固定 source entry；未提供的 taxonomy、编辑和 neural 绑定明确为 null。"""
    return {
        "export_id": export_id, "display_name": display_name,
        "source_snapshot_id": source_snapshot_id, "artifact_sha256": artifact_sha256,
        "artifact_root": artifact_root,
        **{name: None for name in (
            "metal", "finish", "graph_id", "texture_set_id", "parameter_schema_id",
            "package_id", "package_root", "program_id", "asset_id", "instance_id", "parameter_view",
        )},
    }


def source_catalog_document(
    *, mdl_sdk: str, target_code_types: Mapping[str, str],
    renderer_runtime: Mapping[str, str], default_export_id: str,
    entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return finalize_catalog_document({
        "schema_name": FORMAT_NAME, "schema_version": FORMAT_VERSION,
        "registry": None,
        "reference_runtime": {
            "mdl_sdk": mdl_sdk, "target_code_types": dict(target_code_types),
            "renderer_runtime": dict(renderer_runtime),
        },
        "default_export_id": default_export_id, "entries": [dict(entry) for entry in entries],
    })
