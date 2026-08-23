from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


REFERENCE_PACKAGE_SCHEMA = "ncls.reference-package"
REFERENCE_PACKAGE_VERSION = 1
REFERENCE_REGISTRY_SCHEMA = "ncls.reference-registry"
REFERENCE_REGISTRY_VERSION = 2
PATH_ROOTS = {"project", "external", "source-materials"}
ROLES = {"ground-truth", "independent-validation"}
CAPABILITY_NAMES = {
    "source_adapter",
    "native_reference",
    "falcor_runtime",
    "viewer_integration",
    "numerical_parity",
    "image_parity",
}
CAPABILITY_STATES = {"ready", "pending", "not-applicable"}


def resolve_reference_path(
    project_root: str | Path,
    path_root: str,
    relative_path: str | Path,
) -> Path:
    """把 reference manifest 的逻辑根解析到唯一的项目目录。"""

    root = Path(project_root).resolve()
    bases = {
        "project": root,
        "external": root / "external",
        "source-materials": root / "assets" / "source-materials",
    }
    try:
        base = bases[path_root].resolve()
    except KeyError as error:
        raise ValueError(f"unsupported reference path_root {path_root!r}") from error
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("reference manifest path must be relative")
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError("reference manifest path escapes its path_root") from error
    return resolved


def _nonempty(name: str, value: Any) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _object(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class ReferenceImplementation:
    implementation_id: str
    kind: str
    path_root: str
    path: str
    upstream_url: str | None = None
    revision: str | None = None
    license: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceImplementation:
        path_root = _nonempty("path_root", value.get("path_root", ""))
        if path_root not in PATH_ROOTS:
            raise ValueError(f"unsupported implementation path_root {path_root!r}")
        return cls(
            _nonempty("implementation_id", value.get("implementation_id", "")),
            _nonempty("kind", value.get("kind", "")),
            path_root,
            _nonempty("path", value.get("path", "")),
            str(value["upstream_url"]) if value.get("upstream_url") is not None else None,
            str(value["revision"]) if value.get("revision") is not None else None,
            str(value["license"]) if value.get("license") is not None else None,
        )


@dataclass(frozen=True)
class ReferencePackage:
    reference_id: str
    source_material_family_id: str
    role: str
    native_representation: str
    query_contract: Mapping[str, Any]
    implementations: tuple[ReferenceImplementation, ...]
    dependencies: tuple[Mapping[str, Any], ...]
    source_assets: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferencePackage:
        if value.get("schema_name") != REFERENCE_PACKAGE_SCHEMA:
            raise ValueError("unsupported reference package schema_name")
        if value.get("schema_version") != REFERENCE_PACKAGE_VERSION:
            raise ValueError("unsupported reference package schema_version")
        role = _nonempty("role", value.get("role", ""))
        if role not in ROLES:
            raise ValueError(f"unsupported reference role {role!r}")
        implementations = tuple(
            ReferenceImplementation.from_dict(_object("implementation", item))
            for item in value.get("implementations", [])
        )
        if not implementations:
            raise ValueError("reference package requires at least one implementation")
        implementation_ids = [item.implementation_id for item in implementations]
        if len(implementation_ids) != len(set(implementation_ids)):
            raise ValueError("reference implementation IDs must be unique")

        def path_records(field: str, id_field: str) -> tuple[Mapping[str, Any], ...]:
            records = []
            for item in value.get(field, []):
                record = dict(_object(field, item))
                _nonempty(id_field, record.get(id_field, ""))
                path_root = _nonempty("path_root", record.get("path_root", ""))
                if path_root not in PATH_ROOTS:
                    raise ValueError(f"unsupported {field} path_root {path_root!r}")
                _nonempty("path", record.get("path", ""))
                records.append(MappingProxyType(record))
            ids = [str(item[id_field]) for item in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{field} IDs must be unique")
            return tuple(records)

        dependencies = path_records("dependencies", "dependency_id")
        source_assets = path_records("source_assets", "asset_set_id")
        return cls(
            _nonempty("reference_id", value.get("reference_id", "")),
            _nonempty("source_material_family_id", value.get("source_material_family_id", "")),
            role,
            _nonempty("native_representation", value.get("native_representation", "")),
            MappingProxyType(dict(_object("query_contract", value.get("query_contract", {})))),
            implementations,
            dependencies,
            source_assets,
        )


@dataclass(frozen=True)
class ReferenceRegistryEntry:
    reference_id: str
    package: str
    role: str
    source_material_family_id: str
    status: str
    capabilities: Mapping[str, str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceRegistryEntry:
        role = _nonempty("role", value.get("role", ""))
        if role not in ROLES:
            raise ValueError(f"unsupported reference role {role!r}")
        status = _nonempty("status", value.get("status", ""))
        if status not in {"planned", "active", "retired"}:
            raise ValueError(f"unsupported reference status {status!r}")
        capabilities_value = _object("capabilities", value.get("capabilities", {}))
        if set(capabilities_value) != CAPABILITY_NAMES:
            missing = sorted(CAPABILITY_NAMES - set(capabilities_value))
            extra = sorted(set(capabilities_value) - CAPABILITY_NAMES)
            raise ValueError(f"reference capabilities mismatch: missing={missing}, extra={extra}")
        capabilities = {str(name): str(state) for name, state in capabilities_value.items()}
        invalid = {name: state for name, state in capabilities.items() if state not in CAPABILITY_STATES}
        if invalid:
            raise ValueError(f"unsupported reference capability states: {invalid}")
        if capabilities["viewer_integration"] == "ready" and capabilities["falcor_runtime"] != "ready":
            raise ValueError("viewer integration requires a ready Falcor runtime")
        if capabilities["image_parity"] == "ready" and capabilities["viewer_integration"] != "ready":
            raise ValueError("image parity requires ready viewer integration")
        if capabilities["numerical_parity"] == "ready" and capabilities["native_reference"] != "ready":
            raise ValueError("numerical parity requires a ready native reference")
        return cls(
            _nonempty("reference_id", value.get("reference_id", "")),
            _nonempty("package", value.get("package", "")),
            role,
            _nonempty("source_material_family_id", value.get("source_material_family_id", "")),
            status,
            MappingProxyType(capabilities),
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
    return _object(str(path), value)


def load_reference_package(path: str | Path) -> ReferencePackage:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "reference.json"
    return ReferencePackage.from_dict(_load_json(manifest_path))


def load_reference_registry(path: str | Path) -> tuple[ReferenceRegistryEntry, ...]:
    registry_path = Path(path)
    if registry_path.is_dir():
        registry_path = registry_path / "registry.json"
    value = _load_json(registry_path)
    if value.get("schema_name") != REFERENCE_REGISTRY_SCHEMA:
        raise ValueError("unsupported reference registry schema_name")
    if value.get("schema_version") != REFERENCE_REGISTRY_VERSION:
        raise ValueError("unsupported reference registry schema_version")
    entries = tuple(
        ReferenceRegistryEntry.from_dict(_object("reference registry entry", item))
        for item in value.get("references", [])
    )
    ids = [item.reference_id for item in entries]
    packages = [item.package for item in entries]
    if len(ids) != len(set(ids)) or len(packages) != len(set(packages)):
        raise ValueError("reference registry IDs and package paths must be unique")
    return entries


def validate_reference_tree(reference_root: str | Path) -> tuple[ReferencePackage, ...]:
    root = Path(reference_root)
    entries = load_reference_registry(root)
    packages: list[ReferencePackage] = []
    for entry in entries:
        package_root = root / entry.package
        if not package_root.is_dir():
            raise ValueError(f"reference package directory does not exist: {package_root}")
        package = load_reference_package(package_root)
        if package.reference_id != entry.reference_id:
            raise ValueError(f"reference_id mismatch for package {entry.package}")
        if package.role != entry.role:
            raise ValueError(f"role mismatch for package {entry.package}")
        if package.source_material_family_id != entry.source_material_family_id:
            raise ValueError(f"source material family mismatch for package {entry.package}")
        packages.append(package)
    return tuple(packages)
