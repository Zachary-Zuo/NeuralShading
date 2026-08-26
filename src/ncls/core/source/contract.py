from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ncls.core.identity import require_sha256, sha256_bytes, sha256_json


ParameterKind = Literal["group", "list", "variant", "value", "resource", "read-only"]
ParameterValueType = Literal[
    "float", "bool", "enum", "vector2", "vector3", "color3", "string", "resource"
]
EditKind = Literal["set", "insert", "remove", "move", "replace-variant"]
BindingKind = Literal[
    "constant", "texture", "graph", "geometry", "measurement", "derived", "structural"
]


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    return value


@dataclass(frozen=True)
class SourceFamilyDescriptor:
    family_id: str
    source_contract_version: int
    native_schema_id: str
    reference_program_id: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        if not self.family_id or not self.native_schema_id or not self.reference_program_id:
            raise ValueError("source family descriptor identity fields must be nonempty")
        if self.source_contract_version < 1:
            raise ValueError("source contract version must be positive")
        require_sha256("source family implementation_sha256", self.implementation_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "native_schema_id": self.native_schema_id,
            "reference_program_id": self.reference_program_id,
            "implementation_sha256": self.implementation_sha256,
        }


@dataclass(frozen=True)
class SourceSnapshot:
    family_id: str
    source_contract_version: int
    native_schema_id: str
    source_asset_sha256: str
    native_payload: bytes
    resource_hashes: Mapping[str, str] = field(default_factory=dict)
    editor_metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    native_object: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.family_id or not self.native_schema_id:
            raise ValueError("source snapshot identity fields must be nonempty")
        if self.source_contract_version < 1:
            raise ValueError("source snapshot contract version must be positive")
        require_sha256("source_asset_sha256", self.source_asset_sha256)
        if not isinstance(self.native_payload, bytes) or not self.native_payload:
            raise ValueError("source snapshot native payload must be nonempty bytes")
        resources = {str(uri): require_sha256(f"resource hash {uri}", str(digest)) for uri, digest in self.resource_hashes.items()}
        if any(not uri for uri in resources):
            raise ValueError("source snapshot resource URIs must be nonempty")
        object.__setattr__(self, "resource_hashes", resources)
        object.__setattr__(self, "editor_metadata", dict(self.editor_metadata))

    @property
    def snapshot_id(self) -> str:
        return sha256_json(
            {
                "family_id": self.family_id,
                "source_contract_version": self.source_contract_version,
                "native_schema_id": self.native_schema_id,
                "source_asset_sha256": self.source_asset_sha256,
                "native_payload_sha256": sha256_bytes(self.native_payload),
                "resource_hashes": dict(self.resource_hashes),
            }
        )

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "native_schema_id": self.native_schema_id,
            "source_asset_sha256": self.source_asset_sha256,
            "native_payload_sha256": sha256_bytes(self.native_payload),
            "resource_hashes": dict(self.resource_hashes),
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True)
class ParameterNode:
    path: str
    kind: ParameterKind
    label: str
    children: tuple["ParameterNode", ...] = ()
    element_id: str | None = None
    value_type: ParameterValueType | None = None
    value: Any = None
    default: Any = None
    choices: tuple[str, ...] = ()
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    ui_hint: str | None = None
    binding: BindingKind | None = None
    editable: bool = False
    read_only_reason: str | None = None
    allowed_operations: tuple[EditKind, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path.startswith("/") or not self.label:
            raise ValueError("parameter node path must be absolute and label nonempty")
        if self.kind not in {"group", "list", "variant", "value", "resource", "read-only"}:
            raise ValueError("unsupported parameter node kind")
        children = tuple(self.children)
        child_paths = [child.path for child in children]
        if len(set(child_paths)) != len(child_paths):
            raise ValueError("parameter node children must have unique paths")
        if self.kind in {"value", "resource"} and self.value_type is None:
            raise ValueError("value and resource parameter nodes require value_type")
        if self.kind in {"group", "list", "variant"} and self.value_type is not None:
            raise ValueError("container parameter nodes cannot declare value_type")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter range minimum must not exceed maximum")
        if self.step is not None and self.step <= 0.0:
            raise ValueError("parameter step must be positive")
        if self.value_type == "enum" and not self.choices:
            raise ValueError("enum parameter nodes require choices")
        if self.editable and self.read_only_reason:
            raise ValueError("editable parameter nodes cannot have a read-only reason")
        if not self.editable and self.allowed_operations:
            raise ValueError("read-only parameter nodes cannot allow edit operations")
        if len(set(self.allowed_operations)) != len(self.allowed_operations):
            raise ValueError("parameter operations must be unique")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "choices", tuple(self.choices))
        object.__setattr__(self, "allowed_operations", tuple(self.allowed_operations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "label": self.label,
            "children": [child.to_dict() for child in self.children],
            "editable": self.editable,
            "allowed_operations": list(self.allowed_operations),
        }
        for name in (
            "element_id", "value_type", "unit", "minimum", "maximum", "step",
            "ui_hint", "binding", "read_only_reason",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.value is not None:
            result["value"] = _plain(self.value)
        if self.default is not None:
            result["default"] = _plain(self.default)
        if self.choices:
            result["choices"] = list(self.choices)
        if self.metadata:
            result["metadata"] = _plain(self.metadata)
        return result


@dataclass(frozen=True)
class SourceParameterView:
    family_id: str
    source_contract_version: int
    snapshot_id: str
    root: ParameterNode
    schema_name: str = "ncls.source-parameter-view"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.source-parameter-view" or self.schema_version != 1:
            raise ValueError("unsupported SourceParameterView schema")
        if not self.family_id or self.source_contract_version < 1:
            raise ValueError("SourceParameterView family identity is invalid")
        require_sha256("SourceParameterView snapshot_id", self.snapshot_id)
        if self.root.path != "/":
            raise ValueError("SourceParameterView root path must be /")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "snapshot_id": self.snapshot_id,
            "root": self.root.to_dict(),
        }


@dataclass(frozen=True)
class SourceEditOperation:
    operation: EditKind
    target: str
    value: Any = None
    element_id: str | None = None
    destination: int | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in {"set", "insert", "remove", "move", "replace-variant"}:
            raise ValueError("unsupported source edit operation")
        if not self.target.startswith("/"):
            raise ValueError("source edit target must be an absolute parameter path")
        if self.operation == "set" and self.value is None:
            raise ValueError("set operation requires a typed value")
        if self.operation == "insert" and (self.value is None or not self.element_id):
            raise ValueError("insert operation requires value and stable element_id")
        if self.operation in {"remove", "move"} and not self.element_id:
            raise ValueError(f"{self.operation} operation requires stable element_id")
        if self.operation == "move" and (self.destination is None or self.destination < 0):
            raise ValueError("move operation requires nonnegative destination")
        if self.operation == "replace-variant" and not self.variant:
            raise ValueError("replace-variant operation requires variant")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"operation": self.operation, "target": self.target}
        if self.value is not None:
            result["value"] = _plain(self.value)
        if self.element_id is not None:
            result["element_id"] = self.element_id
        if self.destination is not None:
            result["destination"] = self.destination
        if self.variant is not None:
            result["variant"] = self.variant
        return result


@dataclass(frozen=True)
class SourceEditPatch:
    base_snapshot_id: str
    operations: tuple[SourceEditOperation, ...]
    schema_name: str = "ncls.source-edit-patch"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_name != "ncls.source-edit-patch" or self.schema_version != 1:
            raise ValueError("unsupported SourceEditPatch schema")
        require_sha256("SourceEditPatch base_snapshot_id", self.base_snapshot_id)
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("SourceEditPatch requires at least one operation")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "base_snapshot_id": self.base_snapshot_id,
            "operations": [operation.to_dict() for operation in self.operations],
        }


@dataclass(frozen=True)
class SourceEditResult:
    snapshot: SourceSnapshot
    changed_paths: tuple[str, ...]
    invalidation: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        changed = tuple(dict.fromkeys(self.changed_paths))
        invalidation = tuple(dict.fromkeys(self.invalidation))
        if not changed or any(not path.startswith("/") for path in changed):
            raise ValueError("source edit result requires absolute changed paths")
        if any(not value for value in invalidation):
            raise ValueError("source edit invalidation identities must be nonempty")
        object.__setattr__(self, "changed_paths", changed)
        object.__setattr__(self, "invalidation", invalidation)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


class SourceFamilyDefinition(ABC):
    descriptor: SourceFamilyDescriptor

    @abstractmethod
    def describe_parameters(self, snapshot: SourceSnapshot) -> SourceParameterView:
        raise NotImplementedError

    @abstractmethod
    def apply_edit(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> SourceEditResult:
        raise NotImplementedError

    def validate_snapshot(self, snapshot: SourceSnapshot) -> None:
        descriptor = self.descriptor
        if (
            snapshot.family_id != descriptor.family_id
            or snapshot.source_contract_version != descriptor.source_contract_version
            or snapshot.native_schema_id != descriptor.native_schema_id
        ):
            raise ValueError("source snapshot does not match source family descriptor")

    def validate_patch(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> None:
        self.validate_snapshot(snapshot)
        if patch.base_snapshot_id != snapshot.snapshot_id:
            raise ValueError("source edit patch base snapshot is stale")
