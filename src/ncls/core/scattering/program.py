from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
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
    def from_path(
        cls, path: Path, *, content_sha256: str | None = None
    ) -> "FileResourcePayload":
        resolved = path.resolve()
        digest = sha256_file(resolved) if content_sha256 is None else str(content_sha256)
        return cls(resolved, digest, resolved.stat().st_size)

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


def validate_typed_parameter_view(value: Mapping[str, Any]) -> None:
    """Validate the complete generic editor ABI before any runtime sees it."""

    view = dict(value)
    required = {
        "schema_name", "schema_version", "family_id", "source_contract_version",
        "snapshot_id", "root", "runtime_layout",
    }
    if set(view) != required or view.get("schema_name") != "ncls.source-parameter-view":
        raise ValueError("typed material editor parameter_view fields are invalid")
    if int(view["schema_version"]) != 1 or int(view["source_contract_version"]) < 1:
        raise ValueError("typed material editor parameter_view version is invalid")
    if not isinstance(view["family_id"], str) or not view["family_id"]:
        raise ValueError("typed material editor family_id is invalid")
    require_sha256("typed material editor snapshot_id", str(view["snapshot_id"]))
    layout = dict(view["runtime_layout"])
    if (
        set(layout) != {"schema", "word_count", "offsets"}
        or not str(layout["schema"])
        or int(layout["word_count"]) < 1
        or not isinstance(layout["offsets"], Mapping)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or offset >= int(layout["word_count"])
            for name, offset in layout["offsets"].items()
        )
    ):
        raise ValueError("typed material editor runtime_layout is invalid")
    word_count = int(layout["word_count"])
    paths: set[str] = set()

    def finite_number(item: Any) -> bool:
        return isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))

    def validate_value(node: Mapping[str, Any], item: Any) -> None:
        value_type = str(node["value_type"])
        components = {"vector2": 2, "vector3": 3, "color3": 3, "vector4": 4}.get(value_type)
        if value_type == "bool" and not isinstance(item, bool):
            raise ValueError("typed material editor bool value is invalid")
        if value_type == "int" and (not isinstance(item, int) or isinstance(item, bool)):
            raise ValueError("typed material editor int value is invalid")
        if value_type in {"float", "double"} and not finite_number(item):
            raise ValueError("typed material editor scalar value is invalid")
        if components is not None and (
            not isinstance(item, (list, tuple)) or len(item) != components
            or not all(finite_number(component) for component in item)
        ):
            raise ValueError("typed material editor vector value is invalid")
        if value_type == "enum":
            choices = node.get("choices")
            if (
                not isinstance(item, str)
                or not isinstance(choices, (list, tuple))
                or not choices
                or any(not isinstance(choice, str) or not choice for choice in choices)
                or len(set(choices)) != len(choices)
                or item not in choices
            ):
                raise ValueError("typed material editor enum value is invalid")
        if value_type not in {"float", "double", "int", "bool", "enum", "vector2", "vector3", "vector4", "color3"}:
            raise ValueError("typed material editor value_type is unsupported")

    def visit(raw_node: Any) -> None:
        if not isinstance(raw_node, Mapping):
            raise ValueError("typed material editor node must be an object")
        node = dict(raw_node)
        for field in ("path", "kind", "label", "children", "editable", "allowed_operations"):
            if field not in node:
                raise ValueError("typed material editor node fields are incomplete")
        path = node["path"]
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path in paths
            or not isinstance(node["label"], str)
            or not node["label"]
            or not isinstance(node["kind"], str)
            or not node["kind"]
            or not isinstance(node["editable"], bool)
            or not isinstance(node["allowed_operations"], (list, tuple))
            or any(
                not isinstance(operation, str)
                for operation in node["allowed_operations"]
            )
        ):
            raise ValueError("typed material editor node path/label is invalid")
        paths.add(path)
        children = node["children"]
        if not isinstance(children, (list, tuple)):
            raise ValueError("typed material editor node children are invalid")
        if bool(node["editable"]):
            if (
                node["kind"] != "value"
                or list(node["allowed_operations"]) != ["set"]
                or children
            ):
                raise ValueError("editable typed material node operations are invalid")
            if "value" not in node or "value_type" not in node:
                raise ValueError("editable typed material node has no value/type")
            validate_value(node, node["value"])
            metadata = node.get("metadata")
            if not isinstance(metadata, Mapping) or not isinstance(metadata.get("runtime"), Mapping):
                raise ValueError("editable typed material node has no runtime mapping")
            runtime = dict(metadata["runtime"])
            if set(runtime) != {
                "token_index", "continuous_word", "discrete_word", "type_word",
                "normalization", "derived_writes",
            }:
                raise ValueError("typed material node runtime fields are invalid")
            if (
                int(runtime["token_index"]) < 0
                or int(runtime["continuous_word"]) < 0
                or int(runtime["continuous_word"]) + 4 > word_count
                or int(runtime["discrete_word"]) not in range(word_count)
                or int(runtime["type_word"]) not in range(word_count)
            ):
                raise ValueError("typed material node runtime words are out of bounds")
            normalization = dict(runtime["normalization"])
            if not set(normalization).issubset({"default", "minimum", "maximum"}) or "default" not in normalization:
                raise ValueError("typed material node normalization is invalid")
            validate_value(node, normalization["default"])
            if ("minimum" in normalization) != ("maximum" in normalization):
                raise ValueError("typed material node normalization range is incomplete")
            if "minimum" in normalization and (
                not finite_number(normalization["minimum"])
                or not finite_number(normalization["maximum"])
                or float(normalization["minimum"]) >= float(normalization["maximum"])
            ):
                raise ValueError("typed material node normalization range is invalid")
            writes = runtime["derived_writes"]
            if not isinstance(writes, (list, tuple)):
                raise ValueError("typed material node derived_writes are invalid")
            for write in writes:
                if (
                    not isinstance(write, Mapping)
                    or set(write) != {"word", "operation", "component"}
                    or int(write["word"]) not in range(word_count)
                    or write["operation"] not in {"copy", "bool", "degrees-cos", "degrees-sin"}
                    or int(write["component"]) not in range(4)
                ):
                    raise ValueError("typed material node derived write is invalid")
        for child in children:
            visit(child)

    visit(view["root"])
    if str(dict(view["root"])["path"]) != "/":
        raise ValueError("typed material editor root path must be /")


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
class InstancePayload:
    """Per-package editable state, separate from immutable program and asset data."""

    parameters: Mapping[str, Any]
    blobs: Mapping[str, bytes] = field(default_factory=dict)
    blob_descriptors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    editor: Mapping[str, Any] = field(default_factory=dict)
    compiler: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parameters = dict(self.parameters)
        if set(parameters) != {"compiled_material_index"}:
            raise ValueError("instance parameters require exactly compiled_material_index")
        if int(parameters["compiled_material_index"]) < 0:
            raise ValueError("compiled_material_index must be nonnegative")
        if set(self.blobs) != set(self.blob_descriptors):
            raise ValueError("instance blob descriptors must cover blobs exactly")
        editor = dict(self.editor)
        compiler = dict(self.compiler)
        if bool(editor) != bool(compiler):
            raise ValueError("editable instances require both editor and compiler contracts")
        if editor:
            required_editor = {"schema", "parameter_view", "raw_usage", "compiled_usage"}
            required_compiler = {"entry_point", "thread_group_size"}
            if set(editor) != required_editor or set(compiler) != required_compiler:
                raise ValueError("instance editor/compiler contract fields are invalid")
            if editor["schema"] != "ncls.typed-material-editor@1":
                raise ValueError("unsupported typed material editor schema")
            if not isinstance(editor["parameter_view"], Mapping):
                raise ValueError("typed material editor parameter_view must be an object")
            validate_typed_parameter_view(editor["parameter_view"])
            if not str(editor["raw_usage"]) or not str(editor["compiled_usage"]):
                raise ValueError("typed material editor usages must be nonempty")
            group = compiler["thread_group_size"]
            if (
                not str(compiler["entry_point"])
                or not isinstance(group, (tuple, list))
                or len(group) != 3
                or any(int(value) < 1 for value in group)
            ):
                raise ValueError("typed material compiler entry/group is invalid")
            usages = {
                str(descriptor.get("usage")): descriptor
                for descriptor in self.blob_descriptors.values()
            }
            if editor["raw_usage"] not in usages or editor["compiled_usage"] not in usages:
                raise ValueError("typed material editor usages must name instance blobs")
            if any(
                usages[usage].get("kind") != "mutable-structured-buffer"
                for usage in (editor["raw_usage"], editor["compiled_usage"])
            ):
                raise ValueError("typed material editor buffers must be mutable structured buffers")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "blobs", dict(self.blobs))
        object.__setattr__(
            self,
            "blob_descriptors",
            {name: dict(value) for name, value in self.blob_descriptors.items()},
        )
        object.__setattr__(self, "editor", editor)
        object.__setattr__(self, "compiler", compiler)


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
