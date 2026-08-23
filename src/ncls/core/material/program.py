from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Mapping


MATERIAL_PROGRAM_SCHEMA = "ncls.material-program"
MATERIAL_PROGRAM_VERSION = 1
COLOR_MODEL_V1 = "linear-srgb"
OUTPUT_NAMES = (
    "surface",
    "interior_medium",
    "exterior_medium",
    "emission",
    "opacity",
    "displacement",
)


class ValueType(str, Enum):
    FLOAT = "Float"
    FLOAT2 = "Float2"
    FLOAT3 = "Float3"
    COLOR3 = "Color3"
    NORMAL3 = "Normal3"
    SPECTRUM = "Spectrum"
    INTERFACE = "Interface"
    MEDIUM = "Medium"
    SURFACE = "Surface"
    EMISSION = "Emission"
    OPACITY = "Opacity"
    DISPLACEMENT = "Displacement"


@dataclass(frozen=True, order=True)
class OperationId:
    namespace: str
    name: str
    version: int

    def __post_init__(self) -> None:
        if not self.namespace or not self.name:
            raise ValueError("operation namespace and name must be nonempty")
        if self.version < 1:
            raise ValueError("operation version must be positive")

    @property
    def key(self) -> str:
        return f"{self.namespace}.{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {"namespace": self.namespace, "name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperationId:
        return cls(str(value["namespace"]), str(value["name"]), int(value["version"]))


@dataclass(frozen=True)
class NodeConnection:
    node: str
    port: str

    def __post_init__(self) -> None:
        if not self.node or not self.port:
            raise ValueError("node connection requires nonempty node and port")

    def to_dict(self) -> dict[str, str]:
        return {"node": self.node, "port": self.port}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NodeConnection:
        return cls(str(value["node"]), str(value["port"]))


@dataclass(frozen=True)
class ParameterSource:
    source: str
    value_type: ValueType
    value: Any = None
    resource: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_type", ValueType(self.value_type))
        if self.source not in {"constant", "texture", "vertex_attribute", "procedural"}:
            raise ValueError(f"unsupported parameter source {self.source!r}")
        if self.source == "constant" and self.resource is not None:
            raise ValueError("constant parameter cannot reference a resource")
        if self.source != "constant" and self.resource is None:
            raise ValueError(f"{self.source} parameter requires a resource or symbol")

    @classmethod
    def constant(cls, value_type: ValueType, value: Any) -> ParameterSource:
        return cls("constant", value_type, value=value)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source": self.source, "type": self.value_type.value}
        if self.source == "constant":
            result["value"] = _json_value(self.value)
        else:
            result["resource"] = self.resource
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParameterSource:
        source = str(value["source"])
        return cls(
            source,
            ValueType(value["type"]),
            value=value.get("value"),
            resource=str(value["resource"]) if value.get("resource") is not None else None,
        )


NodeInput = NodeConnection | tuple[NodeConnection, ...]


def _freeze_input(value: NodeInput) -> NodeInput:
    if isinstance(value, NodeConnection):
        return value
    return tuple(value)


@dataclass(frozen=True)
class MaterialNode:
    node_id: str
    operation: OperationId
    inputs: Mapping[str, NodeInput] = field(default_factory=dict)
    parameters: Mapping[str, ParameterSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("material node id must be nonempty")
        object.__setattr__(self, "inputs", MappingProxyType({str(k): _freeze_input(v) for k, v in self.inputs.items()}))
        object.__setattr__(self, "parameters", MappingProxyType({str(k): v for k, v in self.parameters.items()}))

    def to_dict(self) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for name, value in self.inputs.items():
            inputs[name] = value.to_dict() if isinstance(value, NodeConnection) else [item.to_dict() for item in value]
        return {
            "id": self.node_id,
            "operation": self.operation.to_dict(),
            "inputs": inputs,
            "parameters": {name: source.to_dict() for name, source in self.parameters.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterialNode:
        inputs: dict[str, NodeInput] = {}
        for name, raw in value.get("inputs", {}).items():
            if isinstance(raw, list):
                inputs[str(name)] = tuple(NodeConnection.from_dict(item) for item in raw)
            else:
                inputs[str(name)] = NodeConnection.from_dict(raw)
        return cls(
            str(value["id"]),
            OperationId.from_dict(value["operation"]),
            inputs,
            {str(name): ParameterSource.from_dict(raw) for name, raw in value.get("parameters", {}).items()},
        )


@dataclass(frozen=True)
class MaterialResource:
    resource_id: str
    uri: str
    semantic: str
    color_space: str | None = None

    def __post_init__(self) -> None:
        if not self.resource_id or not self.uri or not self.semantic:
            raise ValueError("material resource id, uri and semantic must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.resource_id, "uri": self.uri, "semantic": self.semantic}
        if self.color_space is not None:
            result["color_space"] = self.color_space
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterialResource:
        return cls(
            str(value["id"]),
            str(value["uri"]),
            str(value["semantic"]),
            str(value["color_space"]) if value.get("color_space") is not None else None,
        )


@dataclass(frozen=True)
class MaterialProgram:
    nodes: tuple[MaterialNode, ...]
    outputs: Mapping[str, NodeConnection | None]
    resources: tuple[MaterialResource, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    color_model: str = COLOR_MODEL_V1
    schema_name: str = MATERIAL_PROGRAM_SCHEMA
    schema_version: int = MATERIAL_PROGRAM_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "outputs", MappingProxyType({str(k): v for k, v in self.outputs.items()}))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.schema_name != MATERIAL_PROGRAM_SCHEMA or self.schema_version != MATERIAL_PROGRAM_VERSION:
            raise ValueError("unsupported MaterialProgram schema")
        if self.color_model != COLOR_MODEL_V1:
            raise ValueError(f"unsupported v1 color model {self.color_model!r}")
        if set(self.outputs) != set(OUTPUT_NAMES):
            raise ValueError(f"MaterialProgram outputs must be exactly {OUTPUT_NAMES}")
        if self.outputs["surface"] is None:
            raise ValueError("MaterialProgram v1 requires a surface output")

    def to_dict(self, *, include_metadata: bool = True, include_program_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "color_model": self.color_model,
            "nodes": [node.to_dict() for node in self.nodes],
            "resources": [resource.to_dict() for resource in self.resources],
            "outputs": {name: connection.to_dict() if connection is not None else None for name, connection in self.outputs.items()},
        }
        if include_metadata:
            result["metadata"] = _json_value(dict(self.metadata))
        if include_program_id:
            from .canonical import physical_material_hash

            result["program_id"] = physical_material_hash(self)
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterialProgram:
        raw_outputs = value.get("outputs", {})
        outputs = {
            name: NodeConnection.from_dict(raw_outputs[name]) if raw_outputs.get(name) is not None else None
            for name in OUTPUT_NAMES
        }
        program = cls(
            tuple(MaterialNode.from_dict(item) for item in value.get("nodes", [])),
            outputs,
            tuple(MaterialResource.from_dict(item) for item in value.get("resources", [])),
            value.get("metadata", {}),
            str(value.get("color_model", COLOR_MODEL_V1)),
            str(value.get("schema_name", "")),
            int(value.get("schema_version", -1)),
        )
        from .canonical import physical_material_hash, validate_material_program

        validate_material_program(program)
        declared_id = value.get("program_id")
        if declared_id is not None:
            if str(declared_id) != physical_material_hash(program):
                raise ValueError("MaterialProgram program_id does not match physical content")
        return program

    @classmethod
    def from_json(cls, text: str) -> MaterialProgram:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("MaterialProgram JSON root must be an object")
        return cls.from_dict(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
