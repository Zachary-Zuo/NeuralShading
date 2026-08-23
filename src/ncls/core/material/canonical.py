from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .layer_stack import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerInterfaceIR,
    LayerStackIR,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
    pack_layer_stack,
)
from .program import (
    OUTPUT_NAMES,
    MaterialNode,
    MaterialProgram,
    NodeConnection,
    ParameterSource,
    ValueType,
)
from .registry import (
    DIFFUSE,
    HOMOGENEOUS_MEDIUM,
    LAYER_STACK,
    OPERATION_REGISTRY,
    ROUGH_CONDUCTOR,
    ROUGH_DIELECTRIC,
    SHEEN,
)


_OUTPUT_TYPES = {
    "surface": ValueType.SURFACE,
    "interior_medium": ValueType.MEDIUM,
    "exterior_medium": ValueType.MEDIUM,
    "emission": ValueType.EMISSION,
    "opacity": ValueType.OPACITY,
    "displacement": ValueType.DISPLACEMENT,
}


def _as_connections(value: NodeConnection | tuple[NodeConnection, ...]) -> tuple[NodeConnection, ...]:
    return (value,) if isinstance(value, NodeConnection) else tuple(value)


def _validate_constant(source: ParameterSource, name: str) -> None:
    if source.source != "constant":
        return
    value = source.value
    if source.value_type == ValueType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"parameter {name} must be a finite Float")
        return
    if source.value_type in {ValueType.FLOAT2, ValueType.FLOAT3, ValueType.COLOR3, ValueType.NORMAL3}:
        expected = 2 if source.value_type == ValueType.FLOAT2 else 3
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != expected:
            raise ValueError(f"parameter {name} must contain {expected} values")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value):
            raise ValueError(f"parameter {name} must contain finite numeric values")
        return
    raise ValueError(f"constant parameter type {source.value_type.value} is not supported in MaterialProgram v1")


def _connection_type(connection: NodeConnection, nodes: Mapping[str, MaterialNode]) -> ValueType:
    target = nodes.get(connection.node)
    if target is None:
        raise ValueError(f"connection references missing node {connection.node!r}")
    spec = OPERATION_REGISTRY.get(target.operation)
    if spec is None:
        raise ValueError(f"unsupported operation {target.operation.key}")
    try:
        return spec.outputs[connection.port]
    except KeyError as exc:
        raise ValueError(f"node {connection.node!r} has no output port {connection.port!r}") from exc


def _validate_acyclic(nodes: Mapping[str, MaterialNode]) -> None:
    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        marker = state.get(node_id, 0)
        if marker == 1:
            raise ValueError(f"MaterialProgram contains a cycle through node {node_id!r}")
        if marker == 2:
            return
        state[node_id] = 1
        for value in nodes[node_id].inputs.values():
            for connection in _as_connections(value):
                if connection.node not in nodes:
                    raise ValueError(f"connection references missing node {connection.node!r}")
                visit(connection.node)
        state[node_id] = 2

    for node_id in nodes:
        visit(node_id)


def validate_material_program(program: MaterialProgram) -> None:
    nodes: dict[str, MaterialNode] = {}
    for node in program.nodes:
        if node.node_id in nodes:
            raise ValueError(f"duplicate material node id {node.node_id!r}")
        nodes[node.node_id] = node

    resources = {resource.resource_id: resource for resource in program.resources}
    if len(resources) != len(program.resources):
        raise ValueError("duplicate material resource id")

    for node in program.nodes:
        spec = OPERATION_REGISTRY.get(node.operation)
        if spec is None:
            raise ValueError(f"unsupported operation {node.operation.key}")
        if set(node.inputs) != set(spec.inputs):
            raise ValueError(f"node {node.node_id!r} inputs do not match {node.operation.key}")
        if set(node.parameters) != set(spec.parameters):
            raise ValueError(f"node {node.node_id!r} parameters do not match {node.operation.key}")

        for name, input_spec in spec.inputs.items():
            raw = node.inputs[name]
            if input_spec.many == isinstance(raw, NodeConnection):
                expected = "an array" if input_spec.many else "one connection"
                raise ValueError(f"node {node.node_id!r} input {name!r} requires {expected}")
            connections = _as_connections(raw)
            if len(connections) < input_spec.min_count:
                raise ValueError(
                    f"node {node.node_id!r} input {name!r} requires at least {input_spec.min_count} connection(s)"
                )
            for connection in connections:
                actual_type = _connection_type(connection, nodes)
                if actual_type != input_spec.value_type:
                    raise ValueError(
                        f"node {node.node_id!r} input {name!r} expects {input_spec.value_type.value}, got {actual_type.value}"
                    )

        for name, parameter_spec in spec.parameters.items():
            source = node.parameters[name]
            if source.value_type != parameter_spec.value_type:
                raise ValueError(
                    f"node {node.node_id!r} parameter {name!r} expects {parameter_spec.value_type.value}, got {source.value_type.value}"
                )
            if source.source == "texture" and source.resource not in resources:
                raise ValueError(f"node {node.node_id!r} parameter {name!r} references a missing texture resource")
            _validate_constant(source, f"{node.node_id}.{name}")

    _validate_acyclic(nodes)
    for name in OUTPUT_NAMES:
        connection = program.outputs[name]
        if connection is None:
            continue
        actual_type = _connection_type(connection, nodes)
        if actual_type != _OUTPUT_TYPES[name]:
            raise ValueError(f"program output {name!r} expects {_OUTPUT_TYPES[name].value}, got {actual_type.value}")


def canonical_material_json(program: MaterialProgram) -> str:
    validate_material_program(program)
    document = program.to_dict(include_metadata=False, include_program_id=False)
    document["nodes"] = sorted(document["nodes"], key=lambda item: item["id"])
    document["resources"] = sorted(document["resources"], key=lambda item: item["id"])
    return json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def physical_material_hash(program: MaterialProgram) -> str:
    stack = canonicalize_layer_stack(program)
    digest = hashlib.sha256()
    digest.update(b"ncls.material-program\0v1\0linear-srgb\0")
    digest.update(pack_layer_stack(stack))
    return digest.hexdigest()


def _constant(node: MaterialNode, name: str) -> Any:
    source = node.parameters[name]
    if source.source != "constant":
        raise ValueError(f"LayerStackIR v1 does not support {source.source} parameter {node.node_id}.{name}")
    return source.value


def _rgb(value: Iterable[float]) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ValueError("expected three values")
    return result  # type: ignore[return-value]


def _interface_from_node(node: MaterialNode) -> LayerInterfaceIR:
    if node.operation == ROUGH_DIELECTRIC:
        return RoughDielectricInterface(
            float(_constant(node, "alpha_x")),
            float(_constant(node, "alpha_y")),
            float(_constant(node, "relative_ior")),
            float(_constant(node, "tangent_rotation")),
        )
    if node.operation == ROUGH_CONDUCTOR:
        return RoughConductorInterface(
            float(_constant(node, "alpha_x")),
            float(_constant(node, "alpha_y")),
            _rgb(_constant(node, "eta")),
            _rgb(_constant(node, "k")),
            float(_constant(node, "tangent_rotation")),
        )
    if node.operation == DIFFUSE:
        return DiffuseInterface(_rgb(_constant(node, "color")))
    if node.operation == SHEEN:
        return SheenInterface(_rgb(_constant(node, "color")), float(_constant(node, "roughness")))
    raise ValueError(f"node {node.node_id!r} is not a v1 interface")


def canonicalize_layer_stack(program: MaterialProgram) -> LayerStackIR:
    validate_material_program(program)
    nodes = {node.node_id: node for node in program.nodes}
    surface_connection = program.outputs["surface"]
    assert surface_connection is not None
    surface_node = nodes[surface_connection.node]
    if surface_node.operation != LAYER_STACK or surface_connection.port != "surface":
        raise ValueError("MaterialProgram surface cannot be canonicalized to LayerStackIR v1")

    interface_connections = _as_connections(surface_node.inputs["interfaces"])
    medium_connections = _as_connections(surface_node.inputs["media"])
    interfaces = tuple(_interface_from_node(nodes[connection.node]) for connection in interface_connections)
    media = tuple(
        HomogeneousMedium(
            _rgb(_constant(nodes[connection.node], "sigma_a")),
            _rgb(_constant(nodes[connection.node], "sigma_s")),
            float(_constant(nodes[connection.node], "g")),
            float(_constant(nodes[connection.node], "thickness")),
        )
        for connection in medium_connections
    )
    return LayerStackIR(interfaces, media)


def _float(value: float) -> ParameterSource:
    return ParameterSource.constant(ValueType.FLOAT, float(value))


def _color(value: Iterable[float]) -> ParameterSource:
    return ParameterSource.constant(ValueType.COLOR3, tuple(float(item) for item in value))


def _node_from_interface(index: int, interface: LayerInterfaceIR) -> MaterialNode:
    node_id = f"interface-{index:03d}"
    if isinstance(interface, RoughDielectricInterface):
        return MaterialNode(
            node_id,
            ROUGH_DIELECTRIC,
            parameters={
                "alpha_x": _float(interface.alpha_x),
                "alpha_y": _float(interface.alpha_y),
                "relative_ior": _float(interface.relative_ior),
                "tangent_rotation": _float(interface.tangent_rotation),
            },
        )
    if isinstance(interface, RoughConductorInterface):
        return MaterialNode(
            node_id,
            ROUGH_CONDUCTOR,
            parameters={
                "alpha_x": _float(interface.alpha_x),
                "alpha_y": _float(interface.alpha_y),
                "eta": _color(interface.eta),
                "k": _color(interface.k),
                "tangent_rotation": _float(interface.tangent_rotation),
            },
        )
    if isinstance(interface, DiffuseInterface):
        return MaterialNode(node_id, DIFFUSE, parameters={"color": _color(interface.color)})
    if isinstance(interface, SheenInterface):
        return MaterialNode(
            node_id,
            SHEEN,
            parameters={"color": _color(interface.color), "roughness": _float(interface.roughness)},
        )
    raise TypeError(f"unsupported interface {type(interface)!r}")


def material_program_from_layer_stack(stack: LayerStackIR, *, metadata: Mapping[str, Any] | None = None) -> MaterialProgram:
    interface_nodes = [_node_from_interface(index, interface) for index, interface in enumerate(stack.interfaces)]
    medium_nodes = [
        MaterialNode(
            f"medium-{index:03d}",
            HOMOGENEOUS_MEDIUM,
            parameters={
                "sigma_a": _color(medium.sigma_a),
                "sigma_s": _color(medium.sigma_s),
                "g": _float(medium.g),
                "thickness": _float(medium.thickness),
            },
        )
        for index, medium in enumerate(stack.media)
    ]
    stack_node = MaterialNode(
        "surface",
        LAYER_STACK,
        inputs={
            "interfaces": tuple(NodeConnection(node.node_id, "interface") for node in interface_nodes),
            "media": tuple(NodeConnection(node.node_id, "medium") for node in medium_nodes),
        },
    )
    outputs = {name: None for name in OUTPUT_NAMES}
    outputs["surface"] = NodeConnection(stack_node.node_id, "surface")
    program = MaterialProgram(tuple(interface_nodes + medium_nodes + [stack_node]), outputs, metadata=metadata or {})
    validate_material_program(program)
    return program
