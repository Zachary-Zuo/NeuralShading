from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_file, sha256_json
from ncls.core.material import (
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    MaterialProgram,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
    canonicalize_layer_stack,
    material_program_from_layer_stack,
)
from ncls.core.source import (
    ParameterNode,
    SourceEditPatch,
    SourceEditResult,
    SourceFamilyDefinition,
    SourceFamilyDescriptor,
    SourceParameterView,
    SourceSnapshot,
)


_INTERFACE_VARIANTS = ("rough-dielectric", "rough-conductor", "diffuse", "sheen")


def _interface_variant(value: object) -> str:
    if isinstance(value, RoughDielectricInterface):
        return "rough-dielectric"
    if isinstance(value, RoughConductorInterface):
        return "rough-conductor"
    if isinstance(value, DiffuseInterface):
        return "diffuse"
    if isinstance(value, SheenInterface):
        return "sheen"
    raise TypeError(f"unsupported LayerStack interface {type(value)!r}")


def _interface_from_dict(value: Mapping[str, Any]) -> object:
    variant = str(value.get("variant", ""))
    if variant == "rough-dielectric":
        return RoughDielectricInterface(
            float(value.get("alpha_x", 0.2)),
            float(value.get("alpha_y", 0.2)),
            float(value.get("relative_ior", 1.5)),
            float(value.get("tangent_rotation", 0.0)),
        )
    if variant == "rough-conductor":
        return RoughConductorInterface(
            float(value.get("alpha_x", 0.2)),
            float(value.get("alpha_y", 0.2)),
            tuple(value.get("eta", (0.2, 0.9, 1.1))),
            tuple(value.get("k", (3.9, 2.5, 2.4))),
            float(value.get("tangent_rotation", 0.0)),
        )
    if variant == "diffuse":
        return DiffuseInterface(tuple(value.get("color", (0.5, 0.5, 0.5))))
    if variant == "sheen":
        return SheenInterface(
            tuple(value.get("color", (0.5, 0.5, 0.5))),
            float(value.get("roughness", 0.5)),
        )
    raise ValueError(f"unsupported LayerStack interface variant {variant!r}")


def _medium_from_dict(value: Mapping[str, Any]) -> HomogeneousMedium:
    return HomogeneousMedium(
        tuple(value.get("sigma_a", (0.0, 0.0, 0.0))),
        tuple(value.get("sigma_s", (0.0, 0.0, 0.0))),
        float(value.get("g", 0.0)),
        float(value.get("thickness", 1.0)),
    )


def _field_node(path: str, label: str, value: Any) -> ParameterNode:
    if isinstance(value, tuple):
        value_type = "color3" if len(value) == 3 else "vector2"
    else:
        value_type = "float"
    limits: dict[str, float] = {}
    name = path.rsplit("/", 1)[-1]
    if name in {"alpha_x", "alpha_y", "roughness"}:
        limits = {"minimum": 0.0, "maximum": 1.0, "step": 0.01}
    elif name == "g":
        limits = {"minimum": -0.999, "maximum": 0.999, "step": 0.01}
    elif name in {"relative_ior", "thickness"}:
        limits = {"minimum": 0.0, "step": 0.01}
    return ParameterNode(
        path,
        "value",
        label,
        value_type=value_type,
        value=value,
        binding="constant",
        editable=True,
        allowed_operations=("set",),
        **limits,
    )


def _interface_node(index: int, count: int, element_id: str, value: object) -> ParameterNode:
    variant = _interface_variant(value)
    fields = asdict(value)
    children = tuple(
        _field_node(f"/interfaces/{element_id}/{name}", name.replace("_", " ").title(), field_value)
        for name, field_value in fields.items()
    )
    return ParameterNode(
        f"/interfaces/{element_id}",
        "variant",
        f"Interface {index + 1}",
        children,
        element_id=element_id,
        value=variant,
        choices=_INTERFACE_VARIANTS,
        binding="structural",
        editable=True,
        allowed_operations=("remove", "move", "replace-variant") if index < count - 1 else ("replace-variant",),
        metadata={"variant": variant, "index": index},
    )


def _medium_node(index: int, element_id: str, value: HomogeneousMedium) -> ParameterNode:
    children = tuple(
        _field_node(f"/media/{element_id}/{name}", name.replace("_", " ").title(), field_value)
        for name, field_value in asdict(value).items()
    )
    return ParameterNode(
        f"/media/{element_id}",
        "group",
        f"Medium {index + 1}",
        children,
        element_id=element_id,
        binding="structural",
        editable=True,
        allowed_operations=("remove", "move"),
    )


def snapshot_from_layer_stack(
    stack: LayerStackIR,
    *,
    source_asset_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceSnapshot:
    program = material_program_from_layer_stack(stack, metadata=dict(metadata or {}))
    payload = program.to_json().encode("utf-8")
    interface_ids = tuple(f"interface-{index}-{sha256_json(asdict(value))[:12]}" for index, value in enumerate(stack.interfaces))
    medium_ids = tuple(f"medium-{index}-{sha256_json(asdict(value))[:12]}" for index, value in enumerate(stack.media))
    return SourceSnapshot(
        "ncls.layer-stack@1",
        1,
        "ncls.material-program@1",
        source_asset_sha256 or sha256_json({"native_payload": payload.decode("utf-8")}),
        payload,
        editor_metadata={"interface_ids": interface_ids, "medium_ids": medium_ids},
        native_object=stack,
    )


class LayerStackFamilyDefinition(SourceFamilyDefinition):
    descriptor = SourceFamilyDescriptor(
        "ncls.layer-stack@1",
        1,
        "ncls.material-program@1",
        "ncls.layer-stack-random-walk@1",
        sha256_file(Path(__file__)),
    )

    def load_snapshot(self, locator: Mapping[str, Any]) -> SourceSnapshot:
        value = dict(locator)
        if set(value) != {"kind", "path"} or value.get("kind") != "material-program":
            raise ValueError(
                "LayerStack locator fields must be kind=material-program and path"
            )
        path = Path(str(value["path"])).resolve()
        program = MaterialProgram.from_json(path.read_text(encoding="utf-8"))
        stack = canonicalize_layer_stack(program)
        return snapshot_from_layer_stack(
            stack,
            source_asset_sha256=sha256_file(path),
            metadata=program.metadata,
        )

    @staticmethod
    def _stack(snapshot: SourceSnapshot) -> tuple[LayerStackIR, MaterialProgram]:
        program = MaterialProgram.from_json(snapshot.native_payload.decode("utf-8"))
        stack = snapshot.native_object if isinstance(snapshot.native_object, LayerStackIR) else canonicalize_layer_stack(program)
        return stack, program

    @staticmethod
    def _ids(snapshot: SourceSnapshot, stack: LayerStackIR) -> tuple[list[str], list[str]]:
        interface_ids = list(snapshot.editor_metadata.get("interface_ids", ()))
        medium_ids = list(snapshot.editor_metadata.get("medium_ids", ()))
        if len(interface_ids) != len(stack.interfaces):
            interface_ids = [f"interface-{index}-{sha256_json(asdict(value))[:12]}" for index, value in enumerate(stack.interfaces)]
        if len(medium_ids) != len(stack.media):
            medium_ids = [f"medium-{index}-{sha256_json(asdict(value))[:12]}" for index, value in enumerate(stack.media)]
        return interface_ids, medium_ids

    def describe_parameters(self, snapshot: SourceSnapshot) -> SourceParameterView:
        self.validate_snapshot(snapshot)
        stack, _ = self._stack(snapshot)
        interface_ids, medium_ids = self._ids(snapshot, stack)
        interfaces = ParameterNode(
            "/interfaces",
            "list",
            "Interfaces",
            tuple(_interface_node(index, len(stack.interfaces), element_id, value) for index, (element_id, value) in enumerate(zip(interface_ids, stack.interfaces, strict=True))),
            binding="structural",
            editable=True,
            allowed_operations=("insert",),
        )
        media = ParameterNode(
            "/media",
            "list",
            "Media",
            tuple(_medium_node(index, element_id, value) for index, (element_id, value) in enumerate(zip(medium_ids, stack.media, strict=True))),
            binding="structural",
            editable=True,
            allowed_operations=("insert",),
        )
        return SourceParameterView(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.snapshot_id,
            ParameterNode("/", "group", "LayerStack", (interfaces, media)),
        )

    def apply_edit(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> SourceEditResult:
        self.validate_patch(snapshot, patch)
        stack, program = self._stack(snapshot)
        interfaces = list(stack.interfaces)
        media = list(stack.media)
        interface_ids, medium_ids = self._ids(snapshot, stack)
        changed: list[str] = []
        for operation in patch.operations:
            parts = operation.target.strip("/").split("/")
            collection = parts[0] if parts else ""
            if collection not in {"interfaces", "media"}:
                raise ValueError(f"unsupported LayerStack edit target {operation.target!r}")
            values: list[Any] = interfaces if collection == "interfaces" else media
            ids = interface_ids if collection == "interfaces" else medium_ids
            if operation.operation == "insert":
                if operation.element_id in ids:
                    raise ValueError("LayerStack insert element_id must be unique")
                if not isinstance(operation.value, Mapping):
                    raise ValueError("LayerStack insert value must be an object")
                index = len(values) if operation.destination is None else operation.destination
                if collection == "interfaces":
                    if index >= len(interfaces):
                        raise ValueError("LayerStack interfaces may only be inserted above the opaque base")
                    values.insert(index, _interface_from_dict(operation.value))
                else:
                    values.insert(index, _medium_from_dict(operation.value))
                ids.insert(index, str(operation.element_id))
            elif len(parts) < 2:
                raise ValueError("LayerStack element edit requires an element path")
            else:
                try:
                    index = ids.index(parts[1])
                except ValueError as error:
                    raise ValueError(f"unknown LayerStack element_id {parts[1]!r}") from error
                if operation.operation == "remove":
                    values.pop(index)
                    ids.pop(index)
                elif operation.operation == "move":
                    if operation.destination is None or operation.destination >= len(values):
                        raise ValueError("LayerStack move destination is out of range")
                    values.insert(operation.destination, values.pop(index))
                    ids.insert(operation.destination, ids.pop(index))
                elif operation.operation == "replace-variant":
                    if collection != "interfaces":
                        raise ValueError("only LayerStack interfaces have variants")
                    payload = dict(operation.value) if isinstance(operation.value, Mapping) else {}
                    payload["variant"] = operation.variant
                    interfaces[index] = _interface_from_dict(payload)
                elif operation.operation == "set":
                    if len(parts) != 3:
                        raise ValueError("LayerStack set requires a field path")
                    field_name = parts[2]
                    current = values[index]
                    if field_name not in asdict(current):
                        raise ValueError(f"unknown LayerStack field {field_name!r}")
                    value = operation.value
                    if isinstance(asdict(current)[field_name], tuple):
                        value = tuple(value)
                    values[index] = replace(current, **{field_name: value})
                else:
                    raise ValueError(f"unsupported LayerStack edit operation {operation.operation!r}")
            changed.append(operation.target)
        edited = LayerStackIR(tuple(interfaces), tuple(media))
        payload = material_program_from_layer_stack(edited, metadata=program.metadata).to_json().encode("utf-8")
        result = SourceSnapshot(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.native_schema_id,
            snapshot.source_asset_sha256,
            payload,
            snapshot.resource_hashes,
            {"interface_ids": tuple(interface_ids), "medium_ids": tuple(medium_ids)},
            edited,
        )
        return SourceEditResult(
            result,
            tuple(changed),
            ("reference-binding", "compiled-material", "comparison-output"),
        )


SOURCE_FAMILY_DEFINITION = LayerStackFamilyDefinition()
