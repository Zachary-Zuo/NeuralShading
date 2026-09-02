from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_file
from ncls.core.source import (
    ParameterNode,
    SourceEditPatch,
    SourceEditResult,
    SourceFamilyDefinition,
    SourceFamilyDescriptor,
    SourceParameterView,
    SourceSnapshot,
)
from ncls.source_materials.mdl import MDL_FAMILY_ID, MDL_NATIVE_SCHEMA, MdlMaterialSource


_VALUE_TYPES = {
    "bool": "bool",
    "int": "int",
    "float": "float",
    "double": "double",
    "color": "color3",
    "float2": "vector2",
    "float3": "vector3",
    "float4": "vector4",
    "enum": "enum",
    "texture_2d": "resource",
}
_COMPONENTS = {"color": 3, "float2": 2, "float3": 3, "float4": 4}


def _validate_range(descriptor: dict[str, Any], values: tuple[float, ...]) -> None:
    minimum = descriptor.get("minimum")
    maximum = descriptor.get("maximum")
    if minimum is not None and any(item < float(minimum) for item in values):
        raise ValueError(f"MDL edit is below the hard minimum {minimum}")
    if maximum is not None and any(item > float(maximum) for item in values):
        raise ValueError(f"MDL edit is above the hard maximum {maximum}")


def _normalized_value(
    mdl_type: str,
    value: Any,
    descriptor: dict[str, Any],
    module_root: Path,
) -> Any:
    if mdl_type == "bool":
        if not isinstance(value, bool):
            raise ValueError("MDL bool edit requires a bool")
        return value
    if mdl_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("MDL int edit requires an integer")
        _validate_range(descriptor, (float(value),))
        return value
    if mdl_type == "enum":
        if not isinstance(value, str):
            raise ValueError("MDL enum edit requires a choice name")
        choices = tuple(descriptor.get("choices", ()))
        match = next((item for item in choices if item.get("name") == value), None)
        if match is None:
            raise ValueError(f"MDL enum edit is not one of the declared choices: {value}")
        return {"name": value, "value": int(match["value"])}
    if mdl_type in {"float", "double"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"MDL {mdl_type} edit requires a finite number")
        result = float(value)
        _validate_range(descriptor, (result,))
        return result
    if mdl_type in _COMPONENTS:
        if not isinstance(value, (tuple, list)) or len(value) != _COMPONENTS[mdl_type]:
            raise ValueError(f"MDL {mdl_type} edit has the wrong component count")
        result = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in result):
            raise ValueError("MDL compound edit requires finite numbers")
        _validate_range(descriptor, result)
        return result
    if mdl_type == "texture_2d":
        if isinstance(value, str):
            value = {"path": value, "effective_gamma": descriptor.get("value", {}).get("effective_gamma", 1.0)}
        if not isinstance(value, dict) or "path" not in value:
            raise ValueError("MDL texture_2d edit requires a pack-relative resource path")
        relative = Path(str(value["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("MDL texture_2d resource must stay below the module root")
        resolved_root = module_root.resolve()
        resolved = (resolved_root / relative).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("MDL texture_2d resource escapes the module root") from error
        if not resolved.is_file():
            raise ValueError(f"MDL texture_2d resource is missing: {relative.as_posix()}")
        gamma = float(value.get("effective_gamma", 1.0))
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("MDL texture_2d gamma must be finite and positive")
        return {"path": relative.as_posix(), "effective_gamma": gamma}
    raise ValueError(f"MDL parameter type is read-only in V1: {mdl_type}")


class MdlFamilyDefinition(SourceFamilyDefinition):
    descriptor = SourceFamilyDescriptor(
        MDL_FAMILY_ID,
        1,
        MDL_NATIVE_SCHEMA,
        "ncls.mdl-vmaterials2@1",
        sha256_file(Path(__file__)),
    )

    def load_snapshot(self, locator: Mapping[str, Any]) -> SourceSnapshot:
        value = dict(locator)
        kind = value.pop("kind", None)
        if kind != "mdl-export":
            raise ValueError("MDL locator requires kind=mdl-export")
        required = {"module_root", "module", "export"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"MDL locator is missing fields: {sorted(missing)}")
        module_root = Path(str(value.pop("module_root"))).resolve()
        module = str(value.pop("module"))
        export = str(value.pop("export"))
        arguments = value.pop("arguments", {})
        pack_id = str(value.pop("pack_id", "project.fixtures"))
        pack_version = str(value.pop("pack_version", "1"))
        if value or not isinstance(arguments, Mapping):
            raise ValueError(f"unexpected MDL locator fields: {sorted(value)}")

        from ncls.core.identity import sha256_json
        from ncls.references.mdl import (
            MdlCompiledArtifact,
            create_mdl_program_provider,
        )
        from ncls.source_materials.mdl import snapshot_from_mdl_artifact

        compiler = create_mdl_program_provider(module_root)
        cache_key = sha256_json(
            {
                "module_root": str(module_root),
                "module": module,
                "export": export,
                "arguments": dict(arguments),
                "pack_id": pack_id,
                "pack_version": pack_version,
                "semantic_identity": compiler.descriptor.semantic_identity,
                "build_identity": compiler.descriptor.build_identity,
            }
        )
        output = compiler.cache_root / "source-locators" / cache_key
        artifact = (
            MdlCompiledArtifact.load(output, verify_texture_payloads=False)
            if output.is_dir()
            else compiler.inspect(module, export, arguments, output=output)
        )
        return snapshot_from_mdl_artifact(
            artifact,
            module_root,
            pack_id=pack_id,
            pack_version=pack_version,
        )

    @staticmethod
    def _source(snapshot: SourceSnapshot) -> MdlMaterialSource:
        return (
            snapshot.native_object
            if isinstance(snapshot.native_object, MdlMaterialSource)
            else MdlMaterialSource.from_snapshot(snapshot)
        )

    def describe_parameters(self, snapshot: SourceSnapshot) -> SourceParameterView:
        self.validate_snapshot(snapshot)
        source = self._source(snapshot)
        nodes = []
        for name, descriptor in sorted(source.arguments.items()):
            mdl_type = str(descriptor["mdl_type"])
            editable = bool(descriptor.get("editable", False)) and mdl_type in _VALUE_TYPES
            is_resource = editable and mdl_type == "texture_2d"
            choices = tuple(str(item["name"]) for item in descriptor.get("choices", ()))
            value = descriptor.get("value")
            if editable and mdl_type == "enum" and isinstance(value, dict):
                value = value.get("name")
            nodes.append(
                ParameterNode(
                    f"/arguments/{name}",
                    "resource" if is_resource else ("value" if editable else "read-only"),
                    name.replace("_", " ").title(),
                    value_type=_VALUE_TYPES.get(mdl_type) if editable else None,
                    value=value,
                    choices=choices,
                    minimum=descriptor.get("minimum"),
                    maximum=descriptor.get("maximum"),
                    ui_hint="soft-range" if "soft_minimum" in descriptor else None,
                    binding="texture" if is_resource else "constant",
                    editable=editable,
                    read_only_reason=None if editable else descriptor.get(
                        "read_only_reason", f"MDL {mdl_type} parameter is read-only in V1"
                    ),
                    allowed_operations=("set",) if editable else (),
                    metadata={
                        "mdl_name": name,
                        "mdl_type": mdl_type,
                        **({"soft_minimum": descriptor["soft_minimum"]} if "soft_minimum" in descriptor else {}),
                        **({"soft_maximum": descriptor["soft_maximum"]} if "soft_maximum" in descriptor else {}),
                    },
                )
            )
        return SourceParameterView(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.snapshot_id,
            ParameterNode(
                "/",
                "group",
                "MDL Program",
                (ParameterNode("/arguments", "group", "Arguments", tuple(nodes)),),
            ),
        )

    def apply_edit(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> SourceEditResult:
        self.validate_patch(snapshot, patch)
        source = self._source(snapshot)
        arguments = {name: dict(value) for name, value in source.arguments.items()}
        changed = []
        for operation in patch.operations:
            parts = operation.target.strip("/").split("/")
            if operation.operation != "set" or len(parts) != 2 or parts[0] != "arguments":
                raise ValueError("MDL V1 only accepts set operations on /arguments/<name>")
            name = parts[1]
            if name not in arguments or not arguments[name].get("editable", False):
                raise ValueError(f"MDL argument {name!r} is not editable")
            mdl_type = str(arguments[name]["mdl_type"])
            arguments[name]["value"] = _normalized_value(
                mdl_type, operation.value, arguments[name], source.module_root
            )
            changed.append(operation.target)
        edited = MdlMaterialSource(
            source.module_root,
            source.pack_id,
            source.pack_version,
            source.module,
            source.export,
            arguments,
            source.mdl_language,
            source.mdl_sdk,
        )
        resource_hashes = dict(snapshot.resource_hashes)
        for name, descriptor in arguments.items():
            if descriptor.get("mdl_type") != "texture_2d":
                continue
            original = source.arguments.get(name, {}).get("value")
            replacement = descriptor.get("value")
            if isinstance(original, dict) and original.get("path") != replacement.get("path"):
                old_path = str(original.get("path", ""))
                if old_path and not any(
                    other_name != name
                    and isinstance(other.get("value"), dict)
                    and other.get("value", {}).get("path") == old_path
                    for other_name, other in arguments.items()
                ):
                    resource_hashes.pop(old_path, None)
            if isinstance(replacement, dict):
                relative = str(replacement["path"])
                resource_hashes[relative] = sha256_file(source.module_root / relative)
        result = SourceSnapshot(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.native_schema_id,
            snapshot.source_asset_sha256,
            edited.to_payload(),
            resource_hashes,
            snapshot.editor_metadata,
            edited,
        )
        return SourceEditResult(
            result,
            tuple(changed),
            ("reference-binding", "compiled-material", "comparison-output"),
        )


SOURCE_FAMILY_DEFINITION = MdlFamilyDefinition()
