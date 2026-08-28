from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ncls.core.identity import sha256_file, sha256_json
from ncls.core.source import (
    ParameterNode,
    SourceEditPatch,
    SourceEditResult,
    SourceFamilyDefinition,
    SourceFamilyDescriptor,
    SourceParameterView,
    SourceSnapshot,
)
from ncls.source_materials.openpbr import (
    ConstantBinding,
    GeometryBinding,
    GraphBinding,
    OpenPBRMaterial,
    PARAMETERS,
    TextureBinding,
)


_VALUE_TYPES = {
    "float": "float",
    "boolean": "bool",
    "vector2": "vector2",
    "vector3": "vector3",
    "color3": "color3",
}


def snapshot_from_openpbr(
    material: OpenPBRMaterial,
    *,
    source_asset_sha256: str | None = None,
    resource_hashes: Mapping[str, str] | None = None,
    asset_root: Path | None = None,
) -> SourceSnapshot:
    payload = material.to_json().encode("utf-8")
    return SourceSnapshot(
        "openpbr.material@1.1.1",
        1,
        "ncls.openpbr-material@1",
        source_asset_sha256 or sha256_json(material.to_dict()),
        payload,
        dict(resource_hashes or {}),
        editor_metadata={"asset_root": str(asset_root.resolve())} if asset_root is not None else {},
        native_object=material,
    )


class OpenPBRFamilyDefinition(SourceFamilyDefinition):
    descriptor = SourceFamilyDescriptor(
        "openpbr.material@1.1.1",
        1,
        "ncls.openpbr-material@1",
        "ncls.openpbr@1",
        sha256_file(Path(__file__)),
    )

    def load_snapshot(self, locator: Mapping[str, Any]) -> SourceSnapshot:
        value = dict(locator)
        kind = value.pop("kind", None)
        path_value = value.pop("path", None)
        if value or kind not in {"materialx-document", "json-document"} or path_value is None:
            raise ValueError(
                "OpenPBR locator requires kind=materialx-document|json-document and path"
            )
        path = Path(str(path_value)).resolve()
        if kind == "materialx-document":
            material = OpenPBRMaterial.from_materialx(path)
        else:
            material = OpenPBRMaterial.from_json(path.read_text(encoding="utf-8"))
        return snapshot_from_openpbr(
            material,
            source_asset_sha256=sha256_file(path),
            asset_root=path.parent,
        )

    @staticmethod
    def _material(snapshot: SourceSnapshot) -> OpenPBRMaterial:
        return snapshot.native_object if isinstance(snapshot.native_object, OpenPBRMaterial) else OpenPBRMaterial.from_json(snapshot.native_payload.decode("utf-8"))

    def describe_parameters(self, snapshot: SourceSnapshot) -> SourceParameterView:
        self.validate_snapshot(snapshot)
        material = self._material(snapshot)
        nodes = []
        for name, spec in PARAMETERS.items():
            binding = material.parameters[name]
            editable = isinstance(binding, ConstantBinding)
            if isinstance(binding, ConstantBinding):
                value, provenance = binding.value, "constant"
            elif isinstance(binding, TextureBinding):
                value, provenance = binding.uri, "texture"
            elif isinstance(binding, GraphBinding):
                value, provenance = binding.node, "graph"
            elif isinstance(binding, GeometryBinding):
                value, provenance = binding.symbol, "geometry"
            else:
                raise TypeError(f"unsupported OpenPBR binding {type(binding)!r}")
            nodes.append(
                ParameterNode(
                    f"/parameters/{name}",
                    "value" if editable else "read-only",
                    name.replace("_", " ").title(),
                    value_type=_VALUE_TYPES[spec.value_type] if editable else None,
                    value=value,
                    default=spec.default,
                    binding=provenance,
                    editable=editable,
                    read_only_reason=None if editable else f"{provenance} binding is owned by the OpenPBR source graph",
                    allowed_operations=("set",) if editable else (),
                    metadata={"openpbr_name": name, "binding": provenance},
                )
            )
        return SourceParameterView(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.snapshot_id,
            ParameterNode("/", "group", "OpenPBR 1.1.1", (ParameterNode("/parameters", "group", "Parameters", tuple(nodes)),)),
        )

    def apply_edit(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> SourceEditResult:
        self.validate_patch(snapshot, patch)
        material = self._material(snapshot)
        changed = []
        for operation in patch.operations:
            parts = operation.target.strip("/").split("/")
            if operation.operation != "set" or len(parts) != 2 or parts[0] != "parameters":
                raise ValueError("OpenPBR only accepts set operations on /parameters/<name>")
            name = parts[1]
            if not isinstance(material.parameters.get(name), ConstantBinding):
                raise ValueError(f"OpenPBR parameter {name!r} is not a constant binding")
            material = material.with_parameter(name, operation.value)
            changed.append(operation.target)
        result = SourceSnapshot(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.native_schema_id,
            snapshot.source_asset_sha256,
            material.to_json().encode("utf-8"),
            snapshot.resource_hashes,
            snapshot.editor_metadata,
            material,
        )
        return SourceEditResult(result, tuple(changed), ("reference-binding", "compiled-material", "comparison-output"))


SOURCE_FAMILY_DEFINITION = OpenPBRFamilyDefinition()
