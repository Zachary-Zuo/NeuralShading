from __future__ import annotations

from pathlib import Path
from typing import Mapping

import MaterialX as mx

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
from ncls.source_materials.materialx import LoadedMaterialX


_VALUE_TYPES = {
    "float": "float",
    "boolean": "bool",
    "vector2": "vector2",
    "vector3": "vector3",
    "color3": "color3",
    "filename": "resource",
}


def snapshot_from_materialx(
    loaded: LoadedMaterialX,
    *,
    source_asset_sha256: str,
    resource_hashes: Mapping[str, str] | None = None,
    runtime_metadata: Mapping[str, object] | None = None,
) -> SourceSnapshot:
    payload = mx.writeToXmlString(loaded.document).encode("utf-8")
    return SourceSnapshot(
        "materialx.document@1.39.4",
        1,
        "ncls.materialx-document@1.39.4",
        source_asset_sha256,
        payload,
        dict(resource_hashes or {}),
        editor_metadata=dict(runtime_metadata or {}),
        native_object=loaded,
    )


class MaterialXFamilyDefinition(SourceFamilyDefinition):
    descriptor = SourceFamilyDescriptor(
        "materialx.document@1.39.4",
        1,
        "ncls.materialx-document@1.39.4",
        "ncls.materialx-polyhaven@1",
        sha256_file(Path(__file__)),
    )

    @staticmethod
    def _loaded(snapshot: SourceSnapshot) -> LoadedMaterialX:
        if not isinstance(snapshot.native_object, LoadedMaterialX):
            raise ValueError("MaterialX snapshot requires its loaded document runtime object")
        return snapshot.native_object

    def describe_parameters(self, snapshot: SourceSnapshot) -> SourceParameterView:
        self.validate_snapshot(snapshot)
        loaded = self._loaded(snapshot)
        nodes = []
        for element in loaded.document.traverseTree():
            if element.getCategory() != "input":
                continue
            connected = element.getConnectedNode() is not None or element.getConnectedOutput() is not None
            has_value = element.hasValueString()
            path = element.getNamePath()
            if not path:
                continue
            value_type = _VALUE_TYPES.get(element.getType(), "string")
            editable = has_value and not connected
            nodes.append(
                ParameterNode(
                    f"/inputs/{path}",
                    "value" if editable else "read-only",
                    element.getName(),
                    value_type=value_type if editable else None,
                    value=element.getValueString() if has_value else element.getConnectedNode().getNamePath() if element.getConnectedNode() is not None else "",
                    binding="constant" if editable else "graph",
                    editable=editable,
                    read_only_reason=None if editable else "connected MaterialX input requires an explicit graph edit",
                    allowed_operations=("set",) if editable else (),
                    metadata={"name_path": path, "materialx_type": element.getType(), "connected": connected},
                )
            )
        return SourceParameterView(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.snapshot_id,
            ParameterNode("/", "group", "MaterialX", (ParameterNode("/inputs", "group", "Inputs", tuple(nodes)),)),
        )

    def apply_edit(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> SourceEditResult:
        self.validate_patch(snapshot, patch)
        loaded = self._loaded(snapshot)
        document = mx.createDocument()
        mx.readFromXmlString(document, snapshot.native_payload.decode("utf-8"))
        edited = LoadedMaterialX(
            loaded.material,
            loaded.document_path,
            document,
            loaded.standard_library,
            loaded.search_path,
        )
        changed = []
        for operation in patch.operations:
            if operation.operation != "set" or not operation.target.startswith("/inputs/"):
                raise ValueError("MaterialX only accepts set operations on constant input paths")
            name_path = operation.target[len("/inputs/") :]
            edited.set_input_value(name_path, str(operation.value))
            changed.append(operation.target)
        result = SourceSnapshot(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.native_schema_id,
            snapshot.source_asset_sha256,
            mx.writeToXmlString(edited.document).encode("utf-8"),
            snapshot.resource_hashes,
            snapshot.editor_metadata,
            edited,
        )
        return SourceEditResult(result, tuple(changed), ("reference-binding", "compiled-material", "comparison-output"))


SOURCE_FAMILY_DEFINITION = MaterialXFamilyDefinition()
