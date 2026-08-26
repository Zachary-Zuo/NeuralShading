from __future__ import annotations

from pathlib import Path

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
from ncls.source_materials.merl import MerlMaterial


def snapshot_from_merl(
    material: MerlMaterial,
    *,
    source_asset_sha256: str,
    asset_root: Path | None = None,
) -> SourceSnapshot:
    return SourceSnapshot(
        "merl.measured-brdf@1",
        1,
        "ncls.merl-material@1",
        source_asset_sha256,
        material.to_json().encode("utf-8"),
        {material.table_uri: source_asset_sha256},
        editor_metadata={"asset_root": str(asset_root.resolve())} if asset_root is not None else {},
        native_object=material,
    )


class MerlFamilyDefinition(SourceFamilyDefinition):
    descriptor = SourceFamilyDescriptor(
        "merl.measured-brdf@1",
        1,
        "ncls.merl-material@1",
        "ncls.merl-brdf@1",
        sha256_file(Path(__file__)),
    )

    @staticmethod
    def _material(snapshot: SourceSnapshot) -> MerlMaterial:
        return snapshot.native_object if isinstance(snapshot.native_object, MerlMaterial) else MerlMaterial.from_json(snapshot.native_payload.decode("utf-8"))

    def describe_parameters(self, snapshot: SourceSnapshot) -> SourceParameterView:
        self.validate_snapshot(snapshot)
        material = self._material(snapshot)
        measurement = ParameterNode(
            "/measurement",
            "read-only",
            "Measurement Table",
            value=material.table_uri,
            binding="measurement",
            editable=False,
            read_only_reason="MERL has no continuous editable parameter; choose another source asset to switch tables",
            metadata={"material_id": material.material_id, "license": material.license},
        )
        return SourceParameterView(
            snapshot.family_id,
            snapshot.source_contract_version,
            snapshot.snapshot_id,
            ParameterNode("/", "group", "MERL measured BRDF", (measurement,)),
        )

    def apply_edit(self, snapshot: SourceSnapshot, patch: SourceEditPatch) -> SourceEditResult:
        self.validate_patch(snapshot, patch)
        raise ValueError("MERL measured BRDF exposes no continuous source edits")


SOURCE_FAMILY_DEFINITION = MerlFamilyDefinition()
