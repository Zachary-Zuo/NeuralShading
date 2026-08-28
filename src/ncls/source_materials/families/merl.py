from __future__ import annotations

import json
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

    def load_snapshot(self, locator: Mapping[str, Any]) -> SourceSnapshot:
        value = dict(locator)
        kind = value.pop("kind", None)
        material_id = value.pop("material_id", None)
        if kind != "catalog-asset" or not isinstance(material_id, str):
            raise ValueError(
                "MERL locator requires kind=catalog-asset and material_id"
            )
        project_root = Path(__file__).resolve().parents[4]
        asset_root = Path(
            str(value.pop("asset_root", project_root / "assets/source-materials/merl-brdf/v1"))
        ).resolve()
        material_index = Path(
            str(value.pop("material_index", project_root / "references/merl-brdf-v1/materials.json"))
        ).resolve()
        if value:
            raise ValueError(f"unexpected MERL locator fields: {sorted(value)}")
        document = json.loads(material_index.read_text(encoding="utf-8"))
        records = {
            str(record["material_id"]): record for record in document["materials"]
        }
        try:
            record = records[material_id]
        except KeyError as error:
            raise ValueError(f"unknown MERL material_id {material_id!r}") from error
        table = (asset_root / str(record["table_uri"])).resolve()
        table.relative_to(asset_root)
        if not table.is_file() or table.stat().st_size != int(record["size"]):
            raise FileNotFoundError(f"MERL source table is missing or truncated: {table}")
        source_hash = sha256_file(table)
        if source_hash != str(record["sha256"]):
            raise ValueError(f"MERL source table hash mismatch: {table}")
        material = MerlMaterial(
            material_id,
            str(record["table_uri"]),
            str(document["source_record"]),
            str(document["license"]),
        )
        return snapshot_from_merl(
            material,
            source_asset_sha256=source_hash,
            asset_root=asset_root,
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
