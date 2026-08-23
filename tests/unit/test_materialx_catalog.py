from __future__ import annotations

from pathlib import Path

from ncls.source_materials.materialx import MaterialXAssetCatalog, MaterialXSourceMaterial


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_curated_materialx_catalog_has_eight_traceable_assets() -> None:
    catalog = MaterialXAssetCatalog(PROJECT_ROOT / "references" / "materialx-polyhaven-v1" / "assets.json")
    assert len(catalog.asset_ids) == 8
    material = catalog.source_material("american_walnut_veneer")
    assert material.physical_size_mm == (1000.0, 1000.0)
    assert MaterialXSourceMaterial.from_json(material.to_json()) == material
    assert any(record["semantic"] == "materialx-document" for record in catalog.records(material.asset_id))
