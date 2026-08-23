from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ncls.source_materials import (
    MaterialXReference,
    MerlBrdfReference,
    MerlMaterial,
    OpenPBRMaterial,
    OpenPBRReference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_downloaded_merl_table_opens() -> None:
    asset_root = PROJECT_ROOT / "assets" / "source-materials" / "merl-brdf" / "v1"
    if not (asset_root / "complete.json").is_file():
        pytest.skip("MERL source material asset is not downloaded")
    marker = json.loads((asset_root / "complete.json").read_text(encoding="utf-8"))
    assert marker["archive"]["md5"] == "7141af4c12b4c4feed299769260b3604"
    material_index = PROJECT_ROOT / marker["material_index"]
    assert hashlib.sha256(material_index.read_bytes()).hexdigest() == marker["material_index_sha256"]
    reference = MerlBrdfReference(
        MerlMaterial("alum-bronze", "expanded/BRDFDatabase/brdfs/alum-bronze.binary"),
        asset_root,
    )
    assert reference.table_path.is_file()


def test_openpbr_reference_evaluates_and_samples_official_material() -> None:
    executable = PROJECT_ROOT / "build" / "openpbr-probe" / "Release" / "ncls_openpbr_probe.exe"
    if not executable.is_file():
        pytest.skip("OpenPBR reference probe is not built")
    material = OpenPBRMaterial.from_materialx(
        PROJECT_ROOT / "external" / "OpenPBR" / "examples" / "open_pbr_carpaint.mtlx"
    )
    reference = OpenPBRReference(executable)
    evaluated = reference.evaluate(material, [[0.0, 0.0, 1.0]], [[0.3, 0.0, 0.9539392]])
    assert evaluated.response_cos.shape == (1, 3)
    assert evaluated.pdf.shape == (1,)
    assert np.isfinite(evaluated.response_cos).all() and np.isfinite(evaluated.pdf).all()
    sampled = reference.sample(material, [[0.0, 0.0, 1.0]], [[0.3, 0.7, 0.2]])
    assert sampled.light_direction.shape == (1, 3)
    assert sampled.weight.shape == (1, 3)
    assert sampled.pdf.shape == (1,)
    assert np.isfinite(sampled.light_direction).all() and np.isfinite(sampled.weight).all()


def test_downloaded_materialx_assets_validate_and_generate_glsl(tmp_path: Path) -> None:
    asset_root = PROJECT_ROOT / "assets" / "source-materials" / "materialx-polyhaven" / "v1"
    if not (asset_root / "complete.json").is_file():
        pytest.skip("Poly Haven MaterialX source assets are not downloaded")
    marker = json.loads((asset_root / "complete.json").read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / marker["manifest"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == marker["manifest_sha256"]
    reference = MaterialXReference(
        PROJECT_ROOT / "external" / "MaterialX",
        asset_root,
        manifest_path,
    )
    for asset_id in reference.catalog.asset_ids:
        loaded = reference.load(asset_id, verify_files=True)
        records = {str(record["path"]).replace("\\", "/") for record in reference.catalog.records(asset_id)}
        references = loaded.referenced_files()
        assert references
        for filename in references:
            dependency = (loaded.document_path.parent / filename).resolve()
            assert dependency.is_file(), f"native MaterialX dependency is missing: {dependency}"
            assert dependency.relative_to(asset_root.resolve()).as_posix() in records
        assert loaded.generate_glsl()

    editable = reference.load("american_walnut_veneer")
    ior = next(item for item in editable.editable_inputs() if item.name_path.endswith("/specular_IOR"))
    editable.set_input_value(ior.name_path, "1.7")
    edited_path = tmp_path / "american_walnut_veneer_edited.mtlx"
    editable.save(edited_path)
    assert edited_path.is_file()
