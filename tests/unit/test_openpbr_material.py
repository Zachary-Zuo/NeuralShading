from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ncls.source_materials.openpbr import ConstantBinding, OpenPBRMaterial, PARAMETERS, TextureBinding


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_openpbr_defaults_json_round_trip_preserves_complete_parameter_set() -> None:
    material = (
        OpenPBRMaterial.defaults()
        .with_parameter("coat_weight", 0.75)
        .with_parameter("base_color", TextureBinding("textures/base.png", "srgb_texture", encoding="srgb"))
    )
    restored = OpenPBRMaterial.from_json(material.to_json())
    assert set(restored.parameters) == set(PARAMETERS)
    assert restored.parameters["coat_weight"] == ConstantBinding(0.75)
    assert restored.parameters["base_color"] == material.parameters["base_color"]
    assert restored.to_dict() == material.to_dict()


def test_official_openpbr_materialx_edit_round_trip(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "external" / "OpenPBR" / "examples" / "open_pbr_carpaint.mtlx"
    material = OpenPBRMaterial.from_materialx(source)
    assert material.parameters["coat_weight"] == ConstantBinding(1.0)
    assert material.metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    edited = material.with_parameter("coat_roughness", 0.17)
    output = tmp_path / "edited.mtlx"
    edited.save_materialx(output)
    restored = OpenPBRMaterial.from_materialx(output)
    assert restored.parameters["coat_roughness"] == ConstantBinding(0.17)
    assert restored.parameters["base_color"] == ConstantBinding((0.1, 0.6, 0.9))


def test_official_openpbr_material_index_matches_locked_source() -> None:
    index_path = PROJECT_ROOT / "references" / "openpbr-1.1.1-v1" / "materials.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["revision"] == "f8d6d947dfae4c9b599965a86c22826ea7a8dbfb"
    assert len(index["materials"]) == 83
    for record in index["materials"]:
        path = PROJECT_ROOT / "external" / "OpenPBR" / record["document"]
        content = path.read_bytes()
        assert len(content) == record["size"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
