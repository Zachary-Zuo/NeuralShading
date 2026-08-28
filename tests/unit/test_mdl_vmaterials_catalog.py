from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ncls.paths import PROJECT_ROOT
from ncls.source_materials.mdl_catalog import MdlVmaterialsCatalog


CATALOG_PATH = PROJECT_ROOT / "references/mdl-vmaterials2-v1/families.json"
MODULE_ROOT = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"


def _catalog() -> MdlVmaterialsCatalog:
    return MdlVmaterialsCatalog(CATALOG_PATH)


def test_vmaterials_catalog_freezes_eleven_families_and_all_172_presets() -> None:
    catalog = _catalog()
    assert catalog.family_ids == (
        "ceramic-tiles-glazed-versailles",
        "carpaint-metallic",
        "carpaint-shifting-flakes",
        "effect-pigment-metallic",
        "velvet",
        "copper-antique-brushed-patinated",
        "aluminum-scratched",
        "retroreflective-material",
        "carbon-fiber",
        "suede-leather",
        "wood-tiles-pine",
    )
    assert tuple(int(catalog.family(item)["preset_count"]) for item in catalog.family_ids) == (
        27,
        31,
        31,
        11,
        15,
        9,
        6,
        7,
        8,
        16,
        11,
    )
    assert catalog.preset_count == 172
    assert catalog.manifest["runtime_supported_count"] == 164
    assert catalog.manifest["runtime_unsupported_count"] == 8
    assets = json.loads(
        (PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json").read_text(
            encoding="utf-8"
        )
    )["assets"]
    assert tuple(item["asset_id"] for item in assets) == (
        "carpaint-shifting-flakes",
        "copper-antique-brushed-patinated",
        "aluminum-scratched",
        "ceramic-tiles-glazed-versailles",
        "velvet",
        "wood-tiles-pine-mosaic",
        "carpaint-metallic",
        "effect-pigment-metallic",
        "retroreflective-material",
        "carbon-fiber",
        "suede-leather",
    )


def test_vmaterials_catalog_keeps_punched_presets_but_locator_fails_closed() -> None:
    catalog = _catalog()
    suede = catalog.family("suede-leather")
    punched = [item for item in suede["presets"] if not item["runtime_supported"]]
    assert len(punched) == 8
    for preset in punched:
        assert preset["unsupported_reasons"] == ["geometry.cutout_opacity"]
        assert preset["evaluation_subfamily"] == "punched-cutout"
        atlas = [
            item
            for item in preset["runtime_capability_audit"]["textures"]
            if item["pixel_type"] == "Rgba_16"
        ]
        assert len(atlas) == 1
        assert atlas[0]["dimensions"] == [1024, 1024, 1]
        assert atlas[0]["source_sha256"]
    preset = punched[0]
    with pytest.raises(ValueError, match="geometry.cutout_opacity"):
        catalog.locator("suede-leather", preset["preset_id"], module_root=MODULE_ROOT)
    locator = catalog.locator(
        "suede-leather",
        preset["preset_id"],
        module_root=MODULE_ROOT,
        allow_unsupported=True,
    )
    assert locator["kind"] == "mdl-export"
    assert locator["export"] == preset["exact_export"]


def test_vmaterials_catalog_rejects_duplicate_and_tampered_resource_signature(
    tmp_path: Path,
) -> None:
    value = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(value)
    duplicate["families"][0]["presets"][1]["preset_id"] = duplicate["families"][0][
        "presets"
    ][0]["preset_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="preset IDs"):
        MdlVmaterialsCatalog(path)

    tampered = copy.deepcopy(value)
    tampered["families"][0]["presets"][0]["resource_signature"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="resource signature"):
        MdlVmaterialsCatalog(path)
