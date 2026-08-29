from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ncls.learning.mdl_metal_assets import MdlMetalNativeAssetCollection
from ncls.source_materials.mdl_metal import (
    MDL_METAL_EXPECTED_COUNTS,
    MdlMetalRegistry,
    MdlMetalTypedStateRecipe,
    PARAMETER_RESPONSIBILITIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
MODULE_ROOT = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"


def test_tracked_mdl_metal_registry_is_complete_and_fail_closed() -> None:
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    assert registry.payload["counts"] == MDL_METAL_EXPECTED_COUNTS
    assert len(registry.exports) == 692
    assert len(registry.graphs) == 178
    assert len(registry.texture_sets) == 52
    assert len(registry.parameter_schemas) == 64
    assert len(registry.rejected_cutout_exports) == 145
    assert {
        parameter["responsibility"]
        for record in registry.exports
        for parameter in record.parameters
    } == set(PARAMETER_RESPONSIBILITIES)
    cutout = registry.rejected_cutout_exports[0]["exact_locator"]
    with pytest.raises(ValueError, match="unknown, missing or cutout"):
        registry.resolve_exact_locator(str(cutout["module"]), str(cutout["export"]))
    with pytest.raises(ValueError, match="unknown, missing or cutout"):
        registry.resolve_exact_locator("::vMaterials_2::Metal::Missing", "missing")


def test_mdl_metal_registry_retains_texture_precision_roles_and_provenance() -> None:
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    slots = [slot for value in registry.texture_sets.values() for slot in value["slots"]]
    assert any(slot["pixel_type"] == "Rgba_16" for slot in slots)
    assert any(slot["provenance_kind"] == "mdl-sdk-static-table" for slot in slots)
    assert any(len(slot["channels"]) >= 3 for slot in slots)
    assert all(len(value["slots"]) <= 9 for value in registry.texture_sets.values())
    assert all(
        slot["normal_rule"] == "decode-renormalize-tangent"
        for slot in slots
        if "normal-tangent" in slot["roles"]
    )


def test_mdl_metal_registry_identity_detects_any_semantic_change() -> None:
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    changed = copy.deepcopy(registry.payload)
    changed["opaque_exports"][0]["finish"] = "tampered"
    with pytest.raises(ValueError, match="identity mismatch"):
        MdlMetalRegistry(changed)


@pytest.mark.skipif(not MODULE_ROOT.is_dir(), reason="vMaterials 2 assets are not installed")
def test_mdl_metal_native_asset_collection_describes_all_52_sets_without_loading_pixels() -> None:
    registry = MdlMetalRegistry.load(REGISTRY_PATH)
    first = MdlMetalNativeAssetCollection(registry, MODULE_ROOT)
    second = MdlMetalNativeAssetCollection(registry, MODULE_ROOT)
    assert len(first.descriptors) == 52
    assert first.collection_id == second.collection_id
    assert not first._artifact_cache
    for descriptor in first.descriptors:
        assert descriptor.domains
        for domain in descriptor.domains:
            assert domain.level_shapes[-1] == (1, 1)
            assert domain.channel_count >= 1


def test_mdl_metal_typed_state_train_validation_recipes_have_disjoint_identities() -> None:
    train = MdlMetalTypedStateRecipe("metal-full@1", "train", 17, 8)
    validation = MdlMetalTypedStateRecipe("metal-full@1", "validation", 17, 8)
    assert train.identity != validation.identity
    assert train.responsibilities == (
        "metal-core",
        "finish-microstructure",
        "aging-contamination",
        "coating-composite",
    )
