from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ncls.learning.mdl_metal_assets import MdlMetalNativeAssetCollection
from ncls.learning.source_states import expand_source_states
from ncls.source_materials.families.mdl import MdlFamilyDefinition
from ncls.source_materials.mdl_metal import MdlMetalRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0/Materials"


@pytest.mark.skipif(not MODULE_ROOT.is_dir(), reason="vMaterials 2 assets are not installed")
def test_mdl_metal_collection_decodes_rgba16_and_provider_bsdf_tiles_lazily() -> None:
    registry = MdlMetalRegistry.load(
        PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
    )
    collection = MdlMetalNativeAssetCollection(registry, MODULE_ROOT, working_set_capacity=2)
    texture_set_ids = sorted(registry.texture_sets)

    def select(predicate):
        for asset_index, texture_set_id in enumerate(texture_set_ids):
            for slot in registry.texture_sets[texture_set_id]["slots"]:
                if predicate(slot):
                    return asset_index, f"slot-{slot['slot_index']}"
        raise AssertionError("tracked registry lost the required texture slot")

    rgba_asset, rgba_domain = select(lambda slot: slot["pixel_type"] == "Rgba_16")
    bsdf_asset, bsdf_domain = select(
        lambda slot: slot["provenance_kind"] == "mdl-sdk-static-table"
    )
    for asset_index, domain_id in (
        (rgba_asset, rgba_domain),
        (bsdf_asset, bsdf_domain),
    ):
        request = next(collection.iter_tile_requests(asset_index, domain_id, 16, 1))
        tile = collection.acquire_tile(request, torch.device("cpu"))
        try:
            assert tile.values.shape[:2] == (
                request.core_shape[0] + 2,
                request.core_shape[1] + 2,
            )
            assert bool(torch.isfinite(tile.values).all())
            assert float(tile.values.min()) >= 0.0
            assert float(tile.values.max()) <= 1.0
        finally:
            tile.release()
    assert len(collection._artifact_cache) <= 2


@pytest.mark.skipif(not MODULE_ROOT.is_dir(), reason="vMaterials 2 assets are not installed")
def test_generic_source_state_registry_expands_disjoint_mdl_metal_splits() -> None:
    registry = MdlMetalRegistry.load(
        PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json"
    )
    selected = next(
        record
        for record in registry.exports
        if record.exact_locator["module"] == "::vMaterials_2::Metal::Aging_Copper"
        and record.exact_locator["export"].split("::")[-1].startswith("Aging_Copper(")
    )
    family = MdlFamilyDefinition()
    base = family.load_snapshot({**selected.exact_locator, "module_root": str(MODULE_ROOT)})

    def expand(split: str):
        return expand_source_states(
            family,
            (base,),
            {
                "schema": "ncls.mdl-metal-typed-state-recipe@1",
                "registry": "references/mdl-vmaterials2-v1/metal-opaque-v1.json",
                "recipe_id": "integration-state@1",
                "split": split,
                "seed": 13,
                "states_per_export": 4,
                "responsibilities": [
                    "metal-core",
                    "finish-microstructure",
                    "aging-contamination",
                    "coating-composite",
                ],
                "default_weight": 0.0,
                "boundary_weight": 0.0,
            },
        )

    train, validation = expand("train"), expand("validation")
    assert train.base_snapshot_ids == validation.base_snapshot_ids == (base.snapshot_id,)
    assert train.identity != validation.identity
    assert len(train.snapshots) >= 2 and len(validation.snapshots) >= 2
    assert {snapshot.snapshot_id for snapshot in train.snapshots[1:]}.isdisjoint(
        snapshot.snapshot_id for snapshot in validation.snapshots[1:]
    )
