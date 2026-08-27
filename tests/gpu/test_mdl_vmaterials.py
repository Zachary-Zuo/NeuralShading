from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ncls.data.batch_sources import MdlLiveReferenceBatchSource
from ncls.data.collector import CollectionConfig
from ncls.data.providers.mdl import MdlProvider, MdlProviderConfig
from ncls.data.training_batch import TrainingRouteRequest
from ncls.paths import PROJECT_ROOT


ASSET_IDS = (
    "carpaint-shifting-flakes",
    "copper-antique-brushed-patinated",
    "aluminum-scratched",
    "ceramic-tiles-glazed-versailles",
    "velvet",
    "wood-tiles-pine-mosaic",
)


@pytest.mark.falcor
def test_vmaterials_shortlist_discovers_loads_and_evaluates() -> None:
    asset_root = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0"
    if not (asset_root / "PACKAGE-INFO.yaml").is_file():
        pytest.skip("vMaterials 2.4.0 未获取；运行 scripts/fetch_mdl_assets.ps1")
    manifest = json.loads(
        (PROJECT_ROOT / "references/mdl-vmaterials2-v1/assets.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(item["asset_id"] for item in manifest["assets"]) == ASSET_IDS
    results = {}
    for asset_id in ASSET_IDS:
        provider = MdlProvider(
            CollectionConfig(
                name=f"mdl-vmaterials-{asset_id}",
                view_count=1,
                light_count=2,
                spatial_sample_count=1,
                proposal="uniform",
                seed=31,
            ),
            MdlProviderConfig.from_vmaterials2((asset_id,)),
        )
        try:
            state = provider.source_states()[0]
            surfaces = provider.surface_samples(state)
            plan = provider.query_plan(state, surfaces)
            block = provider.evaluate(state, surfaces, plan)
            assert block.mean.shape == (1, 1, 2, 3)
            assert np.all(np.isfinite(block.mean))
            assert np.all(np.isfinite(block.reference_pdf))
            assert np.all(block.reference_pdf >= 0.0)
            results[asset_id] = state.snapshot.snapshot_id
        finally:
            provider.close()
    assert results == {
        item["asset_id"]: item["source_snapshot_id"] for item in manifest["assets"]
    }


@pytest.mark.falcor
@pytest.mark.parametrize(
    "asset_id",
    ("carpaint-shifting-flakes", "copper-antique-brushed-patinated"),
)
def test_fancy_vmaterials_use_unified_live_training_batch(asset_id: str) -> None:
    asset_root = PROJECT_ROOT / "assets/source-materials/mdl-vmaterials2/2.4.0"
    if not (asset_root / "PACKAGE-INFO.yaml").is_file():
        pytest.skip("vMaterials 2.4.0 未获取；运行 scripts/fetch_mdl_assets.ps1")
    provider = MdlProvider(
        CollectionConfig(
            name=f"mdl-live-{asset_id}",
            view_count=1,
            light_count=1,
            spatial_sample_count=1,
            proposal="uniform",
            seed=37,
        ),
        MdlProviderConfig.from_vmaterials2((asset_id,)),
    )
    source = MdlLiveReferenceBatchSource(
        provider,
        provider.source_states()[0],
        max_batch_size=4,
        query_tile_size=4,
        seed=41,
        device="cuda:0",
    )
    request = TrainingRouteRequest(
        "evaluator",
        4,
        1,
        0,
        0,
        43,
        {
            "direction_proposal": "uniform-half-difference@1",
            "target_estimator": "reference",
        },
    )
    try:
        batch = source.next_batch(request)
        assert batch.source_family_id == "mdl.program@1"
        assert batch.provenance["producer"] == "mdl-live-reference"
        assert batch.provenance["host_readback"] is False
        assert torch.all(torch.isfinite(batch.tensors["target"]))
        assert torch.count_nonzero(batch.tensors["mip_level"]) == 0
        batch.release()
    finally:
        if source._active_leases:
            next(iter(source._active_leases.values())).release()
        source.close()
