from __future__ import annotations

import torch

from ncls.data import CollectionConfig
from ncls.data.batch_sources import MaterialXLiveReferenceBatchSource
from ncls.data.providers import MaterialXProvider, MaterialXProviderConfig
from ncls.data.training_batch import TrainingRouteRequest


provider = MaterialXProvider(
    CollectionConfig(
        name="nvidia-materialx-live-smoke",
        view_count=1,
        light_count=1,
        spatial_sample_count=1,
        proposal="uniform",
        seed=91,
    ),
    MaterialXProviderConfig(asset_ids=("american_walnut_veneer",)),
)
source = MaterialXLiveReferenceBatchSource(
    provider,
    tuple(provider.source_states())[0],
    max_batch_size=4,
    query_tile_size=64,
    seed=91,
)
common = {
    "direction_proposal": "uniform-half-difference@1",
    "mip_exponential_scale": 1.0,
    "spatial_samples_per_texel_area": 0.25,
    "maximum_spatial_samples": 2,
    "mollification": {"steps": 2, "start_degrees": 10.0, "samples": 2},
}
try:
    evaluator = source.next_batch(
        TrainingRouteRequest(
            "evaluator", 4, 1, 0, 0, 91,
            {**common, "target_estimator": "reference"},
        )
    )
    sampler = source.next_batch(
        TrainingRouteRequest(
            "sampler", 4, 1, 0, 1, 92,
            {**common, "target_estimator": "learned-evaluator"},
        )
    )
    assert evaluator.device.type == "cuda" and sampler.device.type == "cuda"
    assert evaluator.tensors["target"].shape == (4, 1, 3)
    assert evaluator.tensors["native_features"].shape == (4, 38)
    assert torch.isfinite(evaluator.tensors["target"]).all()
    assert evaluator.provenance["host_readback"] is False
    assert evaluator.provenance["route_name"] != sampler.provenance["route_name"]
    print(
        "materialx-live-ok",
        source.materialization_features().level_shapes,
        float(evaluator.tensors["target"].mean()),
    )
    sampler.release()
    evaluator.release()
finally:
    source.close()
