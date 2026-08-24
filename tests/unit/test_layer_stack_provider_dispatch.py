from __future__ import annotations

import numpy as np

from ncls.data.collector import CollectionConfig
from ncls.data.contract import SurfaceSample
from ncls.data.providers.layer_stack import LayerStackProvider, LayerStackProviderConfig
from ncls.data.reference import EvaluatedReferenceBatch


class _FakeReferenceEvaluator:
    creations: list[tuple[int, int, int]] = []

    def __init__(
        self,
        light_directions: np.ndarray,
        *,
        max_depth: int,
        max_query_group_batch: int,
        light_index_offset: int,
    ) -> None:
        del max_depth
        self.light_count = int(np.asarray(light_directions).shape[-2])
        self.max_query_group_batch = max_query_group_batch
        self.creations.append(
            (self.light_count, max_query_group_batch, light_index_offset)
        )


def _evaluate_fixed(
    evaluator: _FakeReferenceEvaluator,
    materials,
    view_directions: np.ndarray,
    *,
    samples_per_replica: int,
    query_group_seeds: np.ndarray,
    light_directions: np.ndarray,
) -> EvaluatedReferenceBatch:
    group_count = len(materials)
    lights = np.asarray(light_directions, dtype=np.float32)
    assert 1 <= group_count <= evaluator.max_query_group_batch
    assert np.asarray(view_directions).shape[0] == group_count
    assert np.asarray(query_group_seeds).shape == (group_count,)
    assert lights.shape == (group_count, evaluator.light_count, 3)
    mean = np.abs(lights)
    return EvaluatedReferenceBatch(
        mean,
        np.zeros_like(mean),
        mean,
        mean,
        np.full(group_count, 2 * samples_per_replica, dtype=np.uint64),
    )


def test_layer_stack_provider_reuses_dispatch_resources_without_changing_tiles(
    monkeypatch,
) -> None:
    _FakeReferenceEvaluator.creations = []
    monkeypatch.setattr(
        "ncls.data.providers.layer_stack.FalcorReferenceEvaluator",
        _FakeReferenceEvaluator,
    )
    monkeypatch.setattr(
        "ncls.data.providers.layer_stack.evaluate_reference_fixed",
        _evaluate_fixed,
    )
    collection = CollectionConfig(
        view_count=5,
        light_count=10,
        proposal="uniform",
    )
    provider = LayerStackProvider(
        collection,
        LayerStackProviderConfig(
            family_count=4,
            states_per_family=3,
            heldout_family_count=1,
            fixed_samples_per_replica=2,
            max_dispatch_queries=8,
        ),
    )
    state = provider.source_states()[0]
    plan = provider.query_plan(state, (SurfaceSample(),))

    first = provider.evaluate(state, (SurfaceSample(),), plan)
    second = provider.evaluate(state, (SurfaceSample(),), plan)

    assert _FakeReferenceEvaluator.creations == [(8, 1, 0), (2, 4, 8)]
    expected = np.abs(plan.light_directions)[None]
    np.testing.assert_array_equal(first.mean, expected)
    np.testing.assert_array_equal(second.mean, expected)
    np.testing.assert_array_equal(first.mean, second.mean)
    provider.close()
    assert provider._evaluator_cache == {}
