from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ncls.learning.batches import TrainingConditioning, TrainingRouteRequest
from ncls.learning.producer import OnlineTrainingProducer, _group_block_sequence
from ncls.learning.source_adaptation import (
    DenseNativeAssetCollection,
    NativeAssetRole,
)


class _Lease:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        assert not self.released
        self.released = True


class _RejectingSession:
    def __init__(self) -> None:
        self.masks = (
            torch.tensor([True, False, True, False]),
            torch.tensor([False, True]),
            torch.tensor([True]),
        )
        self.leases: list[_Lease] = []

    def evaluate(
        self,
        query,
        wi,
        seeds,
        *,
        evaluation_samples,
        footprint_samples,
        source_execution_mode,
    ):
        del seeds, evaluation_samples, footprint_samples, source_execution_mode
        mask = self.masks[len(self.leases)]
        assert len(mask) == query.batch_size
        lease = _Lease()
        self.leases.append(lease)
        f = query.wo[:, None, 0:1].expand(-1, 1, 3).clone()
        return SimpleNamespace(valid=mask[:, None], f=f, lease=lease)


def test_evaluator_batch_compacts_reference_horizon_rejections() -> None:
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.config = SimpleNamespace(online_query={"evaluation_samples": 1})
    producer.session = _RejectingSession()
    generator = torch.Generator().manual_seed(9)
    cursor = 0

    def conditioning(request, group):
        del group
        nonlocal cursor
        values = torch.arange(cursor, cursor + request.batch_size, dtype=torch.float32)
        cursor += request.batch_size
        wo = torch.stack((values, torch.zeros_like(values), torch.ones_like(values)), dim=1)
        result = TrainingConditioning(
            "fixture.family",
            ("a" * 64,),
            {
                "source_index": torch.zeros(request.batch_size, dtype=torch.int64),
                "wo": wo,
            },
            {
                "route_name": request.name,
                "request_index": len(producer.session.leases),
                "reference_execution_group_id": "fixture-group",
            },
        )
        return result, generator, wo

    producer._conditioning = conditioning
    request = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 4, 1, 0, 7, {}
    )
    batch = producer._evaluator_batch(request, SimpleNamespace())

    torch.testing.assert_close(
        batch.conditioning.tensors["wo"][:, 0],
        torch.tensor([0.0, 2.0, 5.0, 6.0]),
    )
    torch.testing.assert_close(batch.target_f[:, 0, 0], torch.tensor([0.0, 2.0, 5.0, 6.0]))
    assert batch.provenance["candidate_count"] == 7
    assert batch.provenance["rejected_count"] == 3
    assert batch.provenance["rejection_rounds"] == 3
    assert all(lease.released for lease in producer.session.leases)


def test_online_query_resume_rejects_typed_state_pool_drift_before_restoring_cursors() -> None:
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.query_stream_identity = "a" * 64
    producer.typed_state_pool_identity = "b" * 64
    producer._generators = {}
    producer._request_count = {"evaluator": 3}
    producer._group_cursor = {"evaluator": 2}
    producer._asset_tile_cursor = {"asset": 7}
    state = producer.state_dict()
    changed = {**state, "typed_state_pool_identity": "c" * 64}
    with pytest.raises(ValueError, match="typed-state pool identity mismatch"):
        producer.load_state_dict(changed)
    assert producer._request_count == {"evaluator": 3}
    restored = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    restored.device = torch.device("cpu")
    restored.query_stream_identity = producer.query_stream_identity
    restored.typed_state_pool_identity = producer.typed_state_pool_identity
    restored._generators = {}
    restored._request_count = {}
    restored._group_cursor = {}
    restored._asset_tile_cursor = {}
    restored.load_state_dict(state)
    assert restored._request_count == producer._request_count
    assert restored._group_cursor == producer._group_cursor
    assert restored._asset_tile_cursor == producer._asset_tile_cursor


def test_group_block_schedule_is_shared_by_routes_and_changes_only_at_boundary() -> None:
    groups = tuple(SimpleNamespace(group_id=str(index)) for index in range(3))
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer._group_schedule_recipe = "group-block-balanced@1"
    producer._group_block_steps = 64
    producer._group_validation_offset_blocks = 1
    producer._group_sequence = groups
    producer.ddp_world_size = 1
    producer.ddp_rank = 0
    producer._group_cursor = {}

    evaluator = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 1, 1, 63, 1, {}
    )
    sampler = TrainingRouteRequest(
        "sampler", "method-sampler", 1, 1, 63, 2, {}
    )
    next_block = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 1, 1, 64, 3, {}
    )
    validation = TrainingRouteRequest(
        "validation:evaluator",
        "reference-evaluator",
        1,
        1,
        63,
        4,
        {"validation": True},
    )
    assert producer._select_group(evaluator) is groups[0]
    assert producer._select_group(sampler) is groups[0]
    assert producer._select_group(next_block) is groups[1]
    assert producer._select_group(validation) is groups[1]
    assert producer._group_cursor == {}


def test_group_block_cycle_has_full_prefix_and_exact_record_weighting() -> None:
    groups = tuple(
        SimpleNamespace(group_id=str(index), records=(object(),) * weight)
        for index, weight in enumerate((1, 3, 2))
    )
    sequence = _group_block_sequence(groups, "a" * 64)
    assert sequence[:3] == groups
    assert len(sequence) == 6
    assert [sequence.count(group) for group in groups] == [1, 3, 2]
    assert sequence == _group_block_sequence(groups, "a" * 64)


def test_asset_route_selects_explicit_cohort_and_round_robins_assets() -> None:
    collection = DenseNativeAssetCollection(
        (
            (torch.zeros((4, 4, 1)),),
            (torch.ones((1, 1, 1)),),
        ),
        ("large", "small"),
        "fixture-schema",
        "surface",
        "uv0",
        "wrap",
        (NativeAssetRole("value", "scalar", 0, 1, "linear", "box"),),
    )
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.adapter = SimpleNamespace(native_assets=lambda: collection)
    producer.query_stream_identity = "a" * 64
    producer._asset_tile_cursor = {}
    producer._request_count = {}
    request = TrainingRouteRequest(
        "asset",
        "asset-tile",
        2,
        1,
        0,
        0,
        {"asset_indices": [0, 1], "max_core_texels": 4, "halo": 0},
    )
    batch = producer._asset_tile_batch(request)
    try:
        assert [tile.asset_index for tile in batch.tiles] == [0, 1]
        assert batch.provenance["asset_indices"] == [0, 1]
    finally:
        batch.release()
    invalid = TrainingRouteRequest(
        "invalid",
        "asset-tile",
        1,
        1,
        0,
        0,
        {"asset_indices": [1, 1], "max_core_texels": 4, "halo": 0},
    )
    with pytest.raises(ValueError, match="duplicate"):
        producer._asset_tile_batch(invalid)
