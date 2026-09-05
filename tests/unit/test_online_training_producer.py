from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ncls.data import LogicalReferenceRequest
from ncls.learning.batches import TrainingConditioning, TrainingRouteRequest
from ncls.learning.producer import (
    OnlineTrainingProducer,
    _EvaluatorLogicalRequest,
    _balanced_probe_directions,
    _group_block_sequence,
)
from ncls.learning.source_adaptation import (
    DenseNativeAssetCollection,
    NativeAssetRole,
)
from ncls.learning.source_adapters import _balanced_one_native_texel_offsets


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


class _AllValidSession:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

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
        del wi, seeds, evaluation_samples, footprint_samples, source_execution_mode
        self.batch_sizes.append(query.batch_size)
        lease = _Lease()
        target = query.wo[:, None, 0:1].expand(-1, 1, 3).clone()
        return SimpleNamespace(
            valid=torch.ones((query.batch_size, 1), dtype=torch.bool),
            f=target,
            lease=lease,
        )


class _PairedUvSession:
    def __init__(self) -> None:
        self.uvs: list[torch.Tensor] = []
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
        del wi, seeds, evaluation_samples, footprint_samples, source_execution_mode
        assert query.uv is not None
        self.uvs.append(query.uv.clone())
        lease = _Lease()
        self.leases.append(lease)
        target = query.uv[:, None, 0:1].expand(-1, 1, 3).clone()
        return SimpleNamespace(
            valid=torch.ones((query.batch_size, 1), dtype=torch.bool),
            f=target,
            lease=lease,
        )


def test_balanced_probe_directions_are_deterministic_bounded_and_quota_complete() -> None:
    first = _balanced_probe_directions(
        400, torch.Generator().manual_seed(91), torch.device("cpu")
    )
    second = _balanced_probe_directions(
        400, torch.Generator().manual_seed(91), torch.device("cpu")
    )
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    wo, wi = first
    assert bool(torch.all(wo[:, 2] > 0.0))
    assert bool(torch.all(wi[:, 2] > 0.0))
    torch.testing.assert_close(torch.linalg.vector_norm(wo, dim=1), torch.ones(400))
    torch.testing.assert_close(torch.linalg.vector_norm(wi, dim=1), torch.ones(400))
    mirror = torch.stack((-wo[:, 0], -wo[:, 1], wo[:, 2]), dim=1)
    angle = torch.rad2deg(
        torch.acos(torch.clamp(torch.sum(mirror * wi, dim=1), -1.0, 1.0))
    )
    assert int(torch.count_nonzero(angle <= 8.0001)) >= 100
    assert int(torch.count_nonzero((wi[:, 2] >= 0.02) & (wi[:, 2] <= 0.15))) >= 100
    with pytest.raises(ValueError, match="divisible by four"):
        _balanced_probe_directions(
            10, torch.Generator().manual_seed(91), torch.device("cpu")
        )


def test_paired_uv_uses_independent_native_width_and_height() -> None:
    extents = torch.tensor([[8.0, 4.0]]).expand(4, -1)
    offsets = _balanced_one_native_texel_offsets(
        extents, torch.Generator().manual_seed(17)
    )
    x = offsets[:, 0][offsets[:, 0] > 0.0]
    y = offsets[:, 1][offsets[:, 1] > 0.0]
    torch.testing.assert_close(x, torch.full((2,), 1.0 / 8.0))
    torch.testing.assert_close(y, torch.full((2,), 1.0 / 4.0))
    with pytest.raises(ValueError, match="even"):
        _balanced_one_native_texel_offsets(
            extents[:3], torch.Generator().manual_seed(17)
        )


def test_evaluator_batch_compacts_reference_horizon_rejections() -> None:
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.config = SimpleNamespace(online_query={"evaluation_samples": 1})
    producer.session = _RejectingSession()
    generator = torch.Generator().manual_seed(9)
    cursor = 0

    def conditioning(request, group, request_generator, request_index):
        del group
        assert request_generator is generator
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
                "request_index": request_index,
                "reference_execution_group_id": "fixture-group",
            },
        )
        return result, wo

    producer._conditioning = conditioning
    request = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 4, 1, 0, 7, {}
    )
    group = SimpleNamespace(group_id="fixture-group")
    batch = producer._dispatch_evaluator_requests(
        (
            LogicalReferenceRequest(
                0,
                "dispatch",
                _EvaluatorLogicalRequest(request, group, generator, 0, "dispatch"),
                {},
            ),
        )
    )[0]

    torch.testing.assert_close(
        batch.conditioning.tensors["wo"][:, 0],
        torch.tensor([0.0, 2.0, 5.0, 6.0]),
    )
    torch.testing.assert_close(batch.target_f[:, 0, 0], torch.tensor([0.0, 2.0, 5.0, 6.0]))
    assert batch.provenance["candidate_count"] == 7
    assert batch.provenance["rejected_count"] == 3
    assert batch.provenance["rejection_rounds"] == 3
    assert all(lease.released for lease in producer.session.leases)


def test_packed_evaluator_matches_one_step_dispatch_with_one_backend_call() -> None:
    def build(session: _AllValidSession) -> OnlineTrainingProducer:
        producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
        producer.device = torch.device("cpu")
        producer.config = SimpleNamespace(online_query={"evaluation_samples": 1})
        producer.session = session

        def conditioning(request, group, generator, request_index):
            values = torch.rand(request.batch_size, generator=generator)
            wo = torch.stack(
                (values, torch.zeros_like(values), torch.ones_like(values)), dim=1
            )
            return (
                TrainingConditioning(
                    "fixture.family",
                    ("a" * 64,),
                    {
                        "source_index": torch.zeros(
                            request.batch_size, dtype=torch.int64
                        ),
                        "wo": wo,
                    },
                    {
                        "route_name": request.name,
                        "request_index": request_index,
                        "reference_execution_group_id": group.group_id,
                    },
                ),
                wo,
            )

        producer._conditioning = conditioning
        return producer

    group = SimpleNamespace(group_id="fixture-group")
    requests = tuple(
        TrainingRouteRequest(
            "evaluator", "reference-evaluator", 3, 1, index, 7, {}
        )
        for index in range(2)
    )

    baseline_session = _AllValidSession()
    baseline_producer = build(baseline_session)
    baseline = tuple(
        baseline_producer._dispatch_evaluator_requests(
            (
                LogicalReferenceRequest(
                    index,
                    "dispatch",
                    _EvaluatorLogicalRequest(
                        request,
                        group,
                        torch.Generator().manual_seed(20 + index),
                        index,
                        "dispatch",
                    ),
                    {},
                ),
            )
        )[0]
        for index, request in enumerate(requests)
    )

    packed_session = _AllValidSession()
    packed_producer = build(packed_session)
    packed = packed_producer._dispatch_evaluator_requests(
        tuple(
            LogicalReferenceRequest(
                index,
                "dispatch",
                _EvaluatorLogicalRequest(
                    request,
                    group,
                    torch.Generator().manual_seed(20 + index),
                    index,
                    "dispatch",
                ),
                {},
            )
            for index, request in enumerate(requests)
        )
    )

    assert baseline_session.batch_sizes == [3, 3]
    assert packed_session.batch_sizes == [6]
    for expected, actual in zip(baseline, packed, strict=True):
        torch.testing.assert_close(expected.conditioning.tensors["wo"], actual.conditioning.tensors["wo"])
        torch.testing.assert_close(expected.target_f, actual.target_f)
        assert actual.provenance["reference_dispatch_logical_steps"] == 2


def test_evaluator_dispatch_queries_paired_uv_with_shared_directions_and_seeds() -> None:
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.config = SimpleNamespace(online_query={"evaluation_samples": 1})
    producer.session = _PairedUvSession()
    generator = torch.Generator().manual_seed(13)

    def conditioning(request, group, request_generator, request_index):
        del request_generator
        uv = torch.tensor([[0.1, 0.2], [0.3, 0.4]])[: request.batch_size]
        delta = torch.tensor([0.01, 0.0])
        derivative = torch.zeros_like(uv)
        return (
            TrainingConditioning(
                "fixture.family",
                ("a" * 64,),
                {
                    "source_index": torch.zeros(request.batch_size, dtype=torch.int64),
                    "wo": torch.tensor([[0.0, 0.0, 1.0]]).expand(
                        request.batch_size, 3
                    ),
                    "uv": uv,
                    "uv_dx": derivative,
                    "uv_dy": derivative,
                    "paired_uv": uv + delta,
                    "paired_uv_dx": derivative,
                    "paired_uv_dy": derivative,
                },
                {
                    "route_name": request.name,
                    "request_index": request_index,
                    "reference_execution_group_id": group.group_id,
                },
            ),
            torch.tensor([[0.0, 0.0, 1.0]]).expand(request.batch_size, 3),
        )

    producer._conditioning = conditioning
    request = TrainingRouteRequest(
        "evaluator", "reference-evaluator", 2, 1, 0, 7, {}
    )
    group = SimpleNamespace(group_id="fixture-group")
    batch = producer._dispatch_evaluator_requests(
        (
            LogicalReferenceRequest(
                0,
                "dispatch",
                _EvaluatorLogicalRequest(request, group, generator, 0, "dispatch"),
                {},
            ),
        )
    )[0]
    assert len(producer.session.uvs) == 2
    torch.testing.assert_close(batch.target_f[:, 0, 0], torch.tensor([0.1, 0.3]))
    torch.testing.assert_close(
        batch.paired_target_f[:, 0, 0], torch.tensor([0.11, 0.31])
    )
    assert all(lease.released for lease in producer.session.leases)


def test_logical_request_rng_does_not_depend_on_execution_plan_identity() -> None:
    request = TrainingRouteRequest(
        "phase:evaluator", "reference-evaluator", 3, 1, 7, 19, {}
    )
    values = []
    for identity in ("baseline-plan", "packed-plan"):
        producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
        producer.device = torch.device("cpu")
        producer.query_stream_identity = identity
        producer._request_count = {}
        request_index, generator = producer._reserve_request(request)
        values.append((request_index, torch.rand(8, generator=generator)))
    assert values[0][0] == values[1][0] == 0
    torch.testing.assert_close(values[0][1], values[1][1])


def test_online_query_resume_rejects_typed_state_pool_drift_before_restoring_cursors() -> None:
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.query_stream_identity = "a" * 64
    producer.typed_state_pool_identity = "b" * 64
    producer._request_count = {"evaluator": 3}
    producer._group_cursor = {"evaluator": 2}
    producer._asset_tile_cursor = {"asset": 7}
    producer._reference_logical_id = 5
    producer._reference_scheduler = SimpleNamespace(assert_idle=lambda: None)
    state = producer.state_dict()
    changed = {**state, "typed_state_pool_identity": "c" * 64}
    with pytest.raises(ValueError, match="typed-state pool identity mismatch"):
        producer.load_state_dict(changed)
    assert producer._request_count == {"evaluator": 3}
    restored = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    restored.device = torch.device("cpu")
    restored.query_stream_identity = producer.query_stream_identity
    restored.typed_state_pool_identity = producer.typed_state_pool_identity
    restored._request_count = {}
    restored._group_cursor = {}
    restored._asset_tile_cursor = {}
    restored._reference_logical_id = 0
    restored._reference_scheduler = SimpleNamespace(assert_idle=lambda: None)
    restored.load_state_dict(state)
    assert restored._request_count == producer._request_count
    assert restored._group_cursor == producer._group_cursor
    assert restored._asset_tile_cursor == producer._asset_tile_cursor
    assert restored._reference_logical_id == producer._reference_logical_id


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


def test_group_block_schedule_v2_repeats_validation_groups_across_milestones() -> None:
    groups = tuple(SimpleNamespace(group_id=str(index)) for index in range(7))
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer._group_schedule_recipe = "group-block-balanced@2"
    producer._group_block_steps = 2
    producer._group_validation_offset_blocks = 1
    producer._group_sequence = groups
    producer.ddp_world_size = 2
    producer.ddp_rank = 1
    producer._group_cursor = {}

    def validation(step: int, index: int) -> TrainingRouteRequest:
        return TrainingRouteRequest(
            "validation:evaluator",
            "reference-evaluator",
            1,
            1,
            step,
            4,
            {"validation": True, "validation_group_index": index},
        )

    assert producer._select_group(validation(63, 0)) is groups[3]
    assert producer._select_group(validation(1024, 0)) is groups[3]
    assert producer._select_group(validation(1024, 1)) is groups[3]
    assert producer._select_group(validation(1024, 2)) is groups[5]
    missing_index = TrainingRouteRequest(
        "validation:evaluator",
        "reference-evaluator",
        1,
        1,
        63,
        4,
        {"validation": True},
    )
    with pytest.raises(ValueError, match="validation_group_index"):
        producer._select_group(missing_index)


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
