from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ncls.learning.batches import TrainingConditioning, TrainingRouteRequest
from ncls.learning.producer import OnlineTrainingProducer


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
