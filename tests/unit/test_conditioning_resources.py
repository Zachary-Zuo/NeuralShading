from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ncls.data import LogicalReferenceRequest
from ncls.learning.batches import TrainingConditioning, TrainingRouteRequest
from ncls.learning.conditioning_resources import ConditioningResource, ConditioningResources
from ncls.learning.producer import OnlineTrainingProducer, _EvaluatorLogicalRequest
from ncls.references.query import ScatteringQuery


class Lease:
    def __init__(self) -> None:
        self.releases = 0

    def release(self) -> None:
        self.releases += 1
        assert self.releases == 1


def conditioning(keys: tuple[str, ...], binding: list[int]):
    leases = [Lease() for _ in keys]
    resources = ConditioningResources([
        ConditioningResource(key, {"raw": torch.tensor([float(ord(key))])}, {}, lease)
        for key, lease in zip(keys, leases, strict=True)
    ])
    count = len(binding)
    return TrainingConditioning(
        "fixture", ("source",),
        {"source_index": torch.zeros(count, dtype=torch.int64), "wo": torch.ones(count, 3)},
        {"reference_execution_group_id": "group"}, resources,
        {"raw": torch.tensor(binding, dtype=torch.int64)},
    ), leases


def raw_rows(value: TrainingConditioning) -> list[float]:
    return [
        float(value.resources.entries[int(index)].tensors["raw"][0])
        for index in value.bindings["raw"]
    ]


def test_resource_binding_survives_reorder_dedup_and_independent_release() -> None:
    left, left_leases = conditioning(("a", "b"), [1, 0, 1])
    right, right_leases = conditioning(("b", "c"), [1, 0])
    combined = TrainingConditioning.concatenate((left, right))
    selected = combined.select_rows(torch.tensor([4, 1, 3, 0]))
    assert raw_rows(selected) == [98.0, 97.0, 99.0, 98.0]
    assert len(selected.resources) == 3
    left.release()
    right.release()
    combined.release()
    assert right_leases[0].releases == 1  # 同 key 的第二个 lease 已不再需要。
    assert all(lease.releases == 0 for lease in [*left_leases, right_leases[1]])
    assert raw_rows(selected) == [98.0, 97.0, 99.0, 98.0]
    selected.release()
    selected.release()
    assert all(lease.releases == 1 for lease in [*left_leases, *right_leases])


def test_conflicting_resource_identity_and_invalid_binding_fail() -> None:
    source, leases = conditioning(("a",), [0])
    conflict = ConditioningResources([
        ConditioningResource("a", {"raw": torch.ones(2)}, {})
    ])
    with pytest.raises(ValueError, match="conflicting metadata"):
        ConditioningResources.concatenate((source.resources, conflict))
    with pytest.raises(ValueError, match="outside resources"):
        TrainingConditioning(
            "fixture", ("source",), source.tensors, {},
            source.resources.retain(), {"raw": torch.tensor([1])},
        )
    conflict.release()
    source.release()
    assert leases[0].releases == 1


def test_empty_selection_keeps_a_valid_resource_owner() -> None:
    source, leases = conditioning(("a",), [0])
    empty = source.select_rows(torch.empty(0, dtype=torch.int64))
    source.release()
    assert empty.batch_size == 0 and leases[0].releases == 0
    empty.release()
    assert leases[0].releases == 1


@pytest.mark.parametrize("fail_paired", [False, True])
def test_rejection_and_paired_failure_release_all_resource_and_reference_leases(fail_paired) -> None:
    producer = OnlineTrainingProducer.__new__(OnlineTrainingProducer)
    producer.device = torch.device("cpu")
    producer.config = SimpleNamespace(online_query={})
    resource_leases, reference_leases = [], []
    counter = 0

    def adapt(request, group, generator, request_index):
        nonlocal counter
        result, leases = conditioning(("a", "b"), [index % 2 for index in range(request.batch_size)])
        resource_leases.extend(leases)
        result.tensors["row"] = torch.arange(counter, counter + request.batch_size)
        result.tensors["paired_uv"] = torch.zeros(request.batch_size, 2)
        result.tensors["paired_uv_dx"] = torch.zeros(request.batch_size, 2)
        result.tensors["paired_uv_dy"] = torch.zeros(request.batch_size, 2)
        counter += request.batch_size
        return result, result.tensors["wo"]

    calls = 0

    def evaluate(query, wi, seeds, **kwargs):
        nonlocal calls
        calls += 1
        if fail_paired and calls == 2:
            raise RuntimeError("paired reference failed")
        lease = Lease()
        reference_leases.append(lease)
        valid = torch.ones(query.batch_size, 1, dtype=torch.bool)
        if calls <= 2:
            valid[1::2] = False
        return SimpleNamespace(valid=valid, f=torch.ones_like(wi), lease=lease)

    producer._conditioning = adapt
    producer.session = SimpleNamespace(evaluate=evaluate)
    request = TrainingRouteRequest("evaluator", "reference-evaluator", 4, 1, 0, 7)
    packed = (LogicalReferenceRequest(
        0, "dispatch", _EvaluatorLogicalRequest(
            request, SimpleNamespace(group_id="group"), torch.Generator().manual_seed(9), 0, "dispatch",
        ), {},
    ),)
    if fail_paired:
        with pytest.raises(RuntimeError, match="paired reference failed"):
            producer._dispatch_evaluator_requests(packed)
    else:
        batch = producer._dispatch_evaluator_requests(packed)[0]
        assert batch.conditioning.tensors["row"].tolist() == [0, 2, 4, 5]
        assert raw_rows(batch.conditioning) == [97.0, 97.0, 97.0, 98.0]
        assert batch.provenance["rejected_count"] == 2
        batch.release()
    assert all(lease.releases == 1 for lease in resource_leases + reference_leases)


def test_filter_random_has_explicit_shape_and_range() -> None:
    arguments = (torch.zeros(2, dtype=torch.int64), torch.ones(2, 3), "group")
    query = ScatteringQuery(*arguments, filter_random=torch.tensor([0.0, 0.75]))
    assert query.filter_random.tolist() == [0.0, 0.75]
    for value in (torch.tensor([0.0, 1.0]), torch.tensor([0.0, float("nan")])):
        with pytest.raises(ValueError, match=r"\[0,1\)"):
            ScatteringQuery(*arguments, filter_random=value)
