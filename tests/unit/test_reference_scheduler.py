from dataclasses import dataclass

import pytest

from ncls.data import LogicalReferenceRequest, ReferenceScheduler
from ncls.references import ReferenceConcurrencyCapability


@dataclass
class _Value:
    value: int
    releases: list[int]

    def release(self) -> None:
        self.releases.append(self.value)


def _run(batch_steps: int):
    calls: list[tuple[int, ...]] = []
    releases: list[int] = []

    def dispatch(requests):
        calls.append(tuple(item.logical_id for item in requests))
        return tuple(_Value(item.payload * 2, releases) for item in requests)

    scheduler = ReferenceScheduler(
        dispatch,
        capability=ReferenceConcurrencyCapability("global", False, 2, False),
        batch_steps=batch_steps,
        ready_capacity=4,
        maximum_inflight=1,
    )
    for index in range(4):
        scheduler.submit(
            LogicalReferenceRequest(index, "group-a", index + 10, {"seed": 100 + index})
        )
    results = []
    while len(results) < 4:
        result = scheduler.next_result()
        results.append((result.logical_id, result.payload.value, result.provenance["seed"]))
        result.release()
    scheduler.assert_idle()
    profile = scheduler.profile_snapshot()
    scheduler.close()
    return calls, releases, results, profile


def test_packed_reference_dispatch_preserves_baseline_order_and_identity() -> None:
    baseline = _run(1)
    packed = _run(4)
    assert baseline[2] == packed[2]
    assert baseline[0] == [(0,), (1,), (2,), (3,)]
    assert packed[0] == [(0, 1, 2, 3)]
    assert baseline[1] == packed[1] == [20, 22, 24, 26]
    assert packed[3]["counters"]["reference.dispatches"] == 1


def test_reference_scheduler_never_packs_across_execution_group() -> None:
    calls = []

    def dispatch(requests):
        calls.append(tuple(item.execution_group_id for item in requests))
        return tuple(item.payload for item in requests)

    scheduler = ReferenceScheduler(
        dispatch,
        capability=ReferenceConcurrencyCapability("stream-fence", True, 3, False),
        batch_steps=4,
        ready_capacity=3,
        maximum_inflight=2,
    )
    scheduler.submit(LogicalReferenceRequest(0, "a", 0, {}))
    scheduler.submit(LogicalReferenceRequest(1, "a", 1, {}))
    scheduler.submit(LogicalReferenceRequest(2, "b", 2, {}))
    scheduler.pump()
    values = []
    for _ in range(3):
        result = scheduler.next_result()
        values.append(result.payload)
        result.release()
        scheduler.pump()
    assert values == [0, 1, 2]
    assert calls == [("a", "a"), ("b",)]
    scheduler.close()


def test_reference_scheduler_applies_ready_ring_and_lease_boundaries() -> None:
    scheduler = ReferenceScheduler(
        lambda requests: tuple(item.payload for item in requests),
        capability=ReferenceConcurrencyCapability("global", False, 1, False),
        batch_steps=1,
        ready_capacity=1,
        maximum_inflight=1,
    )
    scheduler.submit(LogicalReferenceRequest(0, "a", "first", {}))
    with pytest.raises(RuntimeError, match="capacity"):
        scheduler.submit(LogicalReferenceRequest(1, "a", "second", {}))
    result = scheduler.next_result()
    with pytest.raises(RuntimeError, match="active result leases"):
        scheduler.close()
    result.release()
    scheduler.assert_idle()
    scheduler.close()


def test_reference_concurrency_is_derived_from_backend_api_not_host_os() -> None:
    vulkan = ReferenceConcurrencyCapability.for_device_api("vulkan")
    d3d12 = ReferenceConcurrencyCapability.for_device_api("d3d12")
    assert vulkan.synchronization == "global"
    assert not vulkan.supports_async_submit
    assert vulkan.maximum_inflight == 1
    assert d3d12.synchronization == "stream-fence"
    assert d3d12.supports_async_submit
    assert d3d12.maximum_inflight == 2
    with pytest.raises(ValueError, match="globally synchronized"):
        ReferenceConcurrencyCapability("global", True, 1, False)
