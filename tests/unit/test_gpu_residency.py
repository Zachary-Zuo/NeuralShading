import pytest

from ncls.core.identity import sha256_json
from ncls.data import (
    GpuResidencyManager,
    PipelineTrace,
    ResidentAllocation,
    ResidencyCapacityError,
    ResidencyKey,
)


def _key(name: str) -> ResidencyKey:
    return ResidencyKey(sha256_json(name), "torch-decoded-mip", "cuda:0")


def test_gpu_residency_reuses_materialization_and_evicts_lru_by_bytes() -> None:
    released: list[str] = []
    loads: list[str] = []
    manager: GpuResidencyManager[str] = GpuResidencyManager(10)

    def acquire(name: str, size: int):
        return manager.acquire(
            _key(name),
            estimated_bytes=size,
            materialize=lambda: (
                loads.append(name)
                or ResidentAllocation(name, size, lambda value: released.append(value))
            ),
        )

    first = acquire("a", 6)
    first.release()
    hit = acquire("a", 6)
    hit.release()
    second = acquire("b", 5)
    assert second.value == "b"
    second.release()

    assert loads == ["a", "b"]
    assert released == ["a"]
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 5
    assert snapshot["trace"]["counters"]["residency.hit"] == 1
    assert snapshot["trace"]["counters"]["residency.evict_bytes"] == 6
    manager.close()
    assert released == ["a", "b"]


def test_gpu_residency_never_evicts_active_lease_or_silently_materializes() -> None:
    materialized = False
    manager: GpuResidencyManager[str] = GpuResidencyManager(8)
    active = manager.acquire(
        _key("active"),
        estimated_bytes=8,
        materialize=lambda: ResidentAllocation("active", 8),
    )

    def materialize() -> ResidentAllocation[str]:
        nonlocal materialized
        materialized = True
        return ResidentAllocation("late", 1)

    with pytest.raises(ResidencyCapacityError, match="all resources are leased"):
        manager.acquire(_key("late"), estimated_bytes=1, materialize=materialize)
    assert not materialized
    with pytest.raises(RuntimeError, match="active leases"):
        manager.close()
    active.release()
    manager.close()


def test_gpu_residency_rejects_oversize_before_materialization() -> None:
    manager: GpuResidencyManager[object] = GpuResidencyManager(4)
    with pytest.raises(ResidencyCapacityError, match="above budget"):
        manager.acquire(
            _key("large"),
            estimated_bytes=5,
            materialize=lambda: pytest.fail("oversize resource must not materialize"),
        )
    manager.close()


def test_pipeline_trace_reset_keeps_snapshot_values_stable() -> None:
    trace = PipelineTrace()
    trace.increment("cache.hit", 2)
    trace.gauge("queue.depth", 2)
    trace.gauge("queue.depth", 1)
    with trace.measure("host.decode"):
        pass
    snapshot = trace.snapshot(reset=True)
    assert snapshot.counters == {"cache.hit": 2}
    assert snapshot.gauges == {"queue.depth": 1.0}
    assert snapshot.peaks == {"queue.depth": 2.0}
    assert snapshot.duration_seconds["host.decode"] >= 0.0
    assert trace.snapshot().to_dict() == {
        "counters": {},
        "gauges": {},
        "peaks": {},
        "duration_seconds": {},
    }
