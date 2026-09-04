from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Generic, TypeVar

from ncls.core.identity import require_sha256

from .tracing import PipelineTrace


T = TypeVar("T")


@dataclass(frozen=True, order=True)
class ResidencyKey:
    resource_identity: str
    representation: str
    device: str

    def __post_init__(self) -> None:
        require_sha256("resident resource identity", self.resource_identity)
        if not self.representation or not self.device:
            raise ValueError("resident representation and device are required")


@dataclass(frozen=True)
class ResidentAllocation(Generic[T]):
    value: T
    allocated_bytes: int
    on_evict: Callable[[T], None] | None = None

    def __post_init__(self) -> None:
        if self.allocated_bytes < 1:
            raise ValueError("resident allocation must report positive allocated bytes")


@dataclass
class _Entry(Generic[T]):
    allocation: ResidentAllocation[T]
    leases: int
    last_use: int


class ResidencyCapacityError(RuntimeError):
    pass


class ResidencyLease(Generic[T]):
    def __init__(
        self,
        manager: "GpuResidencyManager[T]",
        key: ResidencyKey,
        value: T,
    ) -> None:
        self._manager = manager
        self.key = key
        self.value = value
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release(self.key)

    def __enter__(self) -> T:
        if self._released:
            raise RuntimeError("residency lease has already been released")
        return self.value

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class GpuResidencyManager(Generic[T]):
    """Rank-local byte-budgeted GPU resource cache with strict leases."""

    def __init__(self, budget_bytes: int, *, trace: PipelineTrace | None = None) -> None:
        if int(budget_bytes) < 1:
            raise ValueError("GPU residency budget must be positive")
        self._budget_bytes = int(budget_bytes)
        self._trace = trace if trace is not None else PipelineTrace()
        self._entries: dict[ResidencyKey, _Entry[T]] = {}
        self._allocated_bytes = 0
        self._clock = 0
        self._closed = False
        self._lock = RLock()

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._allocated_bytes

    def is_resident(self, key: ResidencyKey) -> bool:
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU residency manager is closed")
            return key in self._entries

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _describe_largest(self) -> str:
        largest = sorted(
            self._entries.items(),
            key=lambda item: item[1].allocation.allocated_bytes,
            reverse=True,
        )[:3]
        if not largest:
            return "none"
        return ", ".join(
            f"{key.representation}@{key.device}="
            f"{entry.allocation.allocated_bytes}B/leases={entry.leases}"
            for key, entry in largest
        )

    def _require_capacity(self, required_bytes: int) -> None:
        if required_bytes > self._budget_bytes:
            raise ResidencyCapacityError(
                f"resident resource requires {required_bytes} bytes, above budget "
                f"{self._budget_bytes}; largest resident resources: {self._describe_largest()}"
            )
        while self._allocated_bytes + required_bytes > self._budget_bytes:
            candidates = [
                (key, entry)
                for key, entry in self._entries.items()
                if entry.leases == 0
            ]
            if not candidates:
                raise ResidencyCapacityError(
                    f"GPU residency needs {required_bytes} bytes with "
                    f"{self._allocated_bytes}/{self._budget_bytes} bytes allocated, but all "
                    f"resources are leased; largest resident resources: {self._describe_largest()}"
                )
            key, entry = min(candidates, key=lambda item: item[1].last_use)
            self._evict(key, entry)

    def _evict(self, key: ResidencyKey, entry: _Entry[T]) -> None:
        if entry.leases:
            raise RuntimeError("cannot evict an actively leased GPU resource")
        del self._entries[key]
        self._allocated_bytes -= entry.allocation.allocated_bytes
        callback = entry.allocation.on_evict
        if callback is not None:
            callback(entry.allocation.value)
        self._trace.increment("residency.evict")
        self._trace.increment("residency.evict_bytes", entry.allocation.allocated_bytes)
        self._publish_gauges()

    def _publish_gauges(self) -> None:
        self._trace.gauge("residency.allocated_bytes", self._allocated_bytes)
        self._trace.gauge("residency.entries", len(self._entries))

    def acquire(
        self,
        key: ResidencyKey,
        *,
        estimated_bytes: int,
        materialize: Callable[[], ResidentAllocation[T]],
    ) -> ResidencyLease[T]:
        """Acquire a resource; `materialize` runs only on a cache miss.

        This method intentionally serializes materialization. GPU object creation remains on
        the rank owner thread, while host decode/read parallelism belongs to HostPipeline.
        """

        estimate = int(estimated_bytes)
        if estimate < 1:
            raise ValueError("resident resource byte estimate must be positive")
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU residency manager is closed")
            cached = self._entries.get(key)
            if cached is not None:
                cached.leases += 1
                cached.last_use = self._tick()
                self._trace.increment("residency.hit")
                return ResidencyLease(self, key, cached.allocation.value)

            self._trace.increment("residency.miss")
            self._require_capacity(estimate)
            allocation = materialize()
            if not isinstance(allocation, ResidentAllocation):
                raise TypeError("GPU residency materializer must return ResidentAllocation")
            actual = allocation.allocated_bytes
            try:
                self._require_capacity(actual)
            except BaseException:
                if allocation.on_evict is not None:
                    allocation.on_evict(allocation.value)
                raise
            self._entries[key] = _Entry(allocation, 1, self._tick())
            self._allocated_bytes += actual
            self._trace.increment("residency.materialize")
            self._trace.increment("residency.materialize_bytes", actual)
            self._publish_gauges()
            return ResidencyLease(self, key, allocation.value)

    def _release(self, key: ResidencyKey) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.leases < 1:
                raise RuntimeError("GPU residency lease state is invalid")
            entry.leases -= 1
            entry.last_use = self._tick()
            self._trace.increment("residency.release")

    def remove(self, key: ResidencyKey) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.leases:
                raise RuntimeError("cannot remove an actively leased GPU resource")
            self._evict(key, entry)
            return True

    def snapshot(self, *, reset_trace: bool = False) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_bytes": self._budget_bytes,
                "allocated_bytes": self._allocated_bytes,
                "entry_count": len(self._entries),
                "active_leases": sum(entry.leases for entry in self._entries.values()),
                "entries": [
                    {
                        "resource_identity": key.resource_identity,
                        "representation": key.representation,
                        "device": key.device,
                        "allocated_bytes": entry.allocation.allocated_bytes,
                        "leases": entry.leases,
                    }
                    for key, entry in sorted(self._entries.items())
                ],
                "trace": self._trace.snapshot(reset=reset_trace).to_dict(),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            active = sum(entry.leases for entry in self._entries.values())
            if active:
                raise RuntimeError(
                    f"cannot close GPU residency manager with {active} active leases"
                )
            for key, entry in sorted(
                tuple(self._entries.items()), key=lambda item: item[1].last_use
            ):
                self._evict(key, entry)
            self._closed = True

    def __enter__(self) -> "GpuResidencyManager[T]":
        if self._closed:
            raise RuntimeError("GPU residency manager is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "GpuResidencyManager",
    "ResidentAllocation",
    "ResidencyCapacityError",
    "ResidencyKey",
    "ResidencyLease",
]
