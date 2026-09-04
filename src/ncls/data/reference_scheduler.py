from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar, cast

from ncls.references.backend import ReferenceConcurrencyCapability

from .tracing import PipelineTrace


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class LogicalReferenceRequest(Generic[T]):
    logical_id: int
    execution_group_id: str
    payload: T
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.logical_id < 0 or not self.execution_group_id:
            raise ValueError("logical reference request identity is invalid")
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass
class _DispatchLease:
    remaining: int
    on_release: Callable[[], None] | None
    scheduler: "ReferenceScheduler[Any, Any]"
    released: bool = False

    def release_one(self) -> None:
        if self.remaining < 1:
            raise RuntimeError("reference dispatch lease was released too many times")
        self.remaining -= 1
        if self.remaining:
            return
        if self.released:
            raise RuntimeError("reference dispatch lease was already released")
        self.released = True
        if self.on_release is not None:
            self.on_release()
        self.scheduler._release_dispatch()


@dataclass
class ScheduledReferenceResult(Generic[R]):
    logical_id: int
    payload: R
    provenance: Mapping[str, Any]
    _lease: _DispatchLease
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        release = getattr(self.payload, "release", None)
        if callable(release):
            release()
        self._lease.release_one()

    def __enter__(self) -> R:
        if self._released:
            raise RuntimeError("scheduled reference result has already been released")
        return self.payload

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class ReferenceScheduler(Generic[T, R]):
    """Bounded rank-owner scheduler for packed logical reference dispatches."""

    def __init__(
        self,
        dispatcher: Callable[
            [tuple[LogicalReferenceRequest[T], ...]],
            Sequence[R] | tuple[Sequence[R], Callable[[], None]],
        ],
        *,
        capability: ReferenceConcurrencyCapability,
        batch_steps: int,
        ready_capacity: int,
        maximum_inflight: int,
        trace: PipelineTrace | None = None,
    ) -> None:
        if int(batch_steps) < 1 or int(ready_capacity) < 1:
            raise ValueError("reference batch and ready capacities must be positive")
        if not 1 <= int(maximum_inflight) <= capability.maximum_inflight:
            raise ValueError("reference inflight count exceeds backend capability")
        self._dispatcher = dispatcher
        self.capability = capability
        self.batch_steps = int(batch_steps)
        self.ready_capacity = int(ready_capacity)
        self.maximum_inflight = int(maximum_inflight)
        self._trace = trace if trace is not None else PipelineTrace()
        self._pending: deque[LogicalReferenceRequest[T]] = deque()
        self._ready: deque[ScheduledReferenceResult[R]] = deque()
        self._active_dispatches = 0
        self._outstanding_results: dict[int, ScheduledReferenceResult[R]] = {}
        self._last_submitted = -1
        self._closed = False

    @property
    def queued_requests(self) -> int:
        return len(self._pending) + len(self._ready)

    @property
    def active_dispatches(self) -> int:
        return self._active_dispatches

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reference scheduler is closed")

    def _publish_depth(self) -> None:
        self._trace.gauge("reference.pending_depth", len(self._pending))
        self._trace.gauge("reference.ready_depth", len(self._ready))
        self._trace.gauge("reference.inflight", self._active_dispatches)

    def submit(self, request: LogicalReferenceRequest[T]) -> None:
        self._require_open()
        if request.logical_id <= self._last_submitted:
            raise ValueError("reference logical IDs must be strictly increasing")
        if self.queued_requests >= self.ready_capacity:
            self._trace.increment("reference.backpressure")
            raise RuntimeError(
                f"reference ready ring reached capacity {self.ready_capacity}"
            )
        self._pending.append(request)
        self._last_submitted = request.logical_id
        self._trace.increment("reference.submitted")
        self._publish_depth()

    def _pack(self) -> tuple[LogicalReferenceRequest[T], ...]:
        first = self._pending[0]
        count = min(
            self.batch_steps,
            self.ready_capacity - len(self._ready),
            len(self._pending),
        )
        packed: list[LogicalReferenceRequest[T]] = []
        for request in tuple(self._pending)[:count]:
            if request.execution_group_id != first.execution_group_id:
                break
            packed.append(request)
        return tuple(packed)

    def pump(self) -> int:
        self._require_open()
        dispatched = 0
        while (
            self._pending
            and len(self._ready) < self.ready_capacity
            and self._active_dispatches < self.maximum_inflight
        ):
            packed = self._pack()
            if not packed:
                break
            with self._trace.measure("reference.dispatch"):
                dispatched_value = self._dispatcher(packed)
            if (
                isinstance(dispatched_value, tuple)
                and len(dispatched_value) == 2
                and callable(dispatched_value[1])
            ):
                values = tuple(dispatched_value[0])
                on_release = dispatched_value[1]
            else:
                values = tuple(cast(Sequence[R], dispatched_value))
                on_release = None
            if len(values) != len(packed):
                for value in values:
                    release = getattr(value, "release", None)
                    if callable(release):
                        release()
                if on_release is not None:
                    on_release()
                raise RuntimeError("reference dispatcher result count disagrees with packed input")
            self._active_dispatches += 1
            dispatch_lease = _DispatchLease(len(packed), on_release, self)
            for request, value in zip(packed, values, strict=True):
                candidate = ScheduledReferenceResult(
                    request.logical_id,
                    value,
                    request.provenance,
                    dispatch_lease,
                )
                self._ready.append(candidate)
                self._outstanding_results[request.logical_id] = candidate
                self._pending.popleft()
            dispatched += len(packed)
            self._trace.increment("reference.dispatches")
            self._trace.increment("reference.logical_steps", len(packed))
            self._trace.gauge("reference.last_pack_steps", len(packed))
            self._publish_depth()
        return dispatched

    def next_result(self) -> ScheduledReferenceResult[R]:
        self._require_open()
        if not self._ready:
            self.pump()
        if not self._ready:
            if self._pending and self._active_dispatches >= self.maximum_inflight:
                raise RuntimeError(
                    "reference scheduler is waiting for an unreleased dispatch lease"
                )
            raise RuntimeError("reference scheduler has no ready result")
        result = self._ready.popleft()
        self._trace.increment("reference.consumed")
        self._publish_depth()
        return result

    def _release_dispatch(self) -> None:
        if self._active_dispatches < 1:
            raise RuntimeError("reference scheduler dispatch accounting underflow")
        self._active_dispatches -= 1
        released = [
            logical_id
            for logical_id, result in self._outstanding_results.items()
            if result._released
        ]
        for logical_id in released:
            del self._outstanding_results[logical_id]
        self._trace.increment("reference.released_dispatches")
        self._publish_depth()

    def discard_boundary(self) -> None:
        self._require_open()
        discarded = len(self._pending) + len(self._ready)
        self._pending.clear()
        while self._ready:
            self._ready.popleft().release()
        if self._outstanding_results:
            raise RuntimeError(
                "cannot cross reference boundary with results held by the consumer"
            )
        self._last_submitted = -1
        self._trace.increment("reference.discarded", discarded)
        self._publish_depth()

    def assert_idle(self) -> None:
        self._require_open()
        if self._pending or self._ready or self._outstanding_results or self._active_dispatches:
            raise RuntimeError("reference scheduler boundary requires no pending or leased result")

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, Any]:
        self._require_open()
        return self._trace.snapshot(reset=reset).to_dict()

    def close(self) -> None:
        if self._closed:
            return
        if self._outstanding_results and any(
            not result._released for result in self._outstanding_results.values()
        ):
            raise RuntimeError("cannot close reference scheduler with active result leases")
        self._pending.clear()
        while self._ready:
            self._ready.popleft().release()
        self._closed = True


__all__ = [
    "LogicalReferenceRequest",
    "ReferenceScheduler",
    "ScheduledReferenceResult",
]
