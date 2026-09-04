from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

from .contracts import OnlineBatch, OnlineProducer, OnlineStepRequest
from .tracing import PipelineTrace


@dataclass
class _ProductionLifecycle:
    producer: OnlineProducer
    remaining: int
    iteration_ended: bool

    def release_one(self) -> None:
        if self.remaining < 1:
            raise RuntimeError("online step production lifecycle underflow")
        self.remaining -= 1
        if self.remaining == 0 and not self.iteration_ended:
            self.producer.end_iteration()
            self.iteration_ended = True


@dataclass
class _ReadyStep:
    request: OnlineStepRequest
    batches: dict[str, OnlineBatch]
    lifecycle: _ProductionLifecycle


class OnlineStepBatch:
    """Consumer lease for one ordered step in the ready ring."""

    def __init__(
        self,
        ready: _ReadyStep,
        *,
        consumer_wait_seconds: float,
        on_release: Callable[[int], None],
    ) -> None:
        self.logical_id = ready.request.logical_id
        self.batches = ready.batches
        self.consumer_wait_seconds = float(consumer_wait_seconds)
        self._lifecycle = ready.lifecycle
        self._on_release = on_release
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        error: BaseException | None = None
        try:
            for batch in reversed(tuple(self.batches.values())):
                try:
                    batch.release()
                except BaseException as caught:
                    if error is None:
                        error = caught
        finally:
            try:
                self._lifecycle.release_one()
            finally:
                self._on_release(self.logical_id)
        if error is not None:
            raise error

    def __enter__(self) -> "OnlineStepBatch":
        if self._released:
            raise RuntimeError("online step batch has already been released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class PipelineOnlineDataSession:
    """Bounded rank-owned step queue and ready ring for online training."""

    def __init__(
        self,
        producer: OnlineProducer,
        *,
        execution_plan_identity: str,
        ready_capacity: int,
        production_batch_steps: int,
    ) -> None:
        if not isinstance(producer, OnlineProducer):
            raise TypeError("pipeline data session requires an OnlineProducer")
        if not execution_plan_identity:
            raise ValueError("data execution plan identity is required")
        if int(ready_capacity) < 1 or int(production_batch_steps) < 1:
            raise ValueError("data ready and production capacities must be positive")
        if int(production_batch_steps) > int(ready_capacity):
            raise ValueError("production batch steps cannot exceed ready capacity")
        self._producer = producer
        self._execution_plan_identity = execution_plan_identity
        self._ready_capacity = int(ready_capacity)
        self._production_batch_steps = int(production_batch_steps)
        self._pending: deque[OnlineStepRequest] = deque()
        self._ready: deque[_ReadyStep] = deque()
        self._acquired: dict[int, OnlineStepBatch] = {}
        self._next_logical_id = 0
        self._consumed_steps = 0
        self._consumed_batches = 0
        self._trace = PipelineTrace()
        self._cancelled = False
        self._closed = False

    @property
    def consumed_batches(self) -> int:
        return self._consumed_batches

    @property
    def submission_capacity(self) -> int:
        return self._ready_capacity

    @property
    def production_batch_steps(self) -> int:
        return self._production_batch_steps

    @property
    def device(self) -> Any:
        return self._producer.device

    @property
    def reference_program_identity(self) -> str:
        return str(self._producer.reference_program_identity)

    @property
    def reference_execution_plan_identity(self) -> str:
        return str(self._producer.reference_execution_plan_identity)

    @property
    def native_asset_collection_identity(self) -> str:
        return str(self._producer.native_asset_collection_identity)

    @property
    def query_stream_identity(self) -> str:
        return str(self._producer.query_stream_identity)

    @property
    def source_contracts(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._producer.source_contracts)

    @property
    def source_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(self._producer.source_snapshot_ids)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("online data session is closed")

    def _depth(self) -> int:
        return len(self._pending) + len(self._ready) + len(self._acquired)

    def _publish_depth(self) -> None:
        self._trace.gauge("pending_depth", len(self._pending))
        self._trace.gauge("ready_depth", len(self._ready))
        self._trace.gauge("acquired_depth", len(self._acquired))

    def submit_step(self, routes: Mapping[str, Any], *, boundary_id: str) -> int:
        self._require_open()
        if self._depth() >= self._ready_capacity:
            self._trace.increment("backpressure")
            raise RuntimeError(
                f"online data ready ring reached capacity {self._ready_capacity}"
            )
        logical_id = self._next_logical_id
        request = OnlineStepRequest(logical_id, boundary_id, routes)
        try:
            self._producer.prefetch_steps((request,))
        except BaseException:
            self._cancelled = True
            raise
        self._next_logical_id += 1
        self._pending.append(request)
        self._trace.increment("submitted_steps")
        self._publish_depth()
        return logical_id

    @staticmethod
    def _release_batches(batches: Mapping[str, OnlineBatch]) -> None:
        for batch in reversed(tuple(batches.values())):
            batch.release()

    def _pump(self) -> int:
        if not self._pending:
            return 0
        first = self._pending[0]
        count = min(
            self._production_batch_steps,
            self._ready_capacity - len(self._ready) - len(self._acquired),
            len(self._pending),
        )
        requests: list[OnlineStepRequest] = []
        for request in tuple(self._pending)[:count]:
            if request.boundary_id != first.boundary_id:
                break
            requests.append(request)
        if not requests:
            return 0
        started = time.perf_counter_ns()
        produced_value = self._producer.produce_steps(tuple(requests))
        produced = tuple(dict(value) for value in produced_value)
        self._trace.add_duration_ns("produce_steps", time.perf_counter_ns() - started)
        if len(produced) != len(requests):
            for batches in produced:
                self._release_batches(batches)
            raise RuntimeError("online producer result count disagrees with step requests")
        expected_names = [
            set(request.routes)
            for request in requests
        ]
        for names, batches in zip(expected_names, produced, strict=True):
            if set(batches) != names:
                for value in produced:
                    self._release_batches(value)
                raise RuntimeError("online producer returned the wrong route batch set")
        detached = all(
            getattr(batch, "lease", None) is None
            for batches in produced
            for batch in batches.values()
        )
        lifecycle = _ProductionLifecycle(self._producer, len(produced), detached)
        if detached:
            self._producer.end_iteration()
        for request, batches in zip(requests, produced, strict=True):
            self._pending.popleft()
            self._ready.append(_ReadyStep(request, batches, lifecycle))
        self._trace.increment("production_dispatches")
        self._trace.increment("produced_steps", len(produced))
        self._trace.gauge("last_production_batch_steps", len(produced))
        self._publish_depth()
        return len(produced)

    def acquire_step(self, logical_id: int) -> OnlineStepBatch:
        self._require_open()
        if self._acquired:
            raise RuntimeError("online data session permits one acquired step at a time")
        started = time.perf_counter_ns()
        if not self._ready:
            self._trace.increment("consumer_starvation")
            self._pump()
        wait_ns = time.perf_counter_ns() - started
        self._trace.add_duration_ns("consumer_wait", wait_ns)
        if not self._ready:
            raise RuntimeError("online data session has no ready step")
        ready = self._ready[0]
        if ready.request.logical_id != int(logical_id):
            raise RuntimeError("online data session acquire order is not deterministic")
        self._ready.popleft()
        lease = OnlineStepBatch(
            ready,
            consumer_wait_seconds=wait_ns / 1_000_000_000.0,
            on_release=self._release_acquired,
        )
        self._acquired[lease.logical_id] = lease
        self._consumed_steps += 1
        self._consumed_batches += len(lease.batches)
        self._trace.increment("consumed_steps")
        self._publish_depth()
        return lease

    def _release_acquired(self, logical_id: int) -> None:
        if self._acquired.pop(logical_id, None) is None:
            raise RuntimeError("online data step release does not match an acquired lease")
        self._trace.increment("released_steps")
        self._publish_depth()

    def _require_idle_boundary(self) -> None:
        if self._pending or self._ready or self._acquired:
            raise RuntimeError(
                "online data boundary requires no pending, ready or acquired step"
            )

    def state_dict(self) -> Mapping[str, Any]:
        self._require_open()
        self._require_idle_boundary()
        if self._cancelled:
            raise RuntimeError("cancelled online data work cannot be checkpointed")
        return {
            "format_name": "ncls.pipeline-online-data-session",
            "format_version": 1,
            "execution_plan_identity": self._execution_plan_identity,
            "next_logical_id": self._next_logical_id,
            "consumed_steps": self._consumed_steps,
            "consumed_batches": self._consumed_batches,
            "producer": self._producer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._require_open()
        self._require_idle_boundary()
        required = {
            "format_name",
            "format_version",
            "execution_plan_identity",
            "next_logical_id",
            "consumed_steps",
            "consumed_batches",
            "producer",
        }
        if set(state) != required:
            raise ValueError("online data session state fields are invalid")
        if (
            state["format_name"] != "ncls.pipeline-online-data-session"
            or int(state["format_version"]) != 1
        ):
            raise ValueError("unsupported online data session state format")
        if state["execution_plan_identity"] != self._execution_plan_identity:
            raise ValueError("online data session execution plan identity mismatch")
        next_logical_id = int(state["next_logical_id"])
        consumed_steps = int(state["consumed_steps"])
        consumed_batches = int(state["consumed_batches"])
        if min(next_logical_id, consumed_steps, consumed_batches) < 0:
            raise ValueError("online data session cursor is invalid")
        if next_logical_id != consumed_steps:
            raise ValueError("online data session checkpoint cursor is ahead of consumption")
        producer_state = state["producer"]
        if not isinstance(producer_state, Mapping):
            raise ValueError("online data session producer state must be an object")
        self._producer.load_state_dict(producer_state)
        self._next_logical_id = next_logical_id
        self._consumed_steps = consumed_steps
        self._consumed_batches = consumed_batches

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]:
        self._require_open()
        result = {
            str(name): float(value)
            for name, value in self._producer.profile_snapshot(reset=reset).items()
        }
        trace = self._trace.snapshot(reset=reset).to_dict()
        for category, values in trace.items():
            for name, value in values.items():
                result[f"data/session/{category}/{name}"] = float(value)
        return result

    def drain(self) -> None:
        self._require_open()
        self._require_idle_boundary()

    def cancel_pending(self) -> None:
        self._require_open()
        if self._acquired:
            raise RuntimeError("cannot cancel online data work with an acquired step")
        discarded = len(self._pending) + len(self._ready)
        self._pending.clear()
        while self._ready:
            ready = self._ready.popleft()
            self._release_batches(ready.batches)
            ready.lifecycle.release_one()
        self._cancelled = True
        self._trace.increment("cancelled_steps", discarded)
        self._publish_depth()

    def native_assets(self) -> Any:
        self._require_open()
        return self._producer.native_assets()

    def close(self) -> None:
        if self._closed:
            return
        if self._acquired:
            raise RuntimeError("cannot close online data session with an acquired step")
        self._pending.clear()
        while self._ready:
            ready = self._ready.popleft()
            self._release_batches(ready.batches)
            ready.lifecycle.release_one()
        self._producer.close()
        self._closed = True

    def __enter__(self) -> "PipelineOnlineDataSession":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["OnlineStepBatch", "PipelineOnlineDataSession"]
