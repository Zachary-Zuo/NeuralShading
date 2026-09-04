from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
import pickle
import queue
import time
import traceback
from typing import Any, Callable, Mapping

from .tracing import PipelineTrace


@dataclass(frozen=True)
class HostRequest:
    logical_id: int
    payload: Any
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.logical_id < 0:
            raise ValueError("host request logical ID must be nonnegative")
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class HostResult:
    logical_id: int
    payload: Any
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class HostWorkerStart:
    method: str = "spawn"

    def __post_init__(self) -> None:
        if self.method not in {"spawn", "forkserver"}:
            raise ValueError(
                "host workers require spawn or forkserver; fork may inherit CUDA/Falcor owners"
            )
        if self.method not in mp.get_all_start_methods():
            raise ValueError(f"host worker start method {self.method!r} is unavailable")

    def context(self) -> BaseContext:
        return mp.get_context(self.method)


class HostPipelineError(RuntimeError):
    pass


class HostPipelineBackpressure(HostPipelineError):
    pass


class HostWorkerError(HostPipelineError):
    def __init__(
        self,
        *,
        stage: str,
        logical_id: int | None,
        rank: int,
        detail: str,
    ) -> None:
        request = "unknown" if logical_id is None else str(logical_id)
        super().__init__(
            f"host stage {stage!r} failed for logical request {request} on rank {rank}: "
            f"{detail}"
        )
        self.stage = stage
        self.logical_id = logical_id
        self.rank = rank
        self.detail = detail


_STOP = ("stop",)


def _host_worker_main(
    processor: Callable[[Any], Any],
    task_queue: Queue,
    result_queue: Queue,
    error_queue: Queue,
    stage: str,
    rank: int,
) -> None:
    while True:
        task = task_queue.get()
        if task == _STOP:
            return
        logical_id, payload, provenance = task
        try:
            started = time.perf_counter_ns()
            value = processor(payload)
            duration_ns = time.perf_counter_ns() - started
            result_queue.put((logical_id, value, provenance, duration_ns))
        except BaseException as error:
            error_queue.put(
                (
                    logical_id,
                    type(error).__name__,
                    str(error),
                    traceback.format_exc(),
                )
            )


class HostPipeline:
    """Bounded deterministic CPU/host stage with persistent spawned workers.

    Only the supplied serializable processor executes in child processes. CUDA, Falcor,
    residency materialization, reference dispatch and ready-batch ownership stay in the
    rank process.
    """

    def __init__(
        self,
        processor: Callable[[Any], Any],
        *,
        num_workers: int,
        capacity: int,
        stage: str,
        rank: int = 0,
        start: HostWorkerStart | None = None,
        trace: PipelineTrace | None = None,
    ) -> None:
        if int(num_workers) < 0:
            raise ValueError("host worker count must be nonnegative")
        if int(capacity) < 1:
            raise ValueError("host pipeline capacity must be positive")
        if not stage:
            raise ValueError("host pipeline stage name is required")
        try:
            pickle.dumps(processor)
        except Exception as error:
            raise TypeError("host processor must be serializable for spawned workers") from error

        self._processor = processor
        self._num_workers = int(num_workers)
        self._capacity = int(capacity)
        self._stage = stage
        self._rank = int(rank)
        self._start = start if start is not None else HostWorkerStart()
        self._trace = trace if trace is not None else PipelineTrace()
        self._pending: deque[int] = deque()
        self._ready: dict[int, HostResult] = {}
        self._last_submitted = -1
        self._last_consumed = -1
        self._consumed = 0
        self._closed = False
        self._poisoned: HostWorkerError | None = None
        self._workers: list[mp.Process] = []
        self._task_queue: Queue | None = None
        self._result_queue: Queue | None = None
        self._error_queue: Queue | None = None

        if self._num_workers:
            context = self._start.context()
            self._task_queue = context.Queue(maxsize=self._capacity)
            self._result_queue = context.Queue(maxsize=self._capacity)
            self._error_queue = context.Queue(maxsize=self._capacity)
            for index in range(self._num_workers):
                worker = context.Process(
                    target=_host_worker_main,
                    args=(
                        self._processor,
                        self._task_queue,
                        self._result_queue,
                        self._error_queue,
                        self._stage,
                        self._rank,
                    ),
                    name=f"ncls-host-{stage}-{index}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)
            self._trace.gauge("host.workers", self._num_workers)

    @property
    def consumed_requests(self) -> int:
        return self._consumed

    @property
    def pending_requests(self) -> int:
        return len(self._pending)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("host pipeline is closed")
        if self._poisoned is not None:
            raise self._poisoned

    def _check_error(self) -> None:
        if self._error_queue is not None:
            try:
                logical_id, kind, message, child_traceback = self._error_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._poisoned = HostWorkerError(
                    stage=self._stage,
                    logical_id=int(logical_id),
                    rank=self._rank,
                    detail=f"{kind}: {message}\n{child_traceback}",
                )
                raise self._poisoned
        for worker in self._workers:
            if worker.exitcode is not None and worker.exitcode != 0:
                self._poisoned = HostWorkerError(
                    stage=self._stage,
                    logical_id=self._pending[0] if self._pending else None,
                    rank=self._rank,
                    detail=f"worker {worker.name} exited with code {worker.exitcode}",
                )
                raise self._poisoned

    def _publish_depth(self) -> None:
        self._trace.gauge("host.queue_depth", len(self._pending))
        self._trace.gauge("host.reorder_depth", len(self._ready))

    def submit(self, request: HostRequest, *, timeout: float | None = None) -> None:
        self._require_open()
        if request.logical_id <= self._last_submitted:
            raise ValueError("host request logical IDs must be strictly increasing")
        started = time.perf_counter_ns()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while len(self._pending) >= self._capacity:
            self._check_error()
            if deadline is not None and time.monotonic() >= deadline:
                self._trace.increment("host.backpressure")
                raise HostPipelineBackpressure(
                    f"host pipeline {self._stage!r} reached capacity {self._capacity}"
                )
            time.sleep(0.001)

        if not self._num_workers:
            try:
                with self._trace.measure("host.process"):
                    value = self._processor(request.payload)
            except BaseException as error:
                self._poisoned = HostWorkerError(
                    stage=self._stage,
                    logical_id=request.logical_id,
                    rank=self._rank,
                    detail=f"{type(error).__name__}: {error}",
                )
                raise self._poisoned from error
            self._ready[request.logical_id] = HostResult(
                request.logical_id, value, request.provenance
            )
        else:
            assert self._task_queue is not None
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                self._task_queue.put(
                    (request.logical_id, request.payload, dict(request.provenance)),
                    timeout=remaining,
                )
            except queue.Full as error:
                self._trace.increment("host.backpressure")
                raise HostPipelineBackpressure(
                    f"host pipeline {self._stage!r} task queue is full"
                ) from error
        self._pending.append(request.logical_id)
        self._last_submitted = request.logical_id
        self._trace.increment("host.submitted")
        self._trace.add_duration_ns("host.submit_wait", time.perf_counter_ns() - started)
        self._publish_depth()

    def _receive_one(self, timeout: float | None) -> None:
        self._check_error()
        if not self._num_workers:
            return
        assert self._result_queue is not None
        try:
            logical_id, payload, provenance, duration_ns = self._result_queue.get(
                timeout=timeout
            )
        except queue.Empty:
            self._check_error()
            raise
        logical_id = int(logical_id)
        if logical_id not in self._pending or logical_id in self._ready:
            raise RuntimeError("host worker returned an unknown or duplicate logical request")
        self._ready[logical_id] = HostResult(logical_id, payload, dict(provenance))
        self._trace.increment("host.completed")
        self._trace.add_duration_ns("host.process", int(duration_ns))
        self._publish_depth()

    def next_result(self, *, timeout: float | None = None) -> HostResult:
        self._require_open()
        if not self._pending:
            raise RuntimeError("host pipeline has no pending request")
        expected = self._pending[0]
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        started = time.perf_counter_ns()
        while expected not in self._ready:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                raise TimeoutError(
                    f"timed out waiting for logical request {expected} in {self._stage!r}"
                )
            try:
                self._receive_one(remaining if remaining is None else min(remaining, 0.05))
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for logical request {expected} in {self._stage!r}"
                    ) from None
        result = self._ready.pop(expected)
        self._pending.popleft()
        self._last_consumed = expected
        self._consumed += 1
        self._trace.increment("host.consumed")
        self._trace.add_duration_ns("host.consumer_wait", time.perf_counter_ns() - started)
        self._publish_depth()
        return result

    def drain(self, *, discard: bool = False, timeout: float | None = None) -> None:
        self._require_open()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while len(self._ready) < len(self._pending):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                raise TimeoutError(f"timed out draining host pipeline {self._stage!r}")
            try:
                self._receive_one(remaining if remaining is None else min(remaining, 0.05))
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out draining host pipeline {self._stage!r}"
                    ) from None
        if discard:
            discarded = len(self._pending)
            self._pending.clear()
            self._ready.clear()
            self._last_submitted = self._last_consumed
            self._trace.increment("host.discarded", discarded)
            self._publish_depth()

    def state_dict(self) -> Mapping[str, Any]:
        self._require_open()
        if self._pending:
            raise RuntimeError("host pipeline must be drained and consumed before checkpoint")
        return {
            "format_name": "ncls.host-pipeline",
            "format_version": 1,
            "stage": self._stage,
            "rank": self._rank,
            "consumed_requests": self._consumed,
            "last_consumed_logical_id": self._last_consumed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._require_open()
        if self._pending or self._consumed:
            raise RuntimeError("host pipeline state can only load into a fresh pipeline")
        required = {
            "format_name",
            "format_version",
            "stage",
            "rank",
            "consumed_requests",
            "last_consumed_logical_id",
        }
        if set(state) != required:
            raise ValueError("host pipeline state fields are invalid")
        if state["format_name"] != "ncls.host-pipeline" or int(
            state["format_version"]
        ) != 1:
            raise ValueError("unsupported host pipeline state format")
        if state["stage"] != self._stage or int(state["rank"]) != self._rank:
            raise ValueError("host pipeline state identity mismatch")
        consumed = int(state["consumed_requests"])
        last = int(state["last_consumed_logical_id"])
        if consumed < 0 or last < -1 or (consumed == 0) != (last == -1):
            raise ValueError("host pipeline consumed cursor is invalid")
        self._consumed = consumed
        self._last_consumed = last
        self._last_submitted = last

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, Any]:
        self._require_open()
        return self._trace.snapshot(reset=reset).to_dict()

    def close(self) -> None:
        if self._closed:
            return
        if self._poisoned is None:
            try:
                self.drain(discard=True, timeout=5.0)
            except (HostPipelineError, TimeoutError):
                pass
        if self._task_queue is not None:
            for _ in self._workers:
                try:
                    self._task_queue.put(_STOP, timeout=0.1)
                except queue.Full:
                    break
        for worker in self._workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
        for channel in (self._task_queue, self._result_queue, self._error_queue):
            if channel is not None:
                channel.close()
                channel.join_thread()
        self._closed = True

    def __enter__(self) -> "HostPipeline":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "HostPipeline",
    "HostPipelineBackpressure",
    "HostPipelineError",
    "HostRequest",
    "HostResult",
    "HostWorkerError",
    "HostWorkerStart",
]
