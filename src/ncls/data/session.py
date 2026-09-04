from __future__ import annotations

from typing import Any, Mapping

from .contracts import OnlineBatch, OnlineProducer


class SynchronousOnlineDataSession:
    """`num_workers=0` baseline around one rank-owned GPU/reference producer."""

    def __init__(self, producer: OnlineProducer, execution_plan_identity: str) -> None:
        if not isinstance(producer, OnlineProducer):
            raise TypeError("synchronous data session requires an OnlineProducer")
        if not execution_plan_identity:
            raise ValueError("data execution plan identity is required")
        self._producer = producer
        self._execution_plan_identity = execution_plan_identity
        self._consumed_batches = 0
        self._closed = False

    @property
    def consumed_batches(self) -> int:
        return self._consumed_batches

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

    def next_batch(self, request: Any) -> OnlineBatch:
        self._require_open()
        batch = self._producer.next_batch(request)
        self._consumed_batches += 1
        return batch

    def state_dict(self) -> Mapping[str, Any]:
        self._require_open()
        return {
            "format_name": "ncls.synchronous-online-data-session",
            "format_version": 1,
            "execution_plan_identity": self._execution_plan_identity,
            "consumed_batches": self._consumed_batches,
            "producer": self._producer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._require_open()
        required = {
            "format_name",
            "format_version",
            "execution_plan_identity",
            "consumed_batches",
            "producer",
        }
        if set(state) != required:
            raise ValueError("online data session state fields are invalid")
        if (
            state["format_name"] != "ncls.synchronous-online-data-session"
            or int(state["format_version"]) != 1
        ):
            raise ValueError("unsupported online data session state format")
        if state["execution_plan_identity"] != self._execution_plan_identity:
            raise ValueError("online data session execution plan identity mismatch")
        consumed = int(state["consumed_batches"])
        if consumed < 0:
            raise ValueError("online data session consumed cursor is invalid")
        producer_state = state["producer"]
        if not isinstance(producer_state, Mapping):
            raise ValueError("online data session producer state must be an object")
        self._producer.load_state_dict(producer_state)
        self._consumed_batches = consumed

    def end_iteration(self) -> None:
        self._require_open()
        self._producer.end_iteration()

    def profile_snapshot(self, *, reset: bool = False) -> Mapping[str, float]:
        self._require_open()
        return self._producer.profile_snapshot(reset=reset)

    def drain(self) -> None:
        self._require_open()
        # The synchronous baseline has no queued or in-flight batch.

    def native_assets(self) -> Any:
        self._require_open()
        return self._producer.native_assets()

    def close(self) -> None:
        if self._closed:
            return
        self._producer.close()
        self._closed = True

    def __enter__(self) -> "SynchronousOnlineDataSession":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()
