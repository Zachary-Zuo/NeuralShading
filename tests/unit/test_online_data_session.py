from types import SimpleNamespace

import pytest

from ncls.data import OnlineStepRequest, PipelineOnlineDataSession


class _Batch:
    lease = None

    def __init__(self, request: str, released: list[str]) -> None:
        self.provenance = {"request": request}
        self._request = request
        self._released = released

    def release(self) -> None:
        self._released.append(self._request)


class _Producer:
    def __init__(self) -> None:
        self.cursor = 0
        self.closed = False
        self.iterations = 0
        self.dispatches: list[tuple[int, ...]] = []
        self.released: list[str] = []
        self.device = SimpleNamespace(type="cpu")
        self.reference_program_identity = "reference-program"
        self.reference_execution_plan_identity = "reference-plan"
        self.native_asset_collection_identity = "asset-collection"
        self.query_stream_identity = "query-stream"
        self.source_contracts = ({"family_id": "test@1"},)
        self.source_snapshot_ids = ("source",)

    def produce_steps(self, requests: tuple[OnlineStepRequest, ...]):
        self.dispatches.append(tuple(request.logical_id for request in requests))
        result = []
        for request in requests:
            batches = {}
            for slot, route in request.routes.items():
                self.cursor += 1
                batches[slot] = _Batch(route.name, self.released)
            result.append(batches)
        return tuple(result)

    def prefetch_steps(self, requests: tuple[OnlineStepRequest, ...]):
        del requests

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])

    def end_iteration(self):
        self.iterations += 1

    def profile_snapshot(self, *, reset=False):
        result = {"requests": float(self.cursor)}
        if reset:
            self.cursor = 0
        return result

    def native_assets(self):
        return "assets"

    def close(self):
        self.closed = True


def _session(
    producer: _Producer,
    plan: str = "plan-a",
    *,
    ready: int = 2,
    batch_steps: int = 2,
) -> PipelineOnlineDataSession:
    return PipelineOnlineDataSession(
        producer,
        execution_plan_identity=plan,
        ready_capacity=ready,
        production_batch_steps=batch_steps,
    )


def _routes(*names: str):
    return {name: SimpleNamespace(name=name) for name in names}


def test_pipeline_session_batches_steps_and_roundtrips_idle_cursor() -> None:
    producer = _Producer()
    session = _session(producer)
    first_id = session.submit_step(_routes("a", "b"), boundary_id="train:p0")
    second_id = session.submit_step(_routes("a", "b"), boundary_id="train:p0")

    first = session.acquire_step(first_id)
    assert first.batches["a"].provenance == {"request": "a"}
    assert producer.dispatches == [(0, 1)]
    assert producer.iterations == 1
    first.release()
    second = session.acquire_step(second_id)
    second.release()
    session.drain()
    state = session.state_dict()
    assert session.consumed_batches == 4
    assert producer.released == ["b", "a", "b", "a"]

    restored_producer = _Producer()
    restored = _session(restored_producer)
    restored.load_state_dict(state)
    assert restored.consumed_batches == 4
    assert restored_producer.cursor == 4
    assert restored.profile_snapshot()["requests"] == 4.0
    restored.close()
    restored.close()
    assert restored_producer.closed
    with pytest.raises(RuntimeError, match="closed"):
        restored.submit_step(_routes("late"), boundary_id="late")


def test_pipeline_session_enforces_capacity_order_and_checkpoint_boundary() -> None:
    session = _session(_Producer(), ready=1, batch_steps=1)
    logical_id = session.submit_step(_routes("first"), boundary_id="train")
    with pytest.raises(RuntimeError, match="capacity"):
        session.submit_step(_routes("overflow"), boundary_id="train")
    with pytest.raises(RuntimeError, match="boundary requires"):
        session.state_dict()
    with pytest.raises(RuntimeError, match="order"):
        session.acquire_step(logical_id + 1)
    acquired = session.acquire_step(logical_id)
    with pytest.raises(RuntimeError, match="one acquired"):
        session.acquire_step(logical_id)
    acquired.release()
    session.drain()


def test_pipeline_session_does_not_batch_across_boundaries() -> None:
    producer = _Producer()
    session = _session(producer)
    first_id = session.submit_step(_routes("a"), boundary_id="train")
    second_id = session.submit_step(_routes("a"), boundary_id="validation")
    first = session.acquire_step(first_id)
    first.release()
    second = session.acquire_step(second_id)
    second.release()
    assert producer.dispatches == [(0,), (1,)]


def test_pipeline_session_rejects_cross_plan_resume() -> None:
    source = _session(_Producer(), "plan-a")
    target = _session(_Producer(), "plan-b")
    with pytest.raises(ValueError, match="execution plan identity mismatch"):
        target.load_state_dict(source.state_dict())
