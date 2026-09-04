from types import SimpleNamespace

import pytest

from ncls.data import SynchronousOnlineDataSession


class _Producer:
    def __init__(self) -> None:
        self.cursor = 0
        self.closed = False
        self.iterations = 0
        self.device = SimpleNamespace(type="cpu")
        self.reference_program_identity = "reference-program"
        self.reference_execution_plan_identity = "reference-plan"
        self.native_asset_collection_identity = "asset-collection"
        self.query_stream_identity = "query-stream"
        self.source_contracts = ({"family_id": "test@1"},)
        self.source_snapshot_ids = ("source",)

    def next_batch(self, request):
        self.cursor += 1
        return SimpleNamespace(provenance={"request": request}, release=lambda: None)

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


def test_synchronous_online_data_session_roundtrips_cursor_and_delegates_lifecycle() -> None:
    producer = _Producer()
    session = SynchronousOnlineDataSession(producer, "plan-a")
    assert session.next_batch("first").provenance == {"request": "first"}
    session.end_iteration()
    state = session.state_dict()
    assert session.consumed_batches == 1
    assert producer.iterations == 1

    restored_producer = _Producer()
    restored = SynchronousOnlineDataSession(restored_producer, "plan-a")
    restored.load_state_dict(state)
    assert restored.consumed_batches == 1
    assert restored_producer.cursor == 1
    assert restored.profile_snapshot() == {"requests": 1.0}
    restored.drain()
    restored.close()
    restored.close()
    assert restored_producer.closed
    with pytest.raises(RuntimeError, match="closed"):
        restored.next_batch("late")


def test_synchronous_online_data_session_rejects_cross_plan_resume() -> None:
    source = SynchronousOnlineDataSession(_Producer(), "plan-a")
    target = SynchronousOnlineDataSession(_Producer(), "plan-b")
    with pytest.raises(ValueError, match="execution plan identity mismatch"):
        target.load_state_dict(source.state_dict())
