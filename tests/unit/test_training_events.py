import pytest

from ncls.learning.training import HookBinding, TrainingEvent, TrainingEventBus


class _Hook:
    def __init__(self, *, fail: str | None = None, calls: list[str] | None = None) -> None:
        self.fail = fail
        self.calls = [] if calls is None else calls

    def handle(self, event: TrainingEvent) -> None:
        self.calls.append(f"handle:{event.kind}")
        if self.fail == "handle":
            raise ValueError("handle failed")

    def flush(self) -> None:
        self.calls.append("flush")
        if self.fail == "flush":
            raise ValueError("flush failed")

    def close(self) -> None:
        self.calls.append("close")
        if self.fail == "close":
            raise ValueError("close failed")


def _event(*, rank: int = 0) -> TrainingEvent:
    return TrainingEvent(
        "step-completed",
        global_step=3,
        rank=rank,
        world_size=2,
        phase_name="fit",
        scalars={"loss": 0.25},
    )


def test_event_bus_orders_hooks_skips_nonzero_rank_and_collects_diagnostic_failure() -> None:
    calls: list[str] = []
    first = _Hook(calls=calls)
    failing = _Hook(fail="handle", calls=calls)
    bus = TrainingEventBus(
        (
            HookBinding("rank-zero", first, "fatal", True),
            HookBinding("diagnostic", failing, "diagnostic", False),
        )
    )

    bus.emit(_event(rank=1))
    assert calls == ["handle:step-completed"]
    assert bus.failures[0].hook_name == "diagnostic"
    assert bus.failures[0].event_kind == "step-completed"
    bus.close()
    assert calls[-2:] == ["close", "close"]


def test_event_bus_propagates_fatal_hook_failure() -> None:
    bus = TrainingEventBus(
        (HookBinding("fatal", _Hook(fail="handle"), "fatal", False),)
    )
    with pytest.raises(RuntimeError, match="fatal training hook 'fatal'"):
        bus.emit(_event())


def test_training_event_rejects_nonfinite_scalars() -> None:
    with pytest.raises(ValueError, match="finite"):
        TrainingEvent(
            "step-completed", 1, 0, 1, scalars={"loss": float("nan")}
        )
