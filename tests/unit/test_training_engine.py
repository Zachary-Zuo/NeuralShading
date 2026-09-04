from ncls.learning.training import (
    HookBinding,
    TrainingEngine,
    TrainingEventBus,
)
from tests.unit.test_training_runner_phase_graph import (
    _PhaseMethod,
    _config,
    _data_session,
    _plugin,
)


class _RecordingHook:
    def __init__(self) -> None:
        self.events = []

    def handle(self, event) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_training_engine_emits_fixed_lifecycle_without_method_specific_hooks() -> None:
    hook = _RecordingHook()
    bus = TrainingEventBus((HookBinding("record", hook, "fatal", False),))
    result = TrainingEngine(
        _plugin(_PhaseMethod()), _data_session(), _config(), event_bus=bus
    ).run()

    kinds = [event.kind for event in hook.events]
    assert kinds[0:2] == ["run-started", "phase-started"]
    assert kinds.count("phase-started") == 2
    assert kinds.count("step-completed") == 4
    assert kinds.count("validation-completed") == 2
    assert kinds.count("checkpoint-committed") == 2
    assert kinds[-1] == "run-completed"
    assert hook.events[-1].global_step == result.checkpoint.global_step == 4
    assert hook.events[-1].phase_name == "complete"
    assert not bus.failures
