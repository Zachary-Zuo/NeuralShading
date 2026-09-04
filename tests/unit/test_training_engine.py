from typing import Any, Mapping

import pytest
import torch

from ncls.learning.method import TrainingInitializationRequest
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


class _InitializationMethod(_PhaseMethod):
    def __init__(self) -> None:
        self.initialization_calls = 0

    def initialization_requests(
        self, config: Mapping[str, Any]
    ) -> tuple[TrainingInitializationRequest, ...]:
        del config
        return (
            TrainingInitializationRequest(
                "fixture-calibration", "bootstrap", "evaluator", 3, 17, ("target_f",)
            ),
        )

    def initialize_training_state(
        self,
        model: torch.nn.Module,
        values: Mapping[str, Mapping[str, torch.Tensor]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.initialization_calls += 1
        target = values["fixture-calibration"]["target_f"]
        assert target.shape == (3, 3)
        assert metadata["schema"] == "ncls.train-only-initialization@1"
        with torch.no_grad():
            model.encoder.fill_(float(target.mean()))
        return {"fixture_calibration_count": int(target.shape[0])}


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


def test_fresh_initialization_precedes_forward_and_resume_does_not_reestimate() -> None:
    definition = _InitializationMethod()
    plugin = _plugin(definition)
    fresh = TrainingEngine(plugin, _data_session(), _config()).run(
        stop_at_step=0
    ).checkpoint
    assert definition.initialization_calls == 1
    assert fresh.query_stream_state["producer"]["request_count"] == {
        "initialization:fixture-calibration:evaluator": 2
    }
    resumed = TrainingEngine(plugin, _data_session(), _config()).run(
        resume=fresh, stop_at_step=1
    ).checkpoint
    assert definition.initialization_calls == 1
    assert torch.isfinite(resumed.model_state["encoder"]).all()


def test_invalid_stop_target_fails_before_online_initialization() -> None:
    definition = _InitializationMethod()
    with pytest.raises(ValueError, match="stop_at_step"):
        TrainingEngine(
            _plugin(definition), _data_session(), _config()
        ).run(stop_at_step=5)
    assert definition.initialization_calls == 0
