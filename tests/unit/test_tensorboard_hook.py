from pathlib import Path

from PIL import Image
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ncls.learning.training import TrainingEvent
from ncls.learning.training.hooks import TensorBoardHook


def test_tensorboard_hook_writes_stable_scalar_tags_and_steps(tmp_path: Path) -> None:
    panel = tmp_path / "comparison.png"
    Image.new("RGB", (4, 2), color=(64, 128, 255)).save(panel)
    hook = TensorBoardHook(tmp_path, rank=0, flush_seconds=1, queue_capacity=16)
    hook.handle(
        TrainingEvent(
            "step-completed",
            10,
            0,
            1,
            phase_name="fit",
            scalars={
                "loss": 0.5,
                "learning_rate": 1.0e-3,
                "throughput/steps_per_second": 4.0,
                "pipeline/batch_prepare_seconds": 0.25,
                "reference/dispatch_seconds": 0.2,
                "memory/allocated_bytes": 1024.0,
            },
        )
    )
    hook.handle(
        TrainingEvent(
            "visual-eval-completed",
            10,
            0,
            1,
            artifacts={"comparison": str(panel)},
        )
    )
    hook.handle(
        TrainingEvent(
            "validation-completed",
            10,
            0,
            1,
            phase_name="fit",
            scalars={"loss": 0.6},
        )
    )
    hook.flush()
    hook.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert set(accumulator.Tags()["scalars"]) == {
        "train/learning_rate",
        "train/loss",
        "train/memory/allocated_bytes",
        "train/pipeline/batch_prepare_seconds",
        "train/reference/dispatch_seconds",
        "train/throughput/steps_per_second",
        "validation/loss",
    }
    assert accumulator.Scalars("train/loss")[0].step == 10
    assert accumulator.Scalars("validation/loss")[0].step == 10
    assert accumulator.Tags()["images"] == ["visual-eval/comparison"]
    assert accumulator.Images("visual-eval/comparison")[0].step == 10


def test_tensorboard_resume_discards_only_events_after_restored_checkpoint(tmp_path: Path) -> None:
    hook = TensorBoardHook(tmp_path)
    for step in (1, 2, 3):
        hook.handle(TrainingEvent("step-completed", step, 0, 1, scalars={"loss": float(step)}))
    hook.close()
    hook = TensorBoardHook(tmp_path, resume_step=1)
    hook.handle(TrainingEvent("step-completed", 2, 0, 1, scalars={"loss": 0.5}))
    hook.handle(TrainingEvent("step-completed", 3, 1, 2, scalars={"loss": 99.0}))
    hook.close()
    events = EventAccumulator(str(tmp_path)).Reload().Scalars("train/loss")
    assert [(event.step, event.value) for event in events] == [(1, 1.0), (2, 0.5)]


def test_tensorboard_writer_errors_reach_the_caller(tmp_path: Path) -> None:
    class FailingWriter:
        def __init__(self, **kwargs):
            del kwargs

        def add_scalar(self, *args):
            del args
            raise OSError("disk failed")

        def flush(self):
            pass

        def close(self):
            pass

    hook = TensorBoardHook(
        tmp_path, rank=0, writer_factory=FailingWriter, queue_capacity=4
    )
    with pytest.raises(OSError, match="disk failed"):
        hook.handle(TrainingEvent("step-completed", 1, 0, 1, scalars={"loss": 1.0}))
    hook.close()
