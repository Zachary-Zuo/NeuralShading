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
            artifacts={"display": str(panel)},
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


def test_tensorboard_hook_rejects_step_regression_and_nonzero_rank(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rank 0"):
        TensorBoardHook(tmp_path / "wrong-rank", rank=1)

    hook = TensorBoardHook(tmp_path / "run", rank=0)
    hook.handle(TrainingEvent("step-completed", 2, 0, 1, scalars={"loss": 1.0}))
    with pytest.raises(ValueError, match="step regressed"):
        hook.handle(TrainingEvent("step-completed", 1, 0, 1, scalars={"loss": 0.5}))
    hook.close()


def test_tensorboard_background_errors_are_reported_on_flush(tmp_path: Path) -> None:
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
    hook.handle(TrainingEvent("step-completed", 1, 0, 1, scalars={"loss": 1.0}))
    with pytest.raises(RuntimeError, match="TensorBoard writer failed"):
        hook.flush()
    with pytest.raises(RuntimeError, match="TensorBoard writer failed"):
        hook.close()
