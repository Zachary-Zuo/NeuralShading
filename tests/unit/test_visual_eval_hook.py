from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ncls.visual_eval import create_visual_evaluator
from ncls.visual_eval.evaluator import VisualContext, VisualResult
from ncls.learning.training.plan import VisualEvalSettings
from ncls.learning.training.events import HookBinding, TrainingEventBus
from ncls.learning.training.hooks import TensorBoardHook, VisualEvalHook


def test_linux_eval_command_does_not_probe_runtime_or_load_checkpoint(monkeypatch):
    from ncls import cli, launcher, visual_eval

    def unexpected(*args, **kwargs):
        raise AssertionError("Linux image eval must not prepare runtime or checkpoint")

    monkeypatch.setattr(visual_eval.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "process_environment", unexpected)
    monkeypatch.setattr(cli, "load_checkpoint", unexpected)
    assert launcher.main(["eval", "does-not-exist.pt"]) == 0


def test_eval_command_respects_checkpoint_disabled_setting(monkeypatch):
    from types import SimpleNamespace
    from ncls import cli
    from ncls.visual_eval.evaluator import NoVisualEvaluation

    plan = SimpleNamespace(hooks=SimpleNamespace(visual_eval=VisualEvalSettings(enabled=False)))
    monkeypatch.setattr(cli, "create_visual_evaluator", lambda settings: object() if settings.enabled else NoVisualEvaluation())
    monkeypatch.setattr(cli, "load_checkpoint", lambda path: SimpleNamespace(resolved_plan={}))
    monkeypatch.setattr(cli.ResolvedTrainingPlan, "from_dict", lambda value: plan)
    monkeypatch.setattr(cli, "get_method", lambda key: (_ for _ in ()).throw(AssertionError("disabled eval must not create a model")))
    assert cli._visual_eval(Path("disabled.pt"), None) == 0


def test_common_hook_linux_does_no_work_and_windows_result_reaches_tensorboard(tmp_path):
    settings = VisualEvalSettings(interval_steps=3)
    context = VisualContext(0, None, None, (), settings, tmp_path / "eval")
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    empty_events = TrainingEventBus(())
    hook = VisualEvalHook(create_visual_evaluator(settings, system="Linux"), context, empty_events)
    hook(object(), 3)
    assert not context.output.exists()
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    assert np.array_equal(before_numpy[1], np.random.get_state()[1])

    calls = []

    class Renderer:
        def evaluate(self, model, context):
            calls.append(context.step)
            context.output.mkdir()
            image = context.output / "comparison.png"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(image)
            return VisualResult({"comparison": image}, 0.25)

    events = TrainingEventBus((HookBinding("tensorboard", TensorBoardHook(tmp_path / "tensorboard"), "fatal", True),))
    hook = VisualEvalHook(Renderer(), context, events)
    hook(object(), 2)
    hook(object(), 3)
    events.close()
    assert calls == [3]
    accumulator = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert accumulator.Images("visual-eval/comparison")[0].step == 3
