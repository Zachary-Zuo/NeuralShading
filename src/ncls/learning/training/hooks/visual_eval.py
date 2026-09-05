from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ncls.visual_eval.evaluator import VisualContext, VisualEvaluator
from ..events import TrainingEvent, TrainingEventBus

if TYPE_CHECKING:
    from torch import nn


class VisualEvalHook:
    def __init__(self, evaluator: VisualEvaluator, context: VisualContext, events: TrainingEventBus):
        self.evaluator = evaluator
        self.context = context
        self.events = events

    def __call__(self, model: nn.Module, step: int) -> None:
        if step % self.context.settings.interval_steps:
            return
        result = self.evaluator.evaluate(model, replace(self.context, step=step))
        if result is not None:
            self.events.emit(TrainingEvent(
                "visual-eval-completed", step, 0, 1,
                scalars={"visual_eval_seconds": result.elapsed_seconds},
                artifacts={tag: str(path) for tag, path in result.images.items()},
            ))
