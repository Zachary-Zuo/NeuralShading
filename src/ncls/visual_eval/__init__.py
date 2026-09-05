from __future__ import annotations

import platform
from .evaluator import NoVisualEvaluation, VisualContext, VisualEvaluator, VisualResult


def visual_evaluation_available(*, system: str | None = None) -> bool:
    return (platform.system() if system is None else system) == "Windows"


def create_visual_evaluator(settings, *, system: str | None = None) -> VisualEvaluator:
    if not settings.enabled or not visual_evaluation_available(system=system):
        return NoVisualEvaluation()
    from .windows import WindowsVisualEvaluator

    return WindowsVisualEvaluator()
